#!/usr/bin/env python3

from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup, allreduce, ReduceOp
from torch.optim import SGD
import torch

"""
1. time for allgather and compressor
2. modify gradient_allreduce to allreduce() implemented in Python
"""

class SGD_Record_Time_Optimizer(SGD):
    def __init__(
            self,
            params,
            lr: float = 1e-4,
            momentum_beta: float = 0.9,
            weight_decay: float = 0,
    ):
        """
        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate.
            momentum_beta: momentum constant.
            weight_decay: Weight decay (L2 penalty).
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum_beta:
            raise ValueError("Invalid momentum_beta value: {}".format(momentum_beta))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        # defaults = dict(lr=lr, momentum_beta=momentum_beta, weight_decay=weight_decay)
        super(SGD_Record_Time_Optimizer, self).__init__(params, lr=lr, momentum=momentum_beta, weight_decay=weight_decay)
        self.change_grad_time = 0.0
        self.allreduce_without_compressor_time = 0.0

    def __setstate__(self, state):
        super(SGD_Record_Time_Optimizer, self).__setstate__(state)

    def element_num(self, size):
        num = 1
        for i in range(len(size)):
            num *= size[i]
        return num


class Gradient_Allreduce_Record_Time_AlgorithmImpl(AlgorithmImpl):
    def __init__(
            self,
            process_group: BaguaProcessGroup,
            optimizer: SGD_Record_Time_Optimizer,
            record_time=True,
            hierarchical: bool = True,
            average: bool = True,
            device='cuda'
    ):
        super(Gradient_Allreduce_Record_Time_AlgorithmImpl, self).__init__(process_group)
        self.optimizer = optimizer
        self.record_time = record_time

        self.hierarchical = hierarchical
        self.average = average
        self.device = device
        self.parameters_num = 0
        for layer in self.optimizer.param_groups[0]['params']:
            self.parameters_num += layer.numel()

        self.send_tensor = torch.zeros(self.parameters_num, device=self.device, dtype=torch.float32)
        self.recv_tensor_list = torch.zeros(self.parameters_num, device=self.send_tensor.device,
                                            dtype=self.send_tensor.dtype)
        self.sign_momentum = torch.zeros(self.parameters_num, device=self.send_tensor.device, dtype=torch.float32)

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        """Given a :class:`~bagua.torch_api.data_parallel.BaguaDistributedDataParallel`, return a hook function that will be executed when the
        backward pass is done.

        Args:
            bagua_ddp: :class:`bagua.torch_api.data_parallel.BaguaDistributedDataParallel`.

        Returns:
            A function that takes no argument.
        """

        def compress_momentum_majority_vote():
            send_tensor_list = []
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    send_tensor_list.append(param.grad.view(-1))

            send_tensor_list = torch.cat(send_tensor_list)

            import time
            self.send_tensor.copy_(send_tensor_list)
            if self.record_time:
                start_time = time.time()
                allreduce(self.send_tensor, self.recv_tensor_list, ReduceOp.AVG)
                end_time = time.time()
                self.optimizer.allreduce_without_compressor_time += end_time - start_time
            else:
                allreduce(self.send_tensor, self.recv_tensor_list, ReduceOp.AVG)

            # change tensor.grad
            if self.record_time:
                start_time = time.time()
                cur_idx = 0
                for group in self.optimizer.param_groups:
                    for param in group["params"]:
                        num = self.optimizer.element_num(param.size())
                        param.grad.copy_(self.recv_tensor_list[cur_idx:cur_idx + num].view(param.size()))
                        cur_idx += num
                end_time = time.time()
                self.optimizer.change_grad_time += end_time - start_time
            else:
                cur_idx = 0
                for group in self.optimizer.param_groups:
                    for param in group["params"]:
                        num = self.optimizer.element_num(param.size())
                        param.grad.copy_(self.recv_tensor_list[cur_idx:cur_idx + num].view(param.size()))
                        cur_idx += num

        return compress_momentum_majority_vote


class Gradient_Allreduce_Record_Time_Algorithm(Algorithm):
    def __init__(self, optimizer: SGD_Record_Time_Optimizer, record_time=True,
                 hierarchical: bool = True, average: bool = True):
        self.optimizer = optimizer
        self.record_time = record_time
        self.hierarchical = hierarchical
        self.average = average

    def reify(self, process_group: BaguaProcessGroup) -> Gradient_Allreduce_Record_Time_AlgorithmImpl:
        return Gradient_Allreduce_Record_Time_AlgorithmImpl(
            process_group,
            optimizer=self.optimizer,
            record_time=self.record_time,
            hierarchical=self.hierarchical,
            average=self.average,
        )
