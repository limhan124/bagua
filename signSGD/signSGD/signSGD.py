#!/usr/bin/env python3

import math
import bagua.torch_api as bagua
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup, allgather
from torch.optim.optimizer import Optimizer
import torch
from typing import List
import bit2byte

"""
1. time for allgather and compressor
2. modify gradient_allreduce to allreduce() implemented in Python
"""


class SignTensor:
    def __init__(self, bits, debug=False):
        self.eps = 1e-7
        self.bits = bits
        self.dtype = torch.int32 if self.bits == 32 else torch.uint8
        self.debug = debug

    def packing(self, src_tensor):
        src_tensor_size = src_tensor.size()
        src_tensor = torch.sign(src_tensor.view(-1))
        src_len = len(src_tensor)
        add_elm = self.bits - (src_len % self.bits)
        if src_len % self.bits == 0:
            add_elm = 0
        zero_tensor = torch.zeros([add_elm], dtype=self.dtype, device=src_tensor.device)
        src_tensor = torch.cat((src_tensor, zero_tensor), 0)
        # src_tensor: self.bits * -1
        src_tensor += 1
        src_tensor = src_tensor.view(self.bits, -1)
        src_tensor = src_tensor.to(dtype=self.dtype)
        compressed_tensor = bit2byte.compress(src_tensor)
        compressed_tensor = compressed_tensor.to(dtype=self.dtype)
        return compressed_tensor, src_tensor_size

    def unpacking(self, compressed_tensor, src_tensor_size):
        src_element_num = self.element_num(src_tensor_size)
        param_num = self.element_num(compressed_tensor.size()) * self.bits
        compressed_tensor = compressed_tensor.int()
        dst_tensor = torch.ones(param_num, device=compressed_tensor.device, dtype=self.dtype)
        # src_tensor: self.bits * -1
        dst_tensor = dst_tensor.view(self.bits, -1)
        dst_tensor = bit2byte.uncompress(compressed_tensor, dst_tensor)
        dst_tensor = dst_tensor.view(-1)
        dst_tensor = dst_tensor[:src_element_num]
        dst_tensor = dst_tensor.view(src_tensor_size)
        dst_tensor = dst_tensor.float()
        dst_tensor -= 1
        return dst_tensor

    def majority_vote(self, compressed_tensors, src_tensor_size):
        vote_num = compressed_tensors.size()[0]
        vote_size = self.element_num(compressed_tensors[0].size())
        src_element_num = self.element_num(src_tensor_size)
        compressed_tensor = compressed_tensors.view(-1)
        param_num = self.element_num(compressed_tensor.size()) * self.bits
        dst_tensor = torch.ones(param_num, device=compressed_tensor.device, dtype=self.dtype)
        # src_tensor: self.bits * -1
        dst_tensor = dst_tensor.view(self.bits, -1)
        dst_tensor = bit2byte.uncompress(compressed_tensor, dst_tensor)
        dst_tensor = dst_tensor.float()
        dst_tensor -= 1
        vote_res = [dst_tensor[:, i * vote_size:(i + 1) * vote_size].contiguous().view(-1) for i in range(vote_num)]
        vote_res = torch.stack(vote_res)
        vote_res = torch.sum(vote_res, 0)
        vote_res = torch.sign(vote_res)[:src_element_num]
        vote_res = vote_res.float()
        return vote_res

    def element_num(self, size):
        num = 1
        for i in range(len(size)):
            num *= size[i]
        return num


class SignSGDOptimizer(Optimizer):
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
        defaults = dict(lr=lr, momentum_beta=momentum_beta, weight_decay=weight_decay)
        super(SignSGDOptimizer, self).__init__(params, defaults)
        self.compress_time = 0.0
        self.allgather_time = 0.0
        self.uncompress_time = 0.0
        self.change_grad_time = 0.0
        self.allgather_without_compressor_time = 0.0
        for group_id, group in enumerate(self.param_groups):
            for param_id, param in enumerate(group["params"]):
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(param, memory_format=torch.preserve_format)

    def __setstate__(self, state):
        super(SignSGDOptimizer, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_id, group in enumerate(self.param_groups):
            lr = group["lr"]
            # here the momentum_beta is not used, because the update of momentum is performed before compression
            # momentum_beta=0 yields signSGD not signum
            momentum_beta = group['momentum_beta']
            weight_decay = group['weight_decay']

            for param_id, param in enumerate(group["params"]):
                if param.grad is None:
                    continue

                state = self.state[param]
                state["step"] += 1
                step_id = state["step"]

                grad = param.grad.data
                # sign of the gradient
                sign_grad = torch.sign(grad)
                # weight_decay
                if weight_decay != 0:
                    sign_grad.add_(param.data, alpha=weight_decay)

                param.data.add_(sign_grad, alpha=-lr)
        return loss


class SignSGDAlgorithmImpl(AlgorithmImpl):
    def __init__(
            self,
            process_group: BaguaProcessGroup,
            optimizer: SignSGDOptimizer,
            compress=True,
            print_memory_size=False,
            record_time=True,
            hierarchical: bool = True,
            average: bool = True,
            device='cuda'
    ):
        super(SignSGDAlgorithmImpl, self).__init__(process_group)
        self.optimizer = optimizer
        self.compress = compress
        self.print_memory_size = print_memory_size
        self.record_time = record_time
        self.send_tensor_type = torch.uint8 if self.compress else torch.float32
        self.hierarchical = hierarchical
        self.average = average
        self.compressor = SignTensor(bits=8)
        self.device = device
        self.print_time = bagua.get_rank()==0
        self.parameters_num = 0
        for layer in self.optimizer.param_groups[0]['params']:
            self.parameters_num += layer.numel()

        if self.compress:
            self.send_tensor = torch.zeros(int(math.ceil(self.parameters_num / 8)), device=self.device,
                                           dtype=self.send_tensor_type)
            self.recv_tensor_list = torch.zeros(
                [self.process_group.get_global_communicator().nranks(), len(self.send_tensor)],
                device=self.send_tensor.device, dtype=self.send_tensor.dtype
            )
            self.sign_momentum = torch.zeros(self.parameters_num, device=self.send_tensor.device, dtype=torch.float32)
        else:
            self.send_tensor = torch.zeros(self.parameters_num, device=self.device, dtype=self.send_tensor_type)
            self.recv_tensor_list = torch.zeros(
                [self.process_group.get_global_communicator().nranks(), len(self.send_tensor)],
                device=self.send_tensor.device, dtype=self.send_tensor.dtype
            )
            self.sign_momentum = torch.zeros(self.parameters_num, device=self.send_tensor.device, dtype=torch.float32)

    def tensors_to_buckets(
            self, tensors: List[List[BaguaTensor]], do_flatten: bool
    ) -> List[BaguaBucket]:
        bagua_buckets = []
        for idx, bucket in enumerate(tensors):
            bagua_bucket = BaguaBucket(
                bucket,
                flatten=do_flatten,
                name=str(idx),
                alignment=self.process_group.get_global_communicator().nranks(),
            )
            bagua_buckets.append(bagua_bucket)
        return bagua_buckets

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
            momentum_beta = self.optimizer.param_groups[0]["momentum_beta"]
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    self.optimizer.state[param]["momentum"].mul_(momentum_beta).add_(param.grad, alpha=1 - momentum_beta)
                    send_tensor_list.append(torch.sign(self.optimizer.state[param]["momentum"].data).view(-1))

            send_tensor_list = torch.cat(send_tensor_list)

            import time
            if self.compress:
                if self.print_memory_size:
                    print("original_tensor_size(bytes):", send_tensor_list.element_size() * send_tensor_list.nelement())

                if self.record_time:
                    # compress send_tensor
                    start_time = time.time()
                    send_tensor, _size = self.compressor.packing(send_tensor_list)
                    end_time = time.time()
                    self.optimizer.compress_time += end_time - start_time
                    self.send_tensor.copy_(send_tensor)

                    # allgather of all the params
                    start_time = time.time()
                    allgather(self.send_tensor, self.recv_tensor_list)
                    end_time = time.time()
                    self.optimizer.allgather_time += end_time - start_time

                    # majority_vote/uncompress
                    start_time = time.time()
                    self.sign_momentum = self.compressor.majority_vote(self.recv_tensor_list, _size)
                    end_time = time.time()
                    self.optimizer.uncompress_time += end_time - start_time
                else:
                    # compress send_tensor
                    send_tensor, _size = self.compressor.packing(send_tensor_list)
                    self.send_tensor.copy_(send_tensor)
                    # allgather of all the params
                    allgather(self.send_tensor, self.recv_tensor_list)
                    # majority_vote/uncompress
                    self.sign_momentum = self.compressor.majority_vote(self.recv_tensor_list, _size)

                if self.print_memory_size:
                    print("send_tensor_size(bytes):", self.send_tensor.element_size() * self.send_tensor.nelement())

            else:
                self.send_tensor.copy_(send_tensor_list)
                if self.record_time:
                    start_time = time.time()
                    allgather(self.send_tensor, self.recv_tensor_list)
                    end_time = time.time()
                    self.optimizer.allgather_without_compressor_time += end_time - start_time
                    self.sign_momentum = torch.sign(torch.sum(self.recv_tensor_list, 0)).float()
                else:
                    allgather(self.send_tensor, self.recv_tensor_list)
                    self.sign_momentum = torch.sign(torch.sum(self.recv_tensor_list, 0)).float()

            # # compress send_tensor
            # start_time = time.time()
            # send_tensor, _size = self.compressor.packing(send_tensor_list)
            # end_time = time.time()
            # self.optimizer.compress_time += end_time - start_time
            # self.send_tensor.copy_(send_tensor)
            #
            # # allgather of all the params
            # start_time = time.time()
            # allgather(self.send_tensor, self.recv_tensor_list)
            # end_time = time.time()
            # self.optimizer.allgather_time += end_time - start_time
            #
            # # majority_vote
            # start_time = time.time()
            # self.sign_momentum = self.compressor.majority_vote(self.recv_tensor_list, _size)
            # end_time = time.time()
            # self.optimizer.uncompress_time += end_time - start_time

            # change tensor.grad
            if self.record_time:
                start_time = time.time()
                cur_idx = 0
                for group in self.optimizer.param_groups:
                    for param in group["params"]:
                        num = self.compressor.element_num(param.size())
                        param.grad.copy_(self.sign_momentum[cur_idx:cur_idx + num].view(param.size()))
                        cur_idx += num
                end_time = time.time()
                self.optimizer.change_grad_time += end_time - start_time
            else:
                cur_idx = 0
                for group in self.optimizer.param_groups:
                    for param in group["params"]:
                        num = self.compressor.element_num(param.size())
                        param.grad.copy_(self.sign_momentum[cur_idx:cur_idx + num].view(param.size()))
                        cur_idx += num
            # if self.print_time:
            #     print("allgather_without_compressor_time", self.optimizer.allgather_without_compressor_time)
        return compress_momentum_majority_vote


class SignSGDAlgorithm(Algorithm):
    def __init__(self, optimizer: SignSGDOptimizer, compress=True, print_memory_size=False, record_time=True,
                 hierarchical: bool = True, average: bool = True):
        self.optimizer = optimizer
        self.compress = compress
        self.print_memory_size = print_memory_size
        self.record_time = record_time
        self.hierarchical = hierarchical
        self.average = average

    def reify(self, process_group: BaguaProcessGroup) -> SignSGDAlgorithmImpl:
        return SignSGDAlgorithmImpl(
            process_group,
            optimizer=self.optimizer,
            compress=self.compress,
            print_memory_size=self.print_memory_size,
            record_time=self.record_time,
            hierarchical=self.hierarchical,
            average=self.average,
        )
