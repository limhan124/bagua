#!/usr/bin/env python3

from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup, barrier, allgather
from torch.optim.optimizer import Optimizer
import torch
import math
from typing import List


class SignumOptimizer(Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-3,
            momentum_beta: float = 0.9,
            weight_decay: float = 0.0,
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
        super(SignumOptimizer, self).__init__(params, defaults)
        for group_id, group in enumerate(self.param_groups):
            for param_id, param in enumerate(group["params"]):
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(param, memory_format=torch.preserve_format)

    def __setstate__(self, state):
        super(SignumOptimizer, self).__setstate__(state)

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
            weight_decay = group["weight_decay"]

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

                # # adjust lr using "step" or using LARC
                # state["step"] += 1
                # step_id = state["step"]
                # bias_correction1 = 1 - momentum_beta ** step_id
                # lr = lr / bias_correction1

                # print(sign_grad)
                param.data.add_(sign_grad, alpha=-lr)
        return loss


class SignumAlgorithmImpl(AlgorithmImpl):
    def __init__(
            self,
            process_group: BaguaProcessGroup,
            signum_optimizer: SignumOptimizer,
            hierarchical: bool = True,
    ):
        """
        Args:
            process_group: The process group to work on.
            signum_optimizer: A SignumOptimizer initialized with model parameters.
            hierarchical: Enable hierarchical communication.
        """
        super(SignumAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.optimizer = signum_optimizer
        self.compressor = SignTensor()

    @property
    def optimizer_step_id(self):
        param = self.optimizer.param_groups[0]["params"][0]
        return self.optimizer.state[param].get("step", 0)

    def init_tensors(self, bagua_ddp: BaguaDistributedDataParallel):
        parameters = bagua_ddp.bagua_build_params()
        for idx, (name, param) in enumerate(parameters.__reversed__()):
            param._signum_name = name
            param._signum_idx = idx

        tensor_groups = []
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                # register momentum
                """
                set para.grad or set self.optimizer.state[param]["momentum"] ???
                """
                def set_momentum_fn(param, t):
                    self.optimizer.state[param]["momentum"] = t

                registered_tensor = param.bagua_ensure_grad().ensure_bagua_tensor(
                    param._signum_name,
                    bagua_ddp.bagua_module_name,
                    getter_closure=lambda param: self.optimizer.state[param]["momentum"],
                    setter_closure=set_momentum_fn,
                )
                tensor_groups.append(registered_tensor)
        tensor_groups.sort(key=lambda x: x._signum_idx)

        return tensor_groups

    def tensors_to_buckets(self, tensors: List[List[BaguaTensor]], do_flatten: bool) -> List[BaguaBucket]:
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

    # def init_operations(self, bagua_ddp: BaguaDistributedDataParallel, bucket: BaguaBucket, ):
    #     bucket.clear_ops()
    #
    #     def compress_momentum_majority_vote(*args):
    #         momentum_beta = self.optimizer.param_groups[0]["momentum_beta"]
    #         for tensor in bucket.tensors:
    #             # update momentum of the tensor
    #             tensor.bagua_getter_closure().mul_(momentum_beta).add_(tensor.grad, alpha=1 - momentum_beta)
    #             # compress the sign of the momentum
    #             send_tensor, size = self.compressor.packing(torch.sign(tensor.bagua_getter_closure()))
    #             recv_tensor_list = torch.zeros(
    #                 [self.process_group.get_global_communicator().nranks(), len(send_tensor)],
    #                 device=send_tensor.device, dtype=send_tensor.dtype)
    #             # allgather of the tensor
    #             allgather(send_tensor, recv_tensor_list)
    #             barrier(self.process_group)
    #             sign_momentum = self.compressor.majority_vote(recv_tensor_list)
    #             # update the grad
    #             tensor.grad = sign_momentum
    #
    #     bucket.append_python_op(compress_momentum_majority_vote, group=self.process_group)
    # def init_operations(self, bagua_ddp: BaguaDistributedDataParallel, bucket: BaguaBucket, ):


    def init_operations(self, bagua_ddp: BaguaDistributedDataParallel, bucket: BaguaBucket, ):
        bucket.clear_ops()

        def compress_momentum_majority_vote(*args):
            momentum_beta = self.optimizer.param_groups[0]["momentum_beta"]
            shapes = []
            send_tensor_list = []
            for tensor in bucket.tensors:
                # update momentum of the tensor
                tensor.bagua_getter_closure().mul_(momentum_beta).add_(tensor.grad, alpha=1 - momentum_beta)
                shapes.append(tensor.bagua_getter_closure().size())
                send_tensor_list.append(torch.sign(tensor.bagua_getter_closure()).view(-1))

            # compress the sign of the momentum of the whole bucket
            send_tensor_list = torch.cat(send_tensor_list)
            send_tensor, size = self.compressor.packing(send_tensor_list)
            recv_tensor_list = torch.zeros(
                [self.process_group.get_global_communicator().nranks(), len(send_tensor)], device=send_tensor.device,
                dtype=send_tensor.dtype
            )
            # allgather of the whole bucket
            allgather(send_tensor, recv_tensor_list)
            barrier(self.process_group)
            # majority_vote
            sign_momentum = self.compressor.majority_vote(recv_tensor_list, size)

            # change tensor.grad
            cur_idx = 0
            for tensor, shape in zip(bucket.tensors, shapes):
                num = self.compressor.element_num(shape)
                tensor.grad = sign_momentum[cur_idx:cur_idx + num].view(shape)
                cur_idx += num

        bucket.append_python_op(compress_momentum_majority_vote, group=self.process_group)

        # def calculate_momentum(*args):
        #     momentum_beta = self.optimizer.param_groups[0]["momentum_beta"]
        #     for tensor in bucket.tensors:
        #         tensor.bagua_getter_closure().mul_(momentum_beta).add_(tensor.grad, alpha=1 - momentum_beta)
        # bucket.append_python_op(calculate_momentum, group=self.process_group)
        #
        # bucket.append_centralized_synchronous_op(
        #     hierarchical=self.hierarchical,
        #     average=True,
        #     scattergather=True,
        #     compression="MinMaxUInt8",
        #     group=self.process_group,
        # )

    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook_momentum(parameter_name, parameter):
            assert (
                    parameter.bagua_backend_tensor().data_ptr()
                    == self.optimizer.state[parameter]["momentum"].data_ptr()
            ), "bagua backend tensor data_ptr should match _signum_momentum data_ptr"
            parameter.bagua_mark_communication_ready()

        return hook_momentum


class SignumAlgorithm(Algorithm):
    def __init__(self, signum_optimizer: SignumOptimizer, hierarchical: bool = True):
        """
        Args:
            signum_optimizer: A SignumOptimizer initialized with model parameters.
            hierarchical: Enable hierarchical communication.
        """
        self.hierarchical = hierarchical
        self.optimizer = signum_optimizer

    def reify(self, process_group: BaguaProcessGroup) -> SignumAlgorithmImpl:
        return SignumAlgorithmImpl(
            process_group,
            signum_optimizer=self.optimizer,
            hierarchical=self.hierarchical,
        )


class SignTensor:
    def __init__(self):
        self.eps = 1e-7

    def packing(self, src_tensor):
        src_tensor = torch.sign(src_tensor)
        src_tensor_size = src_tensor.size()
        src_tensor = src_tensor.view(-1)
        src_len = len(src_tensor)
        add_elm = 32 - (src_len % 32)
        if src_len % 32 == 0:
            add_elm = 0
        zero_tensor = torch.zeros([add_elm], dtype=torch.int32, device=src_tensor.device)
        src_tensor = torch.cat((src_tensor, zero_tensor), 0)
        src_tensor = src_tensor.view(32, -1)
        src_tensor = src_tensor.to(dtype=torch.int32)
        compressed_tensor = self.compress(src_tensor)
        compressed_tensor = compressed_tensor.to(dtype=torch.int32)
        return compressed_tensor, src_tensor_size

    def unpacking(self, compressed_tensor, src_tensor_size):
        src_element_num = self.element_num(src_tensor_size)
        add_elm = 32 - (src_element_num % 32)
        if src_element_num % 32 == 0:
            add_elm = 0
        compressed_tensor = compressed_tensor.int()
        dst_tensor = torch.zeros(src_element_num + add_elm, device=compressed_tensor.device, dtype=torch.int32)
        dst_tensor = dst_tensor.view(32, -1)
        dst_tensor = self.uncompress(compressed_tensor, dst_tensor)
        dst_tensor = dst_tensor.view(-1)
        dst_tensor = dst_tensor[:src_element_num]
        dst_tensor = dst_tensor.view(src_tensor_size)
        dst_tensor = dst_tensor.float()
        return dst_tensor

    def majority_vote(self, compressed_tensors, src_tensor_size):
        vote_res = []
        for compressed_tensor in compressed_tensors:
            vote_res.append(self.unpacking(compressed_tensor, src_tensor_size))
        vote_res = torch.stack(vote_res)
        vote_res = torch.sum(vote_res, 0)
        vote_res = torch.sign(vote_res)
        vote_res = vote_res.float()
        return vote_res

    def element_num(self, size):
        num = 1
        for i in range(len(size)):
            num *= size[i]
        return num

    def compress(self, src_tensor):
        src_tensor = src_tensor.permute(1, 0)
        # compressed_tensor: -1
        compressed_tensor = torch.zeros(src_tensor.size()[0], dtype=torch.int32, device=src_tensor.device)
        idx = 0
        # src_tensor: -1*32
        for tensor in src_tensor:
            mask = tensor == -1
            tensor[mask] = 0
            tensor[~mask] = 1
            bits = ""
            for i in range(32):
                bits += str(tensor[i].item())
            compressed_tensor[idx] = torch.tensor(int(bits, 2), dtype=torch.int32)
            idx += 1
        return compressed_tensor

    def uncompress(self, compressed_tensor, dst_tensor):
        idx = 0
        for item in compressed_tensor:
            decimal = item.item()
            if item.item() < 0:
                decimal += 2147483648 + 2147483648
            binary = '{:032b}'.format(decimal)
            # print(binary)
            for i in range(32):
                dst_tensor[i][idx] = torch.tensor(-1 if int(binary[i]) == 0 else 1, dtype=torch.int32)
            idx += 1
        return dst_tensor

# class SignTensor:
#     def __init__(self):
#         self.eps = 1e-7
#
#     def packing(self, src_tensor):
#         src_tensor = torch.sign(src_tensor)
#         src_tensor_size = src_tensor.size()
#         src_tensor = src_tensor.view(-1)
#         src_len = len(src_tensor)
#         add_elm = 32 - (src_len % 32)
#         if src_len % 32 == 0:
#             add_elm = 0
#         new_tensor = torch.zeros([add_elm], dtype=torch.float32, device=src_tensor.device)
#         src_tensor = torch.cat((src_tensor, new_tensor), 0)
#         src_tensor = src_tensor.view(32, -1)
#         src_tensor = src_tensor.to(dtype=torch.int32)
#         dst_tensor = self.compress(src_tensor)
#         dst_tensor = dst_tensor.to(dtype=torch.int32)
#         return dst_tensor, src_tensor_size
#
#     def unpacking(self, src_tensor, src_tensor_size):
#         src_element_num = self.element_num(src_tensor_size)
#         add_elm = 32 - (src_element_num % 32)
#         if src_element_num % 32 == 0:
#             add_elm = 0
#         src_tensor = src_tensor.int()
#         new_tensor = torch.ones(src_element_num + add_elm, device=src_tensor.device, dtype=torch.int32)
#         new_tensor = new_tensor.view(32, -1)
#         new_tensor = self.uncompress(src_tensor, new_tensor)
#         new_tensor = new_tensor.view(-1)
#         new_tensor = new_tensor[:src_element_num]
#         new_tensor = new_tensor.view(src_tensor_size)
#         new_tensor = - new_tensor.add_(-1)
#         new_tensor = new_tensor.float()
#         return new_tensor
#
#     def majority_vote(self, src_tensor_list, src_tensor_size):
#         src_element_num = self.element_num(src_tensor_size)
#         voter_num = len(src_tensor_list)
#         src_tensor = torch.stack(src_tensor_list)
#         src_tensor = src_tensor.view(-1)
#         full_size = 32 * len(src_tensor)
#         new_tensor = torch.ones(full_size, device=src_tensor.device, dtype=torch.int32)
#         new_tensor = new_tensor.view(32, -1)
#         new_tensor = self.uncompress(src_tensor, new_tensor)
#         new_tensor = - new_tensor.add_(-1)
#         # sum
#         new_tensor = new_tensor.permute(1, 0).contiguous().view(voter_num, -1)
#         new_tensor = torch.sum(new_tensor, 0)
#         new_tensor = new_tensor.view(-1, 32).permute(1, 0)
#         new_tensor = torch.sign(new_tensor)
#         new_tensor = new_tensor[:src_element_num]
#         new_tensor = new_tensor.view(src_tensor_size)
#         return new_tensor
#
#     def element_num(self, size):
#         num = 1
#         for i in range(len(size)):
#             num *= size[i]
#         return num
#
#     def compress(self, src_tensor):
#         a = torch.zeros(2, dtype=torch.float32, device=src_tensor.device)
#         a[0] = 1
#         a[1] = -2
#         mask_0 = a[0]
#         mask_1 = a[1]
#         src_tensor[0].__irshift__(mask_0)
#         src_tensor[0].__iand__(mask_0)
#
#         for i in range(32):
#             src_tensor[0].__ilshift__(mask_0)
#             src_tensor[0].__iand__(mask_1)
#             src_tensor[i].__irshift__(mask_0)
#             src_tensor[i].__iand__(mask_0)
#             src_tensor[0].__ior__(src_tensor[i])
#         return src_tensor[0]
#
#     def uncompress(self, src_tensor, dst_tensor):
#         a = torch.zeros(1, dtype=torch.float32, device=src_tensor.device)
#         a[0] = 1
#         mask_0 = a[0]
#         for i in range(32):
#             dst_tensor[i].__iand__(src_tensor)
#             dst_tensor[i].__ilshift__(mask_0)
#             src_tensor.__irshift__(mask_0)
#         return dst_tensor
