from __future__ import annotations
import collections
import io
from multiprocessing import process
import pickle

from bagua.torch_api.communication import (
    get_backend,
    broadcast,
    _get_default_group,
    BaguaProcessGroup,
)
import bagua
from bagua.torch_api.utils import to_bagua_datatype, StatisticalAverage
from bagua.torch_api.env import get_autotune_level, get_rank
from bagua.torch_api.model_parallel.moe import is_moe_param
from bagua.bagua_define import (
    TensorDeclaration,
    BaguaHyperparameter,
)
import gorilla
import time
import logging
import torch
import torch.nn
import itertools
from typing import List, Tuple, Optional


@gorilla.patches(torch.nn.Module, filter=lambda name, obj: "bagua" in name)
class BaguaModule:
    """
    This class patches `torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=module#torch.nn.Module>`_ with several methods to enable Bagua
    functionalities.

    :ivar bagua_optimizers: The optimizers passed in by :meth:`~bagua.torch_api.distributed.BaguaModule.with_bagua`.
    :vartype bagua_optimizers: List[torch.optim.Optimizer]

    :ivar bagua_algorithm: The algorithm passed in by :meth:`~bagua.torch_api.distributed.BaguaModule.with_bagua`.
    :vartype bagua_algorithm: bagua.torch_api.algorithms.Algorithm

    :ivar parameters_to_ignore: The parameter names in ``"{module_name}.{param_name}"`` format to ignore
        when calling ``self.bagua_build_params()``.
    :vartype parameters_to_ignore: List[str]

    :ivar bagua_train_step_counter: Number of iterations in training mode.
    :vartype bagua_train_step_counter: int

    :ivar bagua_buckets: All Bagua buckets in a list.
    :vartype bagua_buckets: List[bagua.torch_api.bucket.BaguaBucket]
    """

    __id_iter = itertools.count()

    def with_bagua(  # pytype: disable=module-attr
        self,
        optimizers: List[torch.optim.Optimizer],
        algorithm: "bagua.torch_api.algorithms.Algorithm",
        process_group: Optional[BaguaProcessGroup] = None,
    ) -> BaguaModule:
        r"""``with_bagua`` enables easy distributed data parallel training on a
        `torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=module#torch.nn.Module>`_.

        Arguments:
            optimizers: Optimizer(s) used by the
                module. It can contain one or more PyTorch optimizers.
            algorithm: Distributed algorithm
                used to do the actual communication and update.
            process_group: The process group to be used for distributed data all-reduction. If ``None``, the default process group,
                which is created by :func:`bagua.torch_api.init_process_group`, will be used. (default: ``None``)

        Returns:
            The original module, with Bagua related environments initialized.

        .. note::
            If we want to ignore some layers for communication, we can first check
            these layer's corresponding keys in the module's ``state_dict`` (they are
            in ``"{module_name}.{param_name}"`` format), then assign the list of
            keys to ``your_module._bagua_params_and_buffers_to_ignore``.

        Examples::

            >>> model = torch.nn.Sequential(
            ...      torch.nn.Linear(D_in, H),
            ...      torch.nn.ReLU(),
            ...      torch.nn.Linear(H, D_out),
            ...    )
            >>> optimizer = torch.optim.SGD(
            ...      model.parameters(),
            ...      lr=0.01,
            ...      momentum=0.9
            ...    )
            >>> model = model.with_bagua(
            ...      [optimizer],
            ...      GradientAllReduce()
            ...    )
        """

        self.bagua_ddp = bagua.torch_api.data_parallel.InnerDistributedDataParallel(
            self,
            optimizers,
            algorithm,
            process_group,
        )
        self.bagua_algorithm = self.bagua_ddp.bagua_algorithm
        self.bagua_optimizers = self.bagua_ddp.bagua_optimizers

        self._bagua_framework_hooks = []
        for hook in self.bagua_ddp.bagua_forward_pre_hooks:
            self._bagua_framework_hooks.append(
                self.register_forward_pre_hook(hook)
            )

        return self


_base = gorilla._get_base(BaguaModule)
_decorator_data = gorilla.get_decorator_data(_base)
for patch in _decorator_data.patches:
    gorilla.apply(patch)
