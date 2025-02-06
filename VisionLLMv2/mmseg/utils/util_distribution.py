# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmseg import digit_version

dp_factory = {'cuda': MMDataParallel, 'cpu': MMDataParallel}

ddp_factory = {'cuda': MMDistributedDataParallel}


def build_dp(model, device='cuda', dim=0, *args, **kwargs):
    """build DataParallel module by device type.

    if device is cuda, return a MMDataParallel module; if device is mlu,
    return a MLUDataParallel module.

    Args:
        model (:class:`nn.Module`): module to be parallelized.
        device (str): device type, cuda, cpu or mlu. Defaults to cuda.
        dim (int): Dimension used to scatter the data. Defaults to 0.

    Returns:
        :class:`nn.Module`: parallelized module.
    """
    if device == 'cuda':
        model = model.cuda()
    elif device == 'mlu':
        assert digit_version(mmcv.__version__) >= digit_version('1.5.0'), \
                'Please use MMCV >= 1.5.0 for MLU training!'
        from mmcv.device.mlu import MLUDataParallel
        dp_factory['mlu'] = MLUDataParallel
        model = model.mlu()

    elif device == 'npu':
        assert digit_version(mmcv.__version__) >= digit_version('1.7.0'), \
                'Please use MMCV >= 1.7.0 for NPU training!'
        from mmcv.device.npu import NPUDataParallel
        torch.npu.set_device(kwargs['device_ids'][0])
        torch.npu.set_compile_mode(jit_compile=False)
        dp_factory['npu'] = NPUDataParallel
        model = model.npu()

    return dp_factory[device](model, dim=dim, *args, **kwargs)


def build_ddp(model, device='cuda', *args, **kwargs):
    """Build DistributedDataParallel module by device type.

    If device is cuda, return a MMDistributedDataParallel module;
    if device is mlu, return a MLUDistributedDataParallel module.

    Args:
        model (:class:`nn.Module`): module to be parallelized.
        device (str): device type, mlu or cuda.

    Returns:
        :class:`nn.Module`: parallelized module.

    References:
        .. [1] https://pytorch.org/docs/stable/generated/torch.nn.parallel.
                     DistributedDataParallel.html
    """
    assert device in ['cuda', 'mlu', 'npu'], 'Only available for cuda, '\
                                             'npu or mlu devices.'
    if device == 'cuda':
        model = model.cuda()
    elif device == 'mlu':
        assert digit_version(mmcv.__version__) >= digit_version('1.5.0'), \
            'Please use MMCV >= 1.5.0 for MLU training!'
        from mmcv.device.mlu import MLUDistributedDataParallel
        ddp_factory['mlu'] = MLUDistributedDataParallel
        model = model.mlu()

    elif device == 'npu':
        assert digit_version(mmcv.__version__) >= digit_version('1.7.0'), \
            'Please use MMCV >= 1.7.0 for NPU training!'
        from mmcv.device.npu import NPUDistributedDataParallel
        torch.npu.set_compile_mode(jit_compile=False)
        ddp_factory['npu'] = NPUDistributedDataParallel
        model = model.npu()

    return ddp_factory[device](model, *args, **kwargs)


def is_mlu_available():
    """Returns a bool indicating if MLU is currently available."""
    return hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()


def is_npu_available():
    """Returns a bool indicating if NPU is currently available."""
    return hasattr(torch, 'npu') and torch.npu.is_available()


def is_npu_support_full_precision() -> bool:
    """Returns True if npu devices support full precision training."""
    if not is_npu_available():
        return False
    import torch_npu.npu.utils as npu_utils
    version_of_support_full_precision = 220
    return npu_utils.get_soc_version() >= version_of_support_full_precision


def get_device():
    """Returns an available device, cpu, npu, cuda or mlu."""
    is_device_available = {
        'npu': is_npu_available(),
        'cuda': torch.cuda.is_available(),
        'mlu': is_mlu_available()
    }
    device_list = [k for k, v in is_device_available.items() if v]
    return device_list[0] if len(device_list) >= 1 else 'cpu'
