from typing import Optional
import torch


_device = None


def set_device(device: Optional[str]):
    global _device
    _device = device


def device(ordinal: Optional[int] = None):
    """Return a device.

    By default, if available, it returns a GPU id string. Optionally, the user
    can specify the ordinal identifying a specific GPU. If GPUs are not available,
    it will return a CPU string.

    It is also possible to use set_device to set a custom device string. If set,
    i.e. if not None (the default value), this value is used.

    This function can only be used by models that run on a single device.
    """
    global _device
    if _device is not None:
        return torch.device(_device)

    ordinal_str = ''
    if ordinal is not None:
        ordinal_str = f':{ordinal}'
    return torch.device("cuda" + ordinal_str if torch.cuda.is_available() else "cpu")
