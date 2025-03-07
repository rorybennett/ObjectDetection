import platform

import torch


def get_device_name():
    """
    Return the name of the device being used by torch (GPU name or CPU name).

    :return: Name of torch device.
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.device('cuda'))
    else:
        return platform.processor()
