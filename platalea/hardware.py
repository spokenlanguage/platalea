from typing import Optional
import torch


def device(ordinal: Optional[int] = None):
    ordinal_str = ''
    if ordinal is not None:
        ordinal_str = f':{ordinal}'
    return torch.device("cuda" + ordinal_str if torch.cuda.is_available() else "cpu")
