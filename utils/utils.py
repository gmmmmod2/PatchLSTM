from typing import Tuple, Dict
import torch

def _safe_torch_load(path, map_location="cpu"):
    """
    Use weights_only=True when available (PyTorch >= 2.4), 
    fallback to default for older versions.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)  # new API
    except TypeError:
        return torch.load(path, map_location=map_location)  # fallback for older torch

def _normalize_state_dict_keys(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Align 'module.' prefix consistently between checkpoint and current model.
    """
    is_dp = isinstance(model, torch.nn.DataParallel)
    keys = list(state.keys())
    has_module = any(k.startswith("module.") for k in keys)

    if is_dp and not has_module:
        # current model is DP, ckpt是非DP：给ckpt键名加上'module.'
        state = {f"module.{k}": v for k, v in state.items()}
    elif (not is_dp) and has_module:
        # 当前非DP，ckpt是DP：去掉前缀
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state
