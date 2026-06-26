import numpy as np
import random
import torch
import imageio


def save_video(path, video, fps=16):
    """Save (T, H, W, C) frames in [0, 255] as mp4. Uses imageio (torchvision write_video is unavailable on some builds)."""
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()
    video = np.clip(video, 0, 255).astype(np.uint8)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for frame in video:
        writer.append_data(frame)
    writer.close()


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def merge_dict_list(dict_list):
    if len(dict_list) == 1:
        return dict_list[0]

    merged_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                merged_dict[k] = torch.stack([d[k] for d in dict_list], dim=0)
            else:
                merged_dict[k] = torch.cat([d[k] for d in dict_list], dim=0)
        else:
            # for non-tensor values, we just copy the value from the first item
            merged_dict[k] = v
    return merged_dict
