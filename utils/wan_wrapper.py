import os
import types
from pathlib import Path
from typing import List, Optional

import torch

from utils.scheduler import FlowMatchScheduler, SchedulerInterface
from wan.modules.model import WanModel
from wan.modules.t5 import umt5_xxl
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.vae import _video_vae


def _resolve_model_dir(model_name: str, model_root: Optional[str] = None) -> Path:
    root = os.environ.get("WAN_MODEL_ROOT", model_root or "wan_models")
    return Path(root) / model_name


class WanTextEncoder(torch.nn.Module):
    def __init__(self, model_name: str = "Wan2.1-T2V-1.3B", model_root: Optional[str] = None) -> None:
        super().__init__()
        model_dir = _resolve_model_dir(model_name, model_root)

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device("cpu"),
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(
                model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
                map_location="cpu",
                weights_only=False,
            )
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=str(model_dir / "google" / "umt5-xxl"),
            seq_len=512,
            clean="whitespace",
        )

    @property
    def device(self):
        return next(self.text_encoder.parameters()).device

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0

        return {"prompt_embeds": context}


class WanVAECore(torch.nn.Module):
    def __init__(self, model_name: str = "Wan2.1-T2V-1.3B", model_root: Optional[str] = None):
        super().__init__()
        model_dir = _resolve_model_dir(model_name, model_root)
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.model = _video_vae(
            pretrained_path=str(model_dir / "Wan2.1_VAE.pth"),
            z_dim=16,
        ).eval().requires_grad_(False)

    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        device, dtype = latent.device, latent.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]
        output = [
            self.model.decode(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
            for u in zs
        ]
        output = torch.stack(output, dim=0)
        return output.permute(0, 2, 1, 3, 4)


class WanVAEDecoder:
    def __init__(self, vae: WanVAECore):
        self.vae = vae

    def __call__(self, latent: torch.Tensor):
        latent = latent.permute(0, 2, 1, 3, 4).contiguous(memory_format=torch.channels_last_3d)
        latent = latent.permute(0, 2, 1, 3, 4)
        return self.vae.decode_to_pixel(latent)


class WanVAEWrapper(torch.nn.Module):
    def __init__(self, model_name: str = "Wan2.1-T2V-1.3B", model_root: Optional[str] = None):
        super().__init__()
        self.vae = WanVAECore(model_name=model_name, model_root=model_root)
        self.vae = torch.nn.utils.convert_conv3d_weight_memory_format(self.vae, torch.channels_last_3d)
        self.vae = torch.nn.utils.convert_conv2d_weight_memory_format(self.vae, torch.channels_last)
        self.decoder = WanVAEDecoder(self.vae)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.vae = torch.nn.utils.convert_conv3d_weight_memory_format(self.vae, torch.channels_last_3d)
        self.vae = torch.nn.utils.convert_conv2d_weight_memory_format(self.vae, torch.channels_last)
        self.decoder = WanVAEDecoder(self.vae)
        return self

    @property
    def model(self):
        return self.vae.model


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "Wan2.1-T2V-1.3B",
        model_root: Optional[str] = None,
        timestep_shift: float = 5.0,
        image_or_video_shape=(1, 21, 16, 60, 104),
    ):
        super().__init__()
        model_dir = _resolve_model_dir(model_name, model_root)
        self.model = WanModel.from_pretrained(str(model_dir)).eval()
        self.uniform_timestep = True
        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.scheduler.set_timesteps(1000, training=True)

        video_shape = list(image_or_video_shape)
        video_shape = [video_shape[1], video_shape[-2], video_shape[-1]]
        for i in range(3):
            video_shape[i] //= self.model.patch_size[i]
        num_frames, frame_height, frame_width = video_shape
        self.frame_seq_len = frame_height * frame_width
        self.seq_len = num_frames * frame_height * frame_width
        self.post_init()

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps],
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(),
            dim=1,
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def forward(self, noisy_image_or_video: torch.Tensor, conditional_dict: dict, timestep: torch.Tensor) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]
        input_timestep = timestep[:, 0]
        flow_pred = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=input_timestep,
            context=prompt_embeds,
            seq_len=self.seq_len,
        ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])
        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise,
            scheduler,
        )
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0,
            scheduler,
        )
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0,
            scheduler,
        )
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        self.get_scheduler()

