from typing import List

import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class MonarchWanPipeline(torch.nn.Module):
    """Few-step Wan2.1 text-to-video pipeline with optional Monarch self-attention."""

    def __init__(self, args, device):
        super().__init__()
        model_kwargs = dict(getattr(args, "model_kwargs", {}))
        image_or_video_shape = getattr(args, "image_or_video_shape", [1, 21, 16, 60, 104])

        self.generator = WanDiffusionWrapper(
            **model_kwargs,
            image_or_video_shape=image_or_video_shape,
        )
        self.generator.model.monarch_args = dict(getattr(args, "monarch_args", {}))
        self.text_encoder = WanTextEncoder(
            model_name=model_kwargs.get("model_name", "Wan2.1-T2V-1.3B"),
            model_root=model_kwargs.get("model_root", None),
        )
        self.vae = WanVAEWrapper(
            model_name=model_kwargs.get("model_name", "Wan2.1-T2V-1.3B"),
            model_root=model_kwargs.get("model_root", None),
        )

        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list,
            dtype=torch.long,
            device="cpu",
        )
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]
        if getattr(args, "warp_denoising_step", False):
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    @torch.inference_mode()
    def inference(self, noise: torch.Tensor, text_prompts: List[str], return_latents=False):
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        noisy_image_or_video = noise
        pred_image_or_video = None

        for index, current_timestep in enumerate(self.denoising_step_list[:-1]):
            _, pred_image_or_video = self.generator(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=torch.ones(noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep,
            )
            next_timestep = self.denoising_step_list[index + 1] * torch.ones(
                noise.shape[:2],
                dtype=torch.long,
                device=noise.device,
            )
            noisy_image_or_video = self.scheduler.add_noise(
                pred_image_or_video.flatten(0, 1),
                torch.randn_like(pred_image_or_video.flatten(0, 1)),
                next_timestep.flatten(0, 1),
            ).unflatten(0, noise.shape[:2])

        if pred_image_or_video is None:
            raise ValueError("denoising_step_list must contain at least two steps.")

        video = self.vae.decoder(pred_image_or_video)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if return_latents:
            return video, pred_image_or_video
        return video

