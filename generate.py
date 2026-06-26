import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from pipeline import MonarchWanPipeline
from utils.misc import save_video, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate videos with Wan2.1 + MonarchAttention.")
    parser.add_argument("--config", default="configs/monarch_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--output_dir", default="videos/monarch_wan")
    parser.add_argument("--num_videos", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    return parser.parse_args()


def load_prompts(path: str, limit: int):
    with open(path, encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if limit >= 0:
        prompts = prompts[:limit]
    return prompts


def load_generator_checkpoint(pipeline: MonarchWanPipeline, checkpoint_path: str, use_ema: bool):
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and ("generator" in state or "generator_ema" in state):
        key = "generator_ema" if use_ema and "generator_ema" in state else "generator"
        state = state[key]
    pipeline.generator.load_state_dict(state)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required for Wan2.1 generation.")

    set_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    config = OmegaConf.load(args.config)
    prompts = load_prompts(args.prompt_path, args.num_videos)
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompt_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Wan2.1 + MonarchAttention on {device} with dtype={dtype}.")
    pipeline = MonarchWanPipeline(config, device=device)
    load_generator_checkpoint(pipeline, args.checkpoint, args.use_ema)
    pipeline = pipeline.to(dtype=dtype)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.eval()

    latent_shape = list(config.image_or_video_shape)
    latent_shape[0] = args.num_samples

    for idx, prompt in tqdm(list(enumerate(prompts)), desc="Generating"):
        set_seed(args.seed + idx)
        noise = torch.randn(latent_shape, device=device, dtype=dtype)
        text_prompts = [prompt] * args.num_samples
        video, _ = pipeline.inference(noise=noise, text_prompts=text_prompts, return_latents=True)
        video = (255.0 * video.permute(0, 1, 3, 4, 2)).cpu()
        pipeline.vae.model.clear_cache()

        for sample_idx in range(args.num_samples):
            output_path = output_dir / f"{idx:03d}-{sample_idx:02d}.mp4"
            save_video(str(output_path), video[sample_idx], fps=args.fps)
            print(f"Saved {output_path}")

    print(f"Done. Generated {len(prompts) * args.num_samples} videos in {output_dir}.")


if __name__ == "__main__":
    os.environ.setdefault("MONARCH_FORCE_TORCH_RMSNORM", "1")
    main()

