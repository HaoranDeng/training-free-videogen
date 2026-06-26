# training-free-videogen

Minimal Wan2.1 text-to-video generation with MonarchAttention.

This repository keeps only the inference path needed to run Wan2.1 with the
MonarchAttention module from MonarchRT. Large model weights and checkpoints are
not stored here.

## Layout

- `attention/`: SDPA fallback attention plus MonarchAttention.
- `wan/`: minimal Wan2.1 model, T5 text encoder, tokenizer, and VAE runtime.
- `pipeline/`: few-step text-to-video pipeline.
- `generate.py`: single-video-generation entry point.

## Run

Set `WAN_MODEL_ROOT` to a directory containing `Wan2.1-T2V-1.3B/`, then run:

```bash
python generate.py \
  --config configs/monarch_wan_fewstep.yaml \
  --checkpoint /path/to/self_forcing_dmd.pt \
  --prompt_path prompts/MovieGenVideoBench_extended.txt \
  --output_dir videos/monarch_wan \
  --num_videos 10 \
  --use_ema
```

On `amdhpc`, the helper script uses the existing `monarch_rt` conda environment,
the existing Wan model directory, and the existing self-forcing checkpoint:

```bash
sbatch amd_scripts/generate_10_monarch_wan.sh
```
