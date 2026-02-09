# nemo-subtitle-generator

Subtitle generation using NVIDIA NeMo ASR models (Parakeet TDT 0.6B v3, Canary 1B v2).

## Setup

```bash
uv sync
```

## Usage

Single file:
```bash
uv run gensub transcribe video.mp4
uv run gensub transcribe audio.wav --model canary --source-lang en
uv run gensub transcribe audio.wav --model nvidia/canary-1b-v2 --source-lang fr --target-lang en
uv run gensub transcribe video.mp4 --local-attention
uv run gensub transcribe audio.wav --text-only
```

Batch (recursive directory scan):
```bash
uv run gensub batch /path/to/videos
uv run gensub batch /path/to/videos --max-depth 5
uv run gensub batch /path/to/videos --model canary --source-lang en
```

Web UI:
```bash
uv run gensub gui
uv run gensub-gui
```

List models:
```bash
uv run gensub models
```

## Model Notes

- `--model` accepts `parakeet`, `canary`, or full model IDs.
- Parakeet supports `--source-lang auto` (default).
- Canary requires an explicit `--source-lang` (no auto mode).
- Canary translation is validated for `en <-> x` pairs; use same source/target for plain transcription.
- SRT generation requires model timestamps; if a model run cannot provide timestamps, use `--text-only`.
- For large files, use `--local-attention` with Parakeet to reduce VRAM usage.
