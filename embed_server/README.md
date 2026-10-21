# embeddinggemma-2 server (GPU machine)

Serves `google/embeddinggemma-2` over HTTP so dip-memory can embed text and images on the GPU
machine instead of the laptop CPU (~2 s per query, ~1 min per image there). It loads the model
exactly as dip's local `st` backend does, so its vectors match the existing gemma build
(checked: cosine ≥ 0.9999 for passages, ≥ 0.998 for images against `data-embeddinggemma2/`).

Ollama can't do this yet: its embeddinggemma-2 support (v0.40.0) runs on MLX, i.e. Apple Silicon only.

## Install (once)

Copy this folder to the GPU machine, then in a fresh virtualenv:

```bash
# a CUDA build of torch; pick the index matching the driver (see `nvidia-smi`, top right)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
hf auth login        # only if the model download asks for it
```

The first start downloads the model (~1.5 GB) to `~/.cache/huggingface/`. In bf16 it needs about
2 GB of VRAM next to whatever Ollama has loaded.

## Run

```bash
uvicorn server:app --host 0.0.0.0 --port 11500
# optional: EMBED_TOKEN=<secret> to require `Authorization: Bearer <secret>`
```

Check it from the laptop: `curl http://192.168.0.15:11500/health` should report
`"device": "cuda:0"`. If it says `cpu`, torch was installed without CUDA.

To keep it running across reboots, a systemd user service:

```ini
# ~/.config/systemd/user/embed-server.service
[Unit]
Description=embeddinggemma-2 server
[Service]
WorkingDirectory=%h/embed_server
ExecStart=%h/embed_server/.venv/bin/uvicorn server:app --host 0.0.0.0 --port 11500
Restart=on-failure
[Install]
WantedBy=default.target
```

`systemctl --user enable --now embed-server`, plus `loginctl enable-linger $USER` so it runs
without a login session.

## Use from dip-memory

In `dip-memory/.env`:

```
DIP_EMBED_BACKEND=remote
DIP_EMBED_URL=http://192.168.0.15:11500
# DIP_EMBED_TOKEN=<secret>   if the server sets EMBED_TOKEN
```

`remote` is a drop-in for `st`: same model, same vectors, same task prompts. The client refuses
to run if the server serves a different model than `DIP_EMBED_MODEL`, so vectors from two
models never get mixed in one build.

## API

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | | `model`, `device`, `dim` |
| `POST /embed` | `texts`, `prompt_name` (`SearchQuery` for queries, `Document` for passages), `dim` | `embeddings` |
| `POST /embed-images` | `images` (base64 file bytes), `dim` | `embeddings` |

Vectors are L2-normalised; `dim` below 768 truncates (Matryoshka).
