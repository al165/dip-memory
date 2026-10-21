"""Embedding server for the GPU machine: google/embeddinggemma-2 over HTTP, for text and images.

Standalone (no `dip` package needed): copy this folder to the GPU machine and run

    pip install -r requirements.txt          # plus a CUDA build of torch, see README.md
    uvicorn server:app --host 0.0.0.0 --port 11500

dip-memory then uses it with DIP_EMBED_BACKEND=remote DIP_EMBED_URL=http://<gpu-host>:11500.

The model is loaded exactly as dip's local `st` backend loads it (bf16, audio encoder skipped,
normalised output, same task prompts), so vectors from either are interchangeable and existing
builds such as data-embeddinggemma2/ stay valid.

Settings (env): EMBED_MODEL (google/embeddinggemma-2), EMBED_DEVICE (cuda if available, else
cpu), EMBED_BATCH (32), EMBED_LAZY=1 (load on first request; for tests), EMBED_TOKEN (if set,
clients must send `Authorization: Bearer <token>`).
"""
import base64
import io
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

MODEL = os.getenv("EMBED_MODEL", "google/embeddinggemma-2")
BATCH = int(os.getenv("EMBED_BATCH", "32"))
TOKEN = os.getenv("EMBED_TOKEN") or None
PROMPTS = {"SearchQuery", "Document"}  # dip's two task prompts; None = no prompt (images)

_model = None
_lock = threading.Lock()  # one encode at a time on the GPU


def get_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        device = os.getenv("EMBED_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        # bf16, never fp16: embeddinggemma's activations overflow fp16 into silent NaNs
        _model = SentenceTransformer(MODEL, device=device, model_kwargs={"torch_dtype": torch.bfloat16},
                                     config_kwargs={"audio_config": None})
    return _model


def check_token(authorization: str | None = Header(None)):
    if TOKEN and authorization != f"Bearer {TOKEN}":
        raise HTTPException(401, "missing or wrong bearer token")


def encode(inputs, prompt_name, dim):
    m = get_model()
    native = m.get_embedding_dimension()
    if dim is not None and dim > native:
        raise HTTPException(400, f"dim {dim} > model dimension {native}")
    with _lock:
        vecs = m.encode(inputs, prompt_name=prompt_name, batch_size=BATCH,
                        truncate_dim=dim if dim and dim < native else None, normalize_embeddings=True)
    return vecs.tolist()


@asynccontextmanager
async def lifespan(app):
    if os.getenv("EMBED_LAZY") != "1":
        get_model()  # fail at start, not on the first request
    yield


app = FastAPI(title="embeddinggemma-2 server", version="0.1", lifespan=lifespan)


class TextsIn(BaseModel):
    texts: list[str] = Field(..., max_length=2048)
    prompt_name: str | None = None
    dim: int | None = None


class ImagesIn(BaseModel):
    images: list[str] = Field(..., max_length=256, description="base64-encoded image files")
    dim: int | None = None


@app.get("/health")
def health():
    m = get_model()
    return {"model": MODEL, "device": str(m.device), "dim": m.get_embedding_dimension(),
            "dtype": "bfloat16", "prompts": sorted(PROMPTS)}


@app.post("/embed", dependencies=[Depends(check_token)])
def embed(body: TextsIn):
    if body.prompt_name is not None and body.prompt_name not in PROMPTS:
        raise HTTPException(400, f"prompt_name must be one of {sorted(PROMPTS)} or null")
    t0 = time.time()
    vecs = encode(body.texts, body.prompt_name, body.dim)
    return {"model": MODEL, "embeddings": vecs, "seconds": round(time.time() - t0, 3)}


@app.post("/embed-images", dependencies=[Depends(check_token)])
def embed_images(body: ImagesIn):
    from PIL import Image, UnidentifiedImageError
    try:
        imgs = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in body.images]
    except (ValueError, UnidentifiedImageError) as e:
        raise HTTPException(400, f"not an image: {e}") from e
    t0 = time.time()
    vecs = encode(imgs, None, body.dim)
    return {"model": MODEL, "embeddings": vecs, "seconds": round(time.time() - t0, 3)}
