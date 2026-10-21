"""Thin client for a local Ollama server: chat generation (optionally JSON-schema
constrained) and embeddings. Nothing leaves the machine."""
import base64
import json
import math
import re

import httpx

from dip import config

# nomic-embed-text is trained with task prefixes; other models ignore them harmlessly
# but you can blank them via env if your model dislikes them.
DOC_PREFIX = config._env("DIP_EMBED_DOC_PREFIX", "search_document: ")
QUERY_PREFIX = config._env("DIP_EMBED_QUERY_PREFIX", "search_query: ")

_client = httpx.Client(base_url=config.OLLAMA_URL, timeout=config.LLM_TIMEOUT)


TEXT_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


class LLMError(RuntimeError):
    pass


def available() -> dict:
    """Report whether Ollama is reachable and the configured models are pulled."""
    try:
        tags = _client.get("/api/tags", timeout=5).json()
    except Exception as e:  # noqa: BLE001
        return {"ollama": False, "error": str(e)}
    names = {m["name"] for m in tags.get("models", [])}

    def has(m):
        return m in names or f"{m}:latest" in names

    return {"ollama": True, "llm": has(config.LLM_MODEL),
            "embed": config.EMBED_BACKEND != "ollama" or has(config.EMBED_MODEL)}


def generate(prompt: str, schema: dict | None = None, system: str | None = None,
             temperature: float = 0.2, images: list[bytes] | None = None, model: str | None = None,
             logprobs: bool = False):
    """Chat completion. With `schema`, Ollama constrains output to that JSON schema and the
    parsed object is returned; otherwise the text is returned. With `logprobs=True` returns
    (result, token_logprobs) so callers can read the model's confidence (see expected_score)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    msg = {"role": "user", "content": prompt}
    if images:
        msg["images"] = [base64.b64encode(b).decode() for b in images]
    messages.append(msg)
    body = {
        "model": model or config.LLM_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,  # qwen3 & co: skip the hidden reasoning trace, much faster
        "options": {"temperature": temperature, "num_ctx": config.LLM_NUM_CTX,
                    "num_predict": config.LLM_MAX_TOKENS},
    }
    # Free text also goes through JSON mode: some Ollama versions ignore think=False for qwen3
    # and the model then writes its reasoning straight into the answer, untagged. A JSON
    # schema forces the output to start with the answer itself.
    body["format"] = schema or TEXT_SCHEMA
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 10
    try:
        r = _client.post("/api/chat", json=body)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise LLMError(f"Ollama chat failed ({body['model']}): {e}") from e
    data = r.json()
    content = data["message"]["content"]
    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        if data.get("done_reason") == "length":
            raise LLMError(f"Model hit the {config.LLM_MAX_TOKENS}-token cap (DIP_LLM_MAX_TOKENS) "
                           f"before finishing its answer") from e
        raise LLMError(f"Model returned invalid JSON: {content[:200]}") from e
    if not schema:
        result = re.sub(r"<think>.*?</think>", "", result["text"], flags=re.S).strip()
    return (result, data.get("logprobs") or []) if logprobs else result


def digit_probs(token_logprobs: list, key: str, lo: int = 1, hi: int = 5) -> dict[int, float] | None:
    """{digit: probability} for an integer field in a JSON answer, read from the token logprobs
    at the position where the field's value is written (normalised over lo..hi)."""
    text = ""
    for tok in token_logprobs:
        t = tok["token"]
        if t.strip().isdigit() and re.search(rf'"{re.escape(key)}"\s*:\s*$', text):
            probs = {}
            for alt in tok.get("top_logprobs") or [tok]:
                d = alt["token"].strip()
                if d.isdigit() and lo <= int(d) <= hi:
                    probs[int(d)] = probs.get(int(d), 0.0) + math.exp(alt["logprob"])
            total = sum(probs.values())
            return {k: p / total for k, p in probs.items()} if total else None
        text += t
    return None


def expected_score(token_logprobs: list, key: str, lo: int = 1, hi: int = 5) -> float | None:
    """Probability-weighted rating for an integer field in a JSON answer.

    Small models collapse ratings onto one value ("3" for everything). Instead of the sampled
    digit, take the model's probability for each digit at that position and return the
    expectation, e.g. P(2)=.15 P(3)=.61 P(4)=.20 -> 3.04. This keeps ties apart (G-Eval trick).
    """
    probs = digit_probs(token_logprobs, key, lo, hi)
    return sum(k * p for k, p in probs.items()) if probs else None


_st_model = None


def _st():
    """Local sentence-transformers model (DIP_EMBED_BACKEND=st), loaded on first use. Audio is
    skipped; text and vision encoders share one space, so images and passages are comparable."""
    global _st_model
    if _st_model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        # bf16, never fp16: embeddinggemma's activations overflow fp16 into silent NaNs
        _st_model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE,
                                        model_kwargs={"torch_dtype": torch.bfloat16},
                                        config_kwargs={"audio_config": None})
    return _st_model


def _st_encode(inputs, prompt_name: str | None = None) -> list[list[float]]:
    m = _st()
    dim = config.EMBED_DIM if config.EMBED_DIM < m.get_embedding_dimension() else None
    vecs = m.encode(inputs, prompt_name=prompt_name, batch_size=16, truncate_dim=dim,
                    normalize_embeddings=True)
    return vecs.tolist()


# --- remote backend: embed_server/ running the same model on the GPU machine -----------------

_remote = None
_remote_checked = False


def _remote_post(path: str, body: dict) -> list[list[float]]:
    """POST to the embedding server. Checks once that it serves config.EMBED_MODEL, so vectors
    from a different model never get mixed into a build."""
    global _remote, _remote_checked
    if _remote is None:
        headers = {"Authorization": f"Bearer {config.EMBED_TOKEN}"} if config.EMBED_TOKEN else {}
        _remote = httpx.Client(base_url=config.EMBED_URL, timeout=config.LLM_TIMEOUT, headers=headers)
    try:
        if not _remote_checked:
            served = _remote.get("/health", timeout=10).json()["model"]
            if served != config.EMBED_MODEL:
                raise LLMError(f"embedding server at {config.EMBED_URL} serves {served}, "
                               f"expected {config.EMBED_MODEL}")
            _remote_checked = True
        r = _remote.post(path, json={**body, "dim": config.EMBED_DIM})
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise LLMError(f"embedding server failed ({config.EMBED_URL}{path}): {e}") from e
    return r.json()["embeddings"]


def embed_backend_info() -> str:
    """Where embeddings are computed, in words, for logs."""
    if config.EMBED_BACKEND == "remote":
        try:
            h = httpx.get(f"{config.EMBED_URL}/health", timeout=5).json()
            return f"remote {h['model']} at {config.EMBED_URL} on {h['device']}"
        except Exception as e:  # noqa: BLE001
            return f"remote {config.EMBED_MODEL} at {config.EMBED_URL} (NOT reachable: {e})"
    if config.EMBED_BACKEND == "st":
        return f"local {config.EMBED_MODEL} on this machine's {config.EMBED_DEVICE}"
    return f"Ollama {config.EMBED_MODEL} at {config.OLLAMA_URL} (text only)"


def remote_available() -> bool:
    try:
        return httpx.get(f"{config.EMBED_URL}/health", timeout=5).json().get("model") == config.EMBED_MODEL
    except Exception:  # noqa: BLE001
        return False


def embed_images(paths: list, batch: int = 8) -> list[list[float]]:
    """Vectors for image files, in the same space as text (gemma: st or remote backend)."""
    if config.EMBED_BACKEND == "remote":
        out = []
        for i in range(0, len(paths), batch):
            imgs = [base64.b64encode(open(p, "rb").read()).decode() for p in paths[i:i + batch]]
            out += _remote_post("/embed-images", {"images": imgs})
        return out
    if config.EMBED_BACKEND != "st":
        raise LLMError(f"image embeddings need DIP_EMBED_BACKEND=st or remote (have {config.EMBED_BACKEND})")
    from PIL import Image
    return _st_encode([Image.open(p).convert("RGB") for p in paths])


def embed(texts: list[str], query: bool = False, batch: int = 32) -> list[list[float]]:
    # gemma's own task prompts: "task: search result | query: " / "title: none | text: "
    prompt_name = "SearchQuery" if query else "Document"
    if config.EMBED_BACKEND == "st":
        return _st_encode(list(texts), prompt_name=prompt_name)
    if config.EMBED_BACKEND == "remote":
        texts, out = list(texts), []
        for i in range(0, len(texts), batch * 4):
            out += _remote_post("/embed", {"texts": texts[i:i + batch * 4], "prompt_name": prompt_name})
        return out
    prefix = QUERY_PREFIX if query else DOC_PREFIX
    out = []
    for i in range(0, len(texts), batch):
        chunk = [prefix + t for t in texts[i:i + batch]]
        try:
            r = _client.post("/api/embed", json={"model": config.EMBED_MODEL, "input": chunk})
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama embed failed ({config.EMBED_MODEL}): {e}") from e
        out.extend(r.json()["embeddings"])
    return out


def embed_one(text: str, query: bool = False) -> list[float]:
    return embed([text], query=query)[0]
