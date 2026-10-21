"""The remote embedding backend against embed_server/ (with a fake model: no weights needed)."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dip import config, llm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "embed_server"))


class FakeModel:
    """Records calls; the vector encodes the prompt and input so the test can check routing."""
    device = "cpu"

    def __init__(self):
        self.calls = []

    def get_embedding_dimension(self):
        return 768

    def encode(self, inputs, prompt_name=None, batch_size=None, truncate_dim=None, normalize_embeddings=None):
        self.calls.append((prompt_name, truncate_dim, [type(x).__name__ for x in inputs]))
        dim = truncate_dim or 768
        out = np.zeros((len(inputs), dim), dtype=np.float32)
        out[:, 0 if prompt_name == "SearchQuery" else 1 if prompt_name == "Document" else 2] = 1
        return out


@pytest.fixture
def remote(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBED_LAZY", "1")
    monkeypatch.setenv("EMBED_TOKEN", "s3cret")
    server = importlib.reload(importlib.import_module("server"))
    fake = FakeModel()
    monkeypatch.setattr(server, "_model", fake)
    client = TestClient(server.app, headers={"Authorization": "Bearer s3cret"})
    monkeypatch.setattr(config, "EMBED_BACKEND", "remote")
    monkeypatch.setattr(config, "EMBED_MODEL", "google/embeddinggemma-2")
    monkeypatch.setattr(config, "EMBED_DIM", 256)
    monkeypatch.setattr(llm, "_remote", client)
    monkeypatch.setattr(llm, "_remote_checked", False)
    return server, fake, client


def test_text_prompts_and_dim(remote):
    _, fake, _ = remote
    q, d = llm.embed_one("a query", query=True), llm.embed(["doc one", "doc two"])
    assert len(q) == 256 and q[0] == 1 and all(v[1] == 1 for v in d)
    assert fake.calls == [("SearchQuery", 256, ["str"]), ("Document", 256, ["str", "str"])]


def test_images(remote, tmp_path):
    _, fake, _ = remote
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (8, 8), (i, 0, 0)).save(p)
        paths.append(str(p))
    vecs = llm.embed_images(paths, batch=2)
    assert len(vecs) == 3 and all(v[2] == 1 for v in vecs)  # no task prompt for images
    assert [c[2] for c in fake.calls] == [["Image", "Image"], ["Image"]]


def test_wrong_model_is_refused(remote, monkeypatch):
    monkeypatch.setattr(config, "EMBED_MODEL", "something-else")
    with pytest.raises(llm.LLMError, match="serves google/embeddinggemma-2"):
        llm.embed(["x"])


def test_token_and_bad_input(remote):
    server, _, client = remote
    anon = TestClient(server.app)
    assert anon.post("/embed", json={"texts": ["x"]}).status_code == 401
    assert client.post("/embed", json={"texts": ["x"], "prompt_name": "Nope"}).status_code == 400
    assert client.post("/embed-images", json={"images": ["bm90IGFuIGltYWdl"]}).status_code == 400
    assert client.get("/health").json()["model"] == "google/embeddinggemma-2"
