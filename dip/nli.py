"""Natural Language Inference with a small local cross-encoder.

Given (premise, hypothesis) it returns probabilities that the premise ENTAILS, CONTRADICTS or
is NEUTRAL to the hypothesis. This is how the claim graph decides "supports" vs "counters":
two passages can be very similar in embedding space and still disagree.
"""
import functools

import numpy as np

from dip import config


@functools.lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder  # heavy import, load on first use

    m = CrossEncoder(config.NLI_MODEL, device="cpu")
    labels = {i: l.lower() for i, l in m.model.config.id2label.items()}
    return m, labels


def score(pairs: list[tuple[str, str]], batch_size: int = 32) -> list[dict]:
    """[(premise, hypothesis)] -> [{'entailment': p, 'contradiction': p, 'neutral': p}]"""
    if not pairs:
        return []
    model, labels = _model()
    logits = np.asarray(model.predict(pairs, batch_size=batch_size, show_progress_bar=False))
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return [{labels[j]: float(row[j]) for j in range(row.shape[0])} for row in probs]
