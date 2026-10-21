# embeddinggemma-2 vs nomic-embed-text on the existing nodes

Clusters, themes and sections were built from nomic vectors; the gemma build is the same corpus re-embedded. Higher = the existing structure holds together better in that space.

## Structure

| measure | nomic | gemma |
|---|---|---|
| silhouette (chunks by cluster, cosine) | 0.084 | 0.045 |
| chunk → own cluster centroid nearest | 1.000 | 0.870 |
| chunk → own cluster node nearest | 0.520 | 0.466 |
| cluster → own theme node nearest | 0.964 | 0.963 |
| chunk → own section nearest | 0.461 | 0.513 |

nomic's centroid score is 1.000 by construction: k-means assigned each chunk to its nearest nomic centroid. The gemma figure is the meaningful one: how many chunks would stay put.

## Neighbours

729 chunks matched across builds. Jaccard of the 10 nearest chunks: mean 0.34, median 0.33; same single nearest neighbour for 42%.

## Retrieval (provocations gold set, experiments/compare_embeddings.py)

```
ollama:nomic-embed-text (768d) on data, 40 provocations, 20 ms/query embed
  hybrid  recall@1  72%  recall@5  92%  MRR 0.810
  dense   recall@1  78%  recall@5  95%  MRR 0.852
st:google/embeddinggemma-2 (768d) on data-embeddinggemma2, 40 provocations, 1971 ms/query embed
  hybrid  recall@1  72%  recall@5  95%  MRR 0.817
  dense   recall@1  75%  recall@5  98%  MRR 0.848
```

## Images (experiments/image_probe.py)

grounding: nearest passage from the image's own PDF for 74/80 images (92%); within top 5 for 79/80 (99%); 8 sources in the DB
