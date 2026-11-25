"""Quick connectivity test:  python -m backend.check_embed"""
from . import embeddings

v = embeddings.embed(["Example document chunk", "Example search query"])
print(len(v), "vectors of length", len(v[0]))
