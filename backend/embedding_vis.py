import sys

from qdrant_client.models import FieldCondition, Filter, MatchValue

from . import store

pid = sys.argv[1] if len(sys.argv) > 1 else None
papers = store.get_papers()
print(f"{len(papers)} papers in SQLite:")
for p in papers:
    print(f"  {p['paper_id']:<28} {p['source']:<17} {str(p['year']):<5} {p['title'][:60]}")

flt = Filter(must=[FieldCondition(key="paper_id", match=MatchValue(value=pid))]) if pid else None
total = store._qd.count(store.COLL, count_filter=flt, exact=True).count
points, _ = store._qd.scroll(store.COLL, scroll_filter=flt, limit=5, with_payload=True, with_vectors=True)
print(f"\n{total} chunks in Qdrant{' for ' + pid if pid else ''}; showing {len(points)}")
for pt in points:
    v, pl = pt.vector, pt.payload
    print(f"\n--- chunk {pt.id}\npaper: {pl['paper_id']} | kind: {pl['kind']} | {len(pl['text'])} chars")
    print(pl["text"])
    print(f"vector: {len(v)} numbers, first 8 = {[round(x, 4) for x in v[:8]]}, length = {sum(x * x for x in v) ** 0.5:.3f}")