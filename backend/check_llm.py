"""Quick connectivity test:  python -m backend.check_llm"""
from . import llm

print(llm.chat([{"role": "user", "content": "Explain RAG in three steps."}], max_tokens=256))
