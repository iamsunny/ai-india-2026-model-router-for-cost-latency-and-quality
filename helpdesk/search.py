"""The search_policy tool, copied from Agents Part 5 (RAG as a tool, not a pipeline).

Build the index once before using it:  python -m helpdesk.ingest
"""
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore

from helpdesk.ingest import INDEX, get_embeddings

MIN_SCORE = 0.30  # below this, a chunk is probably not about the question

_store = None


def get_store():
    global _store
    if _store is None:
        if not INDEX.exists():
            raise FileNotFoundError(
                "The handbook index is missing. Build it once from the repo root with: python -m helpdesk.ingest")
        _store = InMemoryVectorStore.load(str(INDEX), get_embeddings())
    return _store


@tool
def search_policy(query: str) -> dict:
    """Search the employee handbook. Use for any question about rules or policy:
    leave rules, work from home, holidays policy, payroll dates, expenses, notice period."""
    hits = get_store().similarity_search_with_score(query, k=4)
    results = [{"section": doc.metadata["source"], "text": doc.page_content, "score": round(score, 3)}
               for doc, score in hits if score >= MIN_SCORE]
    if not results:
        return {"results": [], "note": "Nothing relevant in the handbook."}
    return {"results": results}
