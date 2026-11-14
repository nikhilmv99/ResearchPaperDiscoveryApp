import argparse

from research_analyzer.config import Settings
from research_analyzer.services.embedding import EmbeddingService
from research_analyzer.services.llm import LLMService
from research_analyzer.services.retrieval import RAGRetriever
from research_analyzer.stores.chroma_store import ChromaVectorStore
from research_analyzer.pipelines.ingestion import PDFIngestionPipeline
from research_analyzer.agents.analysis import AnalysisAgent
from research_analyzer.agents.comparison import ComparisonAgent


def main():
    parser = argparse.ArgumentParser(
        description="Research paper ingestion and analysis"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Index PDFs from a folder")
    ingest.add_argument("folder")

    search = sub.add_parser("search", help="Retrieve matching excerpts")
    search.add_argument("question")
    search.add_argument("--source")
    search.add_argument("--top-k", type=int, default=5)

    ask = sub.add_parser("ask", help="Answer using retrieved evidence and an LLM")
    ask.add_argument("question")
    ask.add_argument("--source")
    ask.add_argument("--top-k", type=int, default=6)

    compare = sub.add_parser("compare", help="Compare at least two indexed PDFs")
    compare.add_argument("question")
    compare.add_argument("--sources", nargs="+", required=True)

    args = parser.parse_args()
    settings = Settings.from_env()
    store = ChromaVectorStore(directory=settings.chroma_path)
    embedder = EmbeddingService(settings)

    if args.command == "ingest":
        print(PDFIngestionPipeline(embedder, store).ingest(args.folder))
        return

    retriever = RAGRetriever(embedder, store)
    if args.command == "search":
        for passage in retriever.retrieve(args.question, args.top_k, args.source):
            print(
                f"{passage.source} | page {passage.page} | "
                f"distance {passage.distance:.3f}\n{passage.text[:850]}\n"
            )
        return

    llm = LLMService(settings)
    if args.command == "ask":
        answer = AnalysisAgent(retriever, llm).answer(
            args.question, args.source, args.top_k
        )
    else:
        answer = ComparisonAgent(retriever, llm).compare(
            args.question, args.sources
        )

    print(answer.text)
    if answer.passages:
        print("\nEvidence references:")
        for i, passage in enumerate(answer.passages, 1):
            print(f"[S{i}] {passage.source}, page {passage.page}")


if __name__ == "__main__":
    main()
