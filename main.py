import os
import argparse
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch


def main():
    parser = argparse.ArgumentParser(
        description="Research Paper RAG Assistant CLI"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing of all documents in the data folder",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single query and exit",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of retrieved chunks (default: 3)",
    )
    args = parser.parse_args()

    persist_dir = "faiss_store"

    if args.reindex:
        print("[INFO] Re-indexing documents from data/ directory...")
        docs = load_all_documents("data")
        if not docs:
            print("[WARN] No documents found in 'data/' to index.")
            return
        store = FaissVectorStore(persist_dir)
        store.build_from_documents(docs)
        print("[INFO] Indexing complete.\n")

    print("[INFO] Initializing RAG Search Engine...")
    rag = RAGSearch(persist_dir=persist_dir)

    if args.query:
        print(f"\n[QUERY]: {args.query}")
        answer = rag.search_and_summarize(args.query, top_k=args.top_k)
        print("\n" + "=" * 60)
        print(answer)
        print("=" * 60)
        return

    print("\n" + "=" * 60)
    print("Welcome to Academic Paper RAG Assistant CLI")
    print("Type your research questions below, or 'exit' / 'quit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("\n[Ask Paper Assistant] > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting RAG Assistant. Good luck with your interviews!")
                break

            print("\nSearching and synthesizing response...")
            answer = rag.search_and_summarize(query, top_k=args.top_k)
            print("\n" + "-" * 60)
            print(answer)
            print("-" * 60)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()

