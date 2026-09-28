import sys
from src.search import RAGSearch


def run():
    print("=" * 60)
    print("Academic Research Paper RAG Assistant")
    print("=" * 60)

    try:
        rag_search = RAGSearch()
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        return

    query = sys.argv[1] if len(sys.argv) > 1 else "What is the attention mechanism in Transformers?"
    print(f"\n[QUERY]: {query}\n")
    print("[INFO] Retrieving relevant contexts and generating response...\n")
    
    result = rag_search.search_and_summarize(query, top_k=3)
    print("=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    run()

