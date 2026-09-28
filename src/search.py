import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-oss-120b",
    ):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents

            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please add it to your .env file or environment variables."
            )

        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized with model: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        if not results:
            return "No relevant documents found in the vector store."

        context_blocks = []
        sources = []
        for i, r in enumerate(results, 1):
            meta = r.get("metadata") or {}
            source = meta.get("source", "Unknown Source")
            source_name = os.path.basename(source) if source != "Unknown Source" else "Unknown Document"
            page = meta.get("page", "N/A")
            text = meta.get("text", "").strip()

            if text:
                context_blocks.append(f"[{i}] File: {source_name} (Page {page}):\n{text}")
                sources.append(f"[{i}] {source_name} (Page {page})")

        context = "\n\n".join(context_blocks)
        if not context:
            return "No text content found in retrieved documents."

        prompt = (
            "You are an expert academic research assistant.\n"
            "Answer the question strictly based on the provided context below.\n"
            "Use inline citations like [1], [2] to reference the source documents.\n"
            "If the provided context does not contain enough information to answer the question, "
            "clearly state that the information is not present in the indexed documents. "
            "Do not make up facts or extrapolate beyond the provided text.\n\n"
            f"Context:\n{context}\n\n"
            f"Query: {query}\n\n"
            "Structured Answer:"
        )

        response = self.llm.invoke([prompt])
        citations_summary = "\n".join(sources)
        return f"{response.content}\n\n### References Retrieved:\n{citations_summary}"



# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
