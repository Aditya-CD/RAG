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

    def search_pipeline_detailed(self, query: str, top_k: int = 5, llm_model: str = None) -> dict:
        import time

        t_start = time.time()
        # 1. Embed query
        query_emb = self.vectorstore.embedding_pipeline.embed_text(query)
        
        # 2. Similarity search in FAISS
        t_retrieval_start = time.time()
        results = self.vectorstore.search(query_emb, top_k=top_k)
        retrieval_time = time.time() - t_retrieval_start

        context_blocks = []
        sources = []
        retrieved_chunks = []

        for i, r in enumerate(results, 1):
            meta = r.get("metadata") or {}
            source = meta.get("source", "Unknown Source")
            source_name = os.path.basename(source) if source != "Unknown Source" else "Unknown Document"
            page = meta.get("page", "N/A")
            text = meta.get("text", "").strip()
            dist = r.get("distance", 0.0)

            chunk_info = {
                "rank": i,
                "source": source_name,
                "full_path": source,
                "page": page,
                "distance": dist,
                "text": text,
            }
            retrieved_chunks.append(chunk_info)

            if text:
                context_blocks.append(f"[{i}] File: {source_name} (Page {page}):\n{text}")
                sources.append(f"[{i}] {source_name} (Page {page})")

        context = "\n\n".join(context_blocks)
        
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

        # 3. LLM generation
        t_gen_start = time.time()
        active_llm = self.llm
        if llm_model and llm_model != getattr(self.llm, "model_name", None):
            groq_api_key = os.getenv("GROQ_API_KEY")
            active_llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)

        response = active_llm.invoke([prompt])
        generation_time = time.time() - t_gen_start
        total_time = time.time() - t_start

        return {
            "query": query,
            "query_embedding": query_emb,
            "retrieved_chunks": retrieved_chunks,
            "context": context,
            "prompt": prompt,
            "answer": response.content,
            "sources": sources,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "model_name": getattr(active_llm, "model_name", self.llm_model if hasattr(self, "llm_model") else "groq"),
        }

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        detailed = self.search_pipeline_detailed(query, top_k=top_k)
        if not detailed["retrieved_chunks"]:
            return "No relevant documents found in the vector store."
        citations_summary = "\n".join(detailed["sources"])
        return f"{detailed['answer']}\n\n### References Retrieved:\n{citations_summary}"



# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
