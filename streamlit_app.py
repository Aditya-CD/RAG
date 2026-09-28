import os
# Force Transformers to use PyTorch only and ignore TensorFlow/Keras
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
from pathlib import Path
import pickle
import faiss
import streamlit as st
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Streamlit page config
st.set_page_config(
    page_title="Academic Paper RAG Assistant | End-to-End Pipeline",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, high-contrast dark theme
st.markdown(
    """
    <style>
    /* Metric Cards */
    .metric-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Pipeline Step Badge */
    .step-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        background-color: #0284c7;
        color: white;
        margin-bottom: 8px;
    }

    /* Chunk Card */
    .chunk-card {
        background-color: #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
    .chunk-header {
        display: flex;
        justify-content: space-between;
        font-weight: 600;
        font-size: 14px;
        color: #e2e8f0;
        margin-bottom: 6px;
    }
    .chunk-distance {
        color: #a5b4fc;
        font-family: monospace;
        font-size: 13px;
    }
    .chunk-text {
        font-size: 13px;
        color: #cbd5e1;
        line-height: 1.5;
        white-space: pre-wrap;
    }

    /* Answer Box */
    .answer-box {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.1);
    }

    /* Code & Vector display */
    .vector-pill {
        display: inline-block;
        font-family: monospace;
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 12px;
        color: #38bdf8;
        margin: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading FAISS index & SentenceTransformer...")
def get_rag_engine(llm_model: str = "openai/gpt-oss-120b"):
    from src.search import RAGSearch

    return RAGSearch(llm_model=llm_model)


def get_vector_store_stats():
    faiss_path = os.path.join("faiss_store", "faiss.index")
    meta_path = os.path.join("faiss_store", "metadata.pkl")
    if os.path.exists(faiss_path) and os.path.exists(meta_path):
        try:
            index = faiss.read_index(faiss_path)
            ntotal = index.ntotal
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
            nmeta = len(metadata)
            return {"exists": True, "vectors": ntotal, "chunks": nmeta}
        except Exception as e:
            return {"exists": False, "error": str(e)}
    return {"exists": False, "vectors": 0, "chunks": 0}


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Configuration")
    
    # Check Groq Key
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        api_key_input = st.text_input("Enter Groq API Key", type="password")
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input
            st.success("API Key updated!")
    else:
        st.success("🟢 Groq API Connected", icon="✅")

    st.markdown("---")
    st.markdown("### 🤖 LLM Engine")
    model_choice = st.selectbox(
        "Model Selection",
        [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
        ],
        index=0,
        help="Models available via your high-throughput Groq Cloud endpoint.",
    )

    st.markdown("### 🔍 Retrieval Parameters")
    top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=8, value=3)

    st.markdown("---")
    st.markdown("### 📊 Index Health")
    stats = get_vector_store_stats()
    if stats.get("exists"):
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Chunks", stats["chunks"])
        with c2:
            st.metric("FAISS Vectors", stats["vectors"])
        st.caption("📍 Index: `faiss.IndexFlatL2` (384-dim)")
    else:
        st.warning("⚠️ FAISS index not built yet.")

    st.markdown("---")
    st.markdown(
        """
        **Pipeline Stack**:
        - 📄 **Ingestion**: PyMuPDF & Multi-loader
        - ✂️ **Chunking**: Recursive Splitter (1000/200)
        - 🧠 **Embeddings**: `all-MiniLM-L6-v2` (384-d)
        - 🗄️ **Index**: FAISS L2 Flat
        - ⚡ **LLM**: Groq LPU Inference
        """
    )


# ==========================================
# MAIN INTERFACE TABS
# ==========================================
tab_query, tab_blueprint, tab_knowledge = st.tabs(
    ["🚀 Interactive Query & Pipeline Inspector", "🏗️ Pipeline Architecture", "📂 Knowledge Base & Re-indexing"]
)

# ------------------------------------------
# TAB 1: INTERACTIVE QUERY & PIPELINE INSPECTOR
# ------------------------------------------
with tab_query:
    st.title("📚 Academic Research Paper RAG Assistant")
    st.markdown(
        "Ask research questions against your indexed documents. Inspect every transformation step across the **Retrieval-Augmented Generation** lifecycle."
    )

    # Example question chips
    st.markdown("**Sample Research Queries:**")
    preset_cols = st.columns(3)
    sample_queries = [
        "What is Docker and how does container isolation work?",
        "What are the core features and benefits of FastAPI?",
        "How do Python functions and generators operate?",
    ]

    selected_sample = None
    for i, col in enumerate(preset_cols):
        if col.button(f"💡 {sample_queries[i]}"):
            selected_sample = sample_queries[i]

    # Query Input Box
    default_text = selected_sample if selected_sample else ""
    query_text = st.text_input(
        "Enter your research query:",
        value=default_text,
        placeholder="e.g. How does asynchronous request handling work in FastAPI?",
    )

    col_btn, col_clear = st.columns([1, 5])
    run_query = col_btn.button("⚡ Run Pipeline", type="primary")

    if run_query and query_text.strip():
        with st.spinner("Executing End-to-End RAG Pipeline..."):
            try:
                rag = get_rag_engine(llm_model=model_choice)
                detailed = rag.search_pipeline_detailed(
                    query_text.strip(), top_k=top_k, llm_model=model_choice
                )
            except Exception as e:
                st.error(f"Execution Error: {e}")
                detailed = None

        if detailed:
            # Latency and Performance Metrics
            st.markdown("### ⏱️ Pipeline Execution Metrics")
            retrieval_ms = detailed['retrieval_time'] * 1000
            retrieval_val = f"{retrieval_ms:.1f}ms" if retrieval_ms >= 0.1 else "< 0.5ms"
            
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f"""<div class="metric-container">
                        <div class="metric-label">Total Latency</div>
                        <div class="metric-value">{detailed['total_time']:.2f}s</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""<div class="metric-container">
                        <div class="metric-label">FAISS Retrieval</div>
                        <div class="metric-value">{retrieval_val}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f"""<div class="metric-container">
                        <div class="metric-label">LLM Generation</div>
                        <div class="metric-value">{detailed['generation_time']:.2f}s</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f"""<div class="metric-container">
                        <div class="metric-label">Chunks Retrieved</div>
                        <div class="metric-value">{len(detailed['retrieved_chunks'])} / {top_k}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # Synthesized Grounded Response
            st.markdown("---")
            st.markdown("### 💡 Grounded Synthesized Response")
            
            # Format any raw token citations like 【1†L1-L8】 to clean readable [1]
            import re
            clean_answer = re.sub(r'【(\d+)[^】]*】', r'[\1]', detailed['answer'])

            with st.container(border=True):
                st.markdown(clean_answer)

            # References Cited
            if detailed["sources"]:
                st.markdown("#### 📑 Source Citations:")
                ref_cols = st.columns(len(detailed["sources"]))
                for idx, src in enumerate(detailed["sources"]):
                    ref_cols[idx].info(src)

            # Deep Pipeline Inspector
            st.markdown("---")
            st.markdown("### 🔬 Step-by-Step Pipeline Inspector")
            
            with st.expander("🧬 Stage 1: Dense Query Vectorization", expanded=False):
                st.markdown(
                    """
                    The user query is transformed into a dense 384-dimensional vector using **Sentence-Transformers (`all-MiniLM-L6-v2`)**.
                    """
                )
                emb = detailed["query_embedding"][0]
                norm = float(np.linalg.norm(emb))
                c_a, c_b = st.columns(2)
                c_a.metric("Vector Dimensionality", f"{len(emb)} floats")
                c_b.metric("Vector L2 Norm", f"{norm:.4f}")
                
                st.markdown("**First 16 Vector Coefficients Preview:**")
                preview_html = "".join([f'<span class="vector-pill">{v:.4f}</span>' for v in emb[:16]])
                preview_html += '<span class="vector-pill">...</span>'
                st.markdown(preview_html, unsafe_allow_html=True)

            with st.expander("🎯 Stage 2: FAISS Top-K Semantic Retrieval (IndexFlatL2)", expanded=True):
                st.markdown(
                    f"Showing top **{len(detailed['retrieved_chunks'])}** nearest semantic chunks matching the query embedding in Euclidean distance space ($L_2$)."
                )
                for chunk in detailed["retrieved_chunks"]:
                    st.markdown(
                        f"""
                        <div class="chunk-card">
                            <div class="chunk-header">
                                <span>📄 [{chunk['rank']}] {chunk['source']} (Page {chunk['page']})</span>
                                <span class="chunk-distance">L2 Distance: {chunk['distance']:.4f}</span>
                            </div>
                            <div class="chunk-text">{chunk['text']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with st.expander("🛡️ Stage 3: Grounded Context Assembly & System Guardrails", expanded=False):
                st.markdown(
                    "This structured prompt enforces **strict grounding** and hallucination mitigation before sending to the Groq inference engine:"
                )
                st.code(detailed["prompt"], language="text")


# ------------------------------------------
# TAB 2: PIPELINE ARCHITECTURE BLUEPRINT
# ------------------------------------------
with tab_blueprint:
    st.header("🏗️ System Architecture & Engineering Flow")
    st.markdown(
        """
        This RAG system implements a low-latency, modular retrieval and generation workflow designed for scientific literature and technical documents.
        """
    )

    if os.path.exists("assets/rag_pipeline_architecture.jpg"):
        st.image(
            "assets/rag_pipeline_architecture.jpg",
            caption="Academic Paper RAG Assistant - End-to-End Architecture Flow",
        )

    st.markdown(
        """
        ```mermaid
        graph TD
            subgraph Offline ["1. Document Ingestion & Vector Indexing"]
                A["📄 Raw Documents\n(PDF, DOCX, TXT, CSV)"] --> B["Document Parsers\n(PyMuPDF / TextLoader)"]
                B --> C["Recursive Character Splitting\n(Chunk: 1000 | Overlap: 200)"]
                C --> D["Embedding Model\n(all-MiniLM-L6-v2, 384-dim)"]
                D --> E[("FAISS Vector Index\nIndexFlatL2 + metadata.pkl")]
            end

            subgraph Online ["2. Online Query & Grounded Synthesis"]
                Q["🔎 User Query"] --> F["Embed Query\n(384-dim Vector)"]
                F --> G["FAISS Semantic Search\n(Euclidean Distance L2)"]
                E --> G
                G --> H["Context Aggregator\n(With Document & Page Citations)"]
                H --> I["Grounded Prompt Guardrails\n(Zero-Hallucination Policy)"]
                I --> J["Groq LPU Engine\n(High-Throughput Inference)"]
                J --> K["💡 Verifiable Response\n+ Exact Source Citations [1], [2]"]
            end
        ```
        """
    )

    st.markdown("### 🧠 Engineering Decisions Breakdown")
    arch_cols = st.columns(3)
    with arch_cols[0]:
        st.markdown(
            """
            #### 1. Ingestion & Chunking
            - **PyMuPDF**: Ultra-fast PDF page-level parsing.
            - **Recursive Splitting**: 1000 characters chunk size with 200 characters overlap preserves paragraph semantic boundaries.
            """
        )
    with arch_cols[1]:
        st.markdown(
            """
            #### 2. Dense Vector Index
            - **Sentence-Transformers**: Compact 384-d semantic representation.
            - **FAISS `IndexFlatL2`**: Exact $L_2$ distance search guarantees zero recall loss without approximation artifacts.
            """
        )
    with arch_cols[2]:
        st.markdown(
            """
            #### 3. Low-Latency Synthesis
            - **Groq LPU Acceleration**: Sub-second time-to-first-token (TTFT).
            - **Attribution Guardrails**: Every claim mapped to `[1]`, `[2]` citation markers backed by persistent metadata.
            """
        )


# ------------------------------------------
# TAB 3: KNOWLEDGE BASE & RE-INDEXING
# ------------------------------------------
with tab_knowledge:
    st.header("📂 Knowledge Base & Real-time Indexer")
    st.markdown("Inspect indexed files or upload new technical papers to re-build your FAISS vector store.")

    data_dir = Path("data")
    if data_dir.exists():
        files = list(data_dir.rglob("*.*"))
        valid_files = [f for f in files if f.suffix.lower() in [".pdf", ".txt", ".docx", ".csv", ".xlsx", ".json"]]
        
        st.subheader(f"📁 Active Documents in `data/` ({len(valid_files)} files)")
        
        doc_data = []
        for f in valid_files:
            doc_data.append({
                "Filename": f.name,
                "Relative Path": str(f.relative_to(data_dir)),
                "Format": f.suffix.upper(),
                "Size (KB)": round(f.stat().st_size / 1024, 2),
            })
        if doc_data:
            st.dataframe(doc_data)
        else:
            st.info("No documents found in `data/` folder.")

    # Upload new file
    st.markdown("---")
    st.subheader("📤 Upload New Documents")
    uploaded_file = st.file_uploader(
        "Upload a PDF, TXT, or DOCX document to the knowledge base:",
        type=["pdf", "txt", "docx", "csv"],
    )

    if uploaded_file is not None:
        save_target = data_dir / "pdf_files" / uploaded_file.name
        save_target.parent.mkdir(parents=True, exist_ok=True)
        with open(save_target, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded `{uploaded_file.name}` to `{save_target}`! Click Re-index below to include it.")

    # Reindex Button
    st.markdown("---")
    st.subheader("🔄 Rebuild FAISS Vector Store")
    st.caption("Re-scans `data/`, chunks all files, computes dense embeddings, and writes to `faiss_store/`.")

    if st.button("🔨 Rebuild Vector Index Now", type="secondary"):
        with st.status("Re-indexing in progress...", expanded=True) as status:
            st.write("Loading raw documents...")
            from src.data_loader import load_all_documents
            from src.vectorstore import FaissVectorStore

            docs = load_all_documents("data")
            st.write(f"Loaded {len(docs)} document pages/sections.")

            st.write("Chunking and generating 384-d embeddings with `all-MiniLM-L6-v2`...")
            store = FaissVectorStore("faiss_store")
            store.build_from_documents(docs)

            status.update(label="Vector Store Rebuilt Successfully! ✅", state="complete")
            st.rerun()
