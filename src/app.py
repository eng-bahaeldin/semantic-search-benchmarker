"""src/app.py - SearchWorkbench: Interactive Search Comparator & IR Benchmark Dashboard."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from src.benchmarker import RetrievalBenchmarker
from src.chunking import FixedSizeChunker, StructureAwareChunker
from src.indexer import BM25SparseIndex, DenseVectorIndex, get_shared_encoder
from src.search import UnifiedSearchEngine

st.set_page_config(
    page_title="SearchWorkbench | Semantic Search & IR Benchmarker",
    layout="wide",
)


@st.cache_resource(show_spinner="Indexing technical documentation corpus across chunking strategies...")
def load_search_engines() -> dict[str, UnifiedSearchEngine]:
    encoder = get_shared_encoder("all-MiniLM-L6-v2")

    chunks_struct = StructureAwareChunker().chunk_corpus("corpus")
    chunks_256 = FixedSizeChunker(chunk_size=256, chunk_overlap=32).chunk_corpus("corpus")
    chunks_1024 = FixedSizeChunker(chunk_size=1024, chunk_overlap=128).chunk_corpus("corpus")

    return {
        "Structure-Aware (Markdown Headers)": UnifiedSearchEngine(
            bm25_index=BM25SparseIndex(chunks_struct),
            dense_index=DenseVectorIndex(chunks_struct, collection_prefix="ui_struct", encoder=encoder),
        ),
        "Fixed-Size (256 Tokens / 32 Overlap)": UnifiedSearchEngine(
            bm25_index=BM25SparseIndex(chunks_256),
            dense_index=DenseVectorIndex(chunks_256, collection_prefix="ui_256", encoder=encoder),
        ),
        "Fixed-Size (1024 Tokens / 128 Overlap)": UnifiedSearchEngine(
            bm25_index=BM25SparseIndex(chunks_1024),
            dense_index=DenseVectorIndex(chunks_1024, collection_prefix="ui_1024", encoder=encoder),
        ),
    }


engines = load_search_engines()

st.title("SearchWorkbench: Semantic Search Engine & Empirical IR Benchmarker")
st.caption(
    "Comparing Sparse Lexical Retrieval (BM25Okapi) vs. Dense Vector Search (all-MiniLM-L6-v2 + ChromaDB) across Fixed & Structure-Aware Chunking"
)

tab_compare, tab_benchmark = st.tabs(
    ["1. Live Search Comparator (BM25 vs. Dense)", "2. Benchmark & Evaluation Dashboard"]
)

with tab_compare:
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        selected_strategy = st.selectbox(
            "Chunking & Indexing Strategy:",
            options=list(engines.keys()),
            index=0,
        )
    with ctrl_col2:
        top_k = st.slider("Top-K Results (K):", min_value=1, max_value=10, value=5)

    st.markdown("**Quick Comparison Presets:**")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    preset_q = ""
    with p_col1:
        if st.button("Lexical: maxUnavailable Default", width="stretch"):
            preset_q = "What is the default value of maxUnavailable in RollingUpdateDeployment?"
    with p_col2:
        if st.button("Lexical: ECONNRESET Pool Error", width="stretch"):
            preset_q = "Error: ECONNRESET in pooled connection"
    with p_col3:
        if st.button("Conceptual: Zero-Downtime Rollout", width="stretch"):
            preset_q = "How do I ensure zero-downtime rolling upgrades during heavy traffic?"
    with p_col4:
        if st.button("Conceptual: Socket Starvation", width="stretch"):
            preset_q = "Preventing network socket starvation in client database pools"

    with st.form("live_search_form", clear_on_submit=False):
        query_input = st.text_input(
            "Enter a technical search query (exact identifier or conceptual question):",
            value=preset_q,
            placeholder="e.g., CUSTOM_CONFIG_FLAG_XYZ or How does Envoy rotate mTLS certificates?",
        )
        run_search = st.form_submit_button("Compare BM25 vs. Dense Retrieval", type="primary")

    if run_search and query_input.strip():
        active_engine = engines[selected_strategy]
        bm25_hits, bm25_ms = active_engine.search_bm25(query_input, top_k=top_k)
        dense_hits, dense_ms = active_engine.search_dense(query_input, top_k=top_k)

        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("Sparse Lexical Search (BM25Okapi)")
            st.metric("Query Latency (ms)", f"{bm25_ms:.2f} ms", delta=f"{len(bm25_hits)} hits")
            if not bm25_hits:
                st.warning("Zero lexical token matches found in corpus.")
            for hit in bm25_hits:
                with st.container(border=True):
                    st.markdown(
                        f"**Rank #{hit.rank}** | BM25 Score: **`{hit.score:.4f}`** | `{hit.chunk.parent_doc}`"
                    )
                    st.caption(
                        f"Section: {hit.chunk.metadata.get('header_hierarchy')} | "
                        f"Tokens: {hit.chunk.metadata.get('token_count')} | "
                        f"Matched Terms: `{ ', '.join(hit.matched_terms) or 'none' }`"
                    )
                    st.markdown(hit.chunk.content[:420] + ("..." if len(hit.chunk.content) > 420 else ""))

        with right_col:
            st.subheader("Dense Vector Search (Cosine Similarity)")
            st.metric("Query Latency (ms)", f"{dense_ms:.2f} ms", delta=f"{len(dense_hits)} hits")
            for hit in dense_hits:
                with st.container(border=True):
                    st.markdown(
                        f"**Rank #{hit.rank}** | Cosine Similarity: **`{hit.score:.4f}`** | `{hit.chunk.parent_doc}`"
                    )
                    st.caption(
                        f"Section: {hit.chunk.metadata.get('header_hierarchy')} | "
                        f"Tokens: {hit.chunk.metadata.get('token_count')} | "
                        f"Lexical Overlap: `{ ', '.join(hit.matched_terms) or 'semantic match only' }`"
                    )
                    st.markdown(hit.chunk.content[:420] + ("..." if len(hit.chunk.content) > 420 else ""))

with tab_benchmark:
    st.markdown(
        "Evaluates **labeled technical queries** (`data/eval_queries.json`) "
        "across **BM25 Baseline**, **Dense Fixed-256**, **Dense Fixed-1024**, and **Dense Structure-Aware**."
    )

    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = None

    if st.button("Run Benchmark Suite (Ground-Truth Queries)", type="primary"):
        with st.spinner("Running 4-way empirical IR benchmark across Hit Rate@K, Recall@K, MRR@K, and Latency..."):
            benchmarker = RetrievalBenchmarker(corpus_dir="corpus", eval_path="data/eval_queries.json")
            summary_rows, failure_cases = benchmarker.run_all_benchmarks()
            st.session_state.benchmark_results = (summary_rows, failure_cases)

    if st.session_state.benchmark_results is not None:
        summary_rows, failure_cases = st.session_state.benchmark_results
        df = pd.DataFrame([row.model_dump() for row in summary_rows])

        st.subheader("Empirical Retrieval Metrics Comparison")
        display_df = df.rename(
            columns={
                "configuration": "Configuration",
                "chunk_count": "Total Chunks",
                "hit_rate_at_3": "Hit Rate@3",
                "hit_rate_at_5": "Hit Rate@5",
                "recall_at_5": "Recall@5",
                "mrr_at_5": "MRR@5",
                "avg_latency_ms": "Avg Latency (ms)",
                "p50_latency_ms": "p50 Latency (ms)",
                "p95_latency_ms": "p95 Latency (ms)",
            }
        )
        st.dataframe(display_df, width="stretch", hide_index=True)

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("#### Ranking Quality: Hit Rate@5 vs. MRR@5")
            quality_chart = display_df.set_index("Configuration")[["Hit Rate@3", "Hit Rate@5", "MRR@5"]]
            st.bar_chart(quality_chart)

        with chart_col2:
            st.markdown("#### Query Execution Latency: p50 vs. p95 (ms)")
            latency_chart = display_df.set_index("Configuration")[["p50 Latency (ms)", "p95 Latency (ms)"]]
            st.bar_chart(latency_chart)

        st.divider()
        st.subheader("Head-to-Head Failure & Trade-Off Analysis (BM25 vs. Structure-Aware Dense)")
        if failure_cases:
            st.dataframe(pd.DataFrame(failure_cases), width="stretch", hide_index=True)
        else:
            st.success("Both engines tied across every query in the evaluation dataset.")
