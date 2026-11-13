from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on sys.path when invoked via `streamlit run src/app.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.benchmarker import BenchmarkSummary, run_full_benchmark
from src.indexer import SearchResult, tokenize_for_bm25
from src.search import SearchEngine, StrategyName, build_all_engines

load_dotenv()

st.set_page_config(
    page_title="SearchWorkbench | Semantic vs. Lexical IR Benchmarker",
    page_icon="🔍",
    layout="wide",
)


def get_api_key() -> str | None:
    """Safely checks Streamlit secrets first, then environment variables."""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


@st.cache_resource(show_spinner="Building BM25 & Dense Vector indices across chunking strategies...")
def load_search_engines(backend: str, api_key: str | None) -> Dict[StrategyName, SearchEngine]:
    return build_all_engines(
        backend=backend,
        openai_api_key=api_key,
        corpus_dir=str(ROOT_DIR / "corpus"),
    )


def highlight_matching_terms(text: str, query: str) -> str:
    """Wraps exact query tokens found in the chunk text in bold markdown markers."""
    tokens = sorted(set(tokenize_for_bm25(query)), key=len, reverse=True)
    highlighted = text
    for tok in tokens:
        if len(tok) < 3:
            continue
        pattern = re.compile(rf"(?i)\b({re.escape(tok)})\b")
        highlighted = pattern.sub(r"**[\1]**", highlighted)
    return highlighted


def render_result_card(res: SearchResult, query: str) -> None:
    score_label = "Cosine Similarity" if res.engine == "dense" else "BM25 Score"
    header_info = res.metadata.get("header_hierarchy", "N/A")
    token_count = res.metadata.get("token_count", "?")
    lines_span = f"L{res.metadata.get('start_line', '?')}-L{res.metadata.get('end_line', '?')}"

    with st.container(border=True):
        top_col1, top_col2 = st.columns([3, 1])
        with top_col1:
            st.markdown(f"**Rank #{res.rank}** — `{res.parent_doc}` ({lines_span})")
            st.caption(f"Section: {header_info} | Tokens: {token_count}")
        with top_col2:
            st.metric(label=score_label, value=f"{res.score:.4f}")

        st.markdown(highlight_matching_terms(res.content, query))


# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("⚙️ SearchWorkbench")
    st.caption("Empirical IR & Chunking Evaluation Harness")

    openai_key = get_api_key()
    backend_options = ["local (ONNX all-MiniLM-L6-v2)"]
    if openai_key and not openai_key.startswith("your_"):
        backend_options.append("openai (text-embedding-3-small)")

    selected_backend_label = st.selectbox("Embedding Backend", options=backend_options)
    backend_key = "openai" if selected_backend_label.startswith("openai") else "local"

    engines = load_search_engines(backend_key, openai_key)

    st.divider()
    st.subheader("Corpus Index Stats")
    for strat_key, eng in engines.items():
        st.write(f"• **{strat_key}**: `{len(eng.chunks)}` chunks")

    st.divider()
    top_k = st.slider("Top-K Cutoff (K)", min_value=1, max_value=10, value=5)


# --- MAIN HEADER ---
st.title("🔍 Semantic Document Search Engine & Empirical Benchmarker")
st.markdown(
    "Compare **Sparse Lexical Retrieval (BM25)** against **Dense Vector Search (ChromaDB Cosine)** "
    "across **Fixed-Token** and **Structure-Aware Markdown Chunking** strategies."
)

tab_search, tab_benchmark = st.tabs(
    ["⚖️ Live Search Comparator", "📊 Benchmark & Failure Analysis Dashboard"]
)

# ==============================================================================
# TAB 1: LIVE SEARCH COMPARATOR
# ==============================================================================
with tab_search:
    st.subheader("Side-by-Side Lexical vs. Dense Retrieval")

    preset_col1, preset_col2, preset_col3 = st.columns(3)
    if "live_query" not in st.session_state:
        st.session_state.live_query = "What is the default value of maxUnavailable in RollingUpdateDeployment?"

    with preset_col1:
        if st.button("🎯 Lexical Preset: maxUnavailable", use_container_width=True):
            st.session_state.live_query = "What is the default value of maxUnavailable in RollingUpdateDeployment?"
    with preset_col2:
        if st.button("🎯 Lexical Preset: ECONNRESET", use_container_width=True):
            st.session_state.live_query = "Error: ECONNRESET in pooled connection"
    with preset_col3:
        if st.button("🧠 Semantic Preset: Zero-Downtime", use_container_width=True):
            st.session_state.live_query = "How do I ensure zero-downtime rolling upgrades during heavy traffic?"

    q_col, strat_col = st.columns([3, 1])
    with q_col:
        user_query = st.text_input(
            "Enter a technical search query:",
            key="live_query",
            placeholder="e.g., Preventing network socket starvation in client pools",
        )
    with strat_col:
        active_strategy: StrategyName = st.selectbox(
            "Chunking Strategy",
            options=["structure_aware", "fixed_256", "fixed_1024"],
            index=0,
        )  # type: ignore[assignment]

    if user_query.strip():
        active_engine = engines[active_strategy]
        bm25_results, bm25_ms = active_engine.query_bm25(user_query, top_k=top_k)
        dense_results, dense_ms = active_engine.query_dense(user_query, top_k=top_k)

        left_col, right_col = st.columns(2)
        with left_col:
            st.markdown(f"### 🔤 BM25 Lexical Search (`{bm25_ms:.2f} ms`)")
            if not bm25_results or bm25_results[0].score == 0.0:
                st.warning("No exact lexical token matches found in corpus.")
            for res in bm25_results:
                render_result_card(res, user_query)

        with right_col:
            st.markdown(f"### 🧬 Dense Vector Search (`{dense_ms:.2f} ms`)")
            for res in dense_results:
                render_result_card(res, user_query)


# ==============================================================================
# TAB 2: BENCHMARK & EVALUATION DASHBOARD
# ==============================================================================
with tab_benchmark:
    st.subheader("Automated Retrieval Evaluation Harness (`data/eval_queries.json`)")
    st.write(
        "Evaluates **18 curated technical queries** (9 exact-identifier lexical queries + "
        "9 paraphrased semantic queries) across all 4 required retrieval configurations."
    )

    run_btn = st.button("🚀 Run Full Benchmark Suite", type="primary")

    if run_btn or "benchmark_results" not in st.session_state:
        with st.spinner("Running evaluation across all 4 configurations..."):
            st.session_state.benchmark_results = run_full_benchmark(
                engines=engines,
                eval_path=ROOT_DIR / "data" / "eval_queries.json",
                k=top_k,
            )
            st.session_state.benchmark_k = top_k

    # Re-run automatically if the user changed Top-K slider
    if st.session_state.get("benchmark_k") != top_k:
        st.session_state.benchmark_results = run_full_benchmark(
            engines=engines,
            eval_path=ROOT_DIR / "data" / "eval_queries.json",
            k=top_k,
        )
        st.session_state.benchmark_k = top_k

    summaries: List[BenchmarkSummary] = st.session_state.benchmark_results

    # Summary Dataframe
    summary_rows = [
        {
            "Configuration": s.configuration,
            "Chunk Count": s.total_chunks,
            f"Hit Rate@{top_k}": s.hit_rate,
            f"Recall@{top_k}": s.recall,
            f"MRR@{top_k}": s.mrr,
            "Avg Latency (ms)": s.avg_latency_ms,
            "p50 Latency (ms)": s.p50_latency_ms,
            "p95 Latency (ms)": s.p95_latency_ms,
        }
        for s in summaries
    ]
    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # Bar Charts: Accuracy Metrics & Latency
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        df_melted = df_summary.melt(
            id_vars=["Configuration"],
            value_vars=[f"Hit Rate@{top_k}", f"Recall@{top_k}", f"MRR@{top_k}"],
            var_name="Metric",
            value_name="Score",
        )
        fig_metrics = px.bar(
            df_melted,
            x="Configuration",
            y="Score",
            color="Metric",
            barmode="group",
            title=f"Retrieval Quality Comparison (K={top_k})",
            range_y=[0, 1.05],
        )
        fig_metrics.update_layout(xaxis_tickangle=-15)
        st.plotly_chart(fig_metrics, use_container_width=True)

    with chart_col2:
        df_lat = df_summary.melt(
            id_vars=["Configuration"],
            value_vars=["p50 Latency (ms)", "p95 Latency (ms)"],
            var_name="Percentile",
            value_name="Latency (ms)",
        )
        fig_lat = px.bar(
            df_lat,
            x="Configuration",
            y="Latency (ms)",
            color="Percentile",
            barmode="group",
            title="Query Execution Latency (ms)",
        )
        fig_lat.update_layout(xaxis_tickangle=-15)
        st.plotly_chart(fig_lat, use_container_width=True)

    # --- FAILURE ANALYSIS PANEL ---
    st.divider()
    st.subheader("🔬 Failure Analysis: Where BM25 Beats Dense Search (and Vice Versa)")

    bm25_struct = summaries[0].per_query
    dense_struct = summaries[3].per_query

    bm25_wins = []
    dense_wins = []
    for b_item, d_item in zip(bm25_struct, dense_struct):
        if b_item.reciprocal_rank > d_item.reciprocal_rank:
            bm25_wins.append((b_item, d_item))
        elif d_item.reciprocal_rank > b_item.reciprocal_rank:
            dense_wins.append((b_item, d_item))

    fa_col1, fa_col2 = st.columns(2)
    with fa_col1:
        st.markdown(f"#### 🔤 BM25 Outperformed Dense ({len(bm25_wins)} queries)")
        st.caption(
            "Why: Exact configuration flags, sysctl keys, and error constants carry high IDF weight "
            "in BM25, whereas dense embeddings smooth specific tokens into broader semantic neighborhoods."
        )
        if not bm25_wins:
            st.info("Both engines tied on all exact-keyword queries at this K.")
        for b_q, d_q in bm25_wins:
            with st.expander(f"[{b_q.query_id}] {b_q.query}"):
                st.write(f"**Query Type:** `{b_q.query_type}` | **Target:** `{b_q.expected_section}`")
                st.write(f"**BM25 RR@{top_k}:** `{b_q.reciprocal_rank:.2f}` vs. **Dense RR@{top_k}:** `{d_q.reciprocal_rank:.2f}`")

    with fa_col2:
        st.markdown(f"#### 🧬 Dense Outperformed BM25 ({len(dense_wins)} queries)")
        st.caption(
            "Why: Conceptual queries using synonyms (e.g., 'thundering herd', 'zero-downtime') share "
            "few exact surface tokens with the target spec, causing BM25 term-frequency starvation."
        )
        if not dense_wins:
            st.info("Both engines tied on all semantic queries at this K.")
        for b_q, d_q in dense_wins:
            with st.expander(f"[{d_q.query_id}] {d_q.query}"):
                st.write(f"**Query Type:** `{d_q.query_type}` | **Target:** `{d_q.expected_section}`")
                st.write(f"**Dense RR@{top_k}:** `{d_q.reciprocal_rank:.2f}` vs. **BM25 RR@{top_k}:** `{b_q.reciprocal_rank:.2f}`")
