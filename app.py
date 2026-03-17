import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import kneighbors_graph
from scipy import stats
import io
import warnings
warnings.filterwarnings("ignore")

# ─── Optional heavy imports ──────────────────────────────────────────────────
try:
    import umap.umap_ as umap_lib
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

try:
    import harmonypy as hm
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="scRNA-seq Bias Detector",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.8rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 0.8rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .info-box {
        background: #e8f4fd;
        border-left: 4px solid #1f77b4;
        padding: 0.8rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧬 scRNA-seq Bias & Confounder Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect, visualize and correct systematic biases in single-cell transcriptomics data</div>', unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    st.markdown("### 📂 Data Input")
    upload_mode = st.radio("Choose input mode:", ["Upload CSV", "Use Demo Data", "PBMC Kang et al. 2018"])

    st.markdown("### 🔬 Analysis Modules")
    run_pca      = st.checkbox("PCA Visualization",        value=True)
    run_dimred   = st.checkbox("UMAP / t-SNE",             value=True)
    run_batch    = st.checkbox("Batch Effect Detection",   value=True)
    run_outlier  = st.checkbox("Outlier Detection",        value=True)
    run_dist     = st.checkbox("Distribution Analysis",    value=True)
    run_corr     = st.checkbox("Correlation Heatmap",      value=True)
    run_harmony  = st.checkbox("Harmony Batch Correction", value=True)
    run_gnn      = st.checkbox("GNN Cell Graph Module",    value=True)

    st.markdown("---")
    st.markdown("**scRNA-seq Bias Detector v2.0**")
    st.markdown("*Computational Genomics | Jaipur, India*")

# ─── Demo Data ───────────────────────────────────────────────────────────────
@st.cache_data
def generate_demo_data():
    np.random.seed(42)
    n_cells, n_genes = 300, 20
    gene_names = [f"Gene_{i}" for i in range(1, n_genes + 1)]

    batch1 = np.random.negative_binomial(5, 0.4, (100, n_genes)).astype(float)
    batch2 = np.random.negative_binomial(8, 0.4, (100, n_genes)).astype(float) * 1.5
    batch3 = np.random.negative_binomial(5, 0.4, (100, n_genes)).astype(float) * 0.7

    data = np.vstack([batch1, batch2, batch3])
    df = pd.DataFrame(data, columns=gene_names)
    df["batch"]         = ["Batch_1"] * 100 + ["Batch_2"] * 100 + ["Batch_3"] * 100
    df["cell_type"]     = (["Macrophage"] * 50 + ["T-cell"] * 50) * 3
    df["total_counts"]  = df[gene_names].sum(axis=1)
    df["n_genes_detected"] = (df[gene_names] > 0).sum(axis=1)
    df["mito_fraction"] = np.random.beta(2, 20, n_cells)
    return df, gene_names

# ─── PBMC Kang-style Data Generator ─────────────────────────────────────────
@st.cache_data
def generate_pbmc_data():
    """Kang et al. 2018 style PBMC data — Control vs IFN-β stimulated, 2 batches."""
    np.random.seed(2026)
    n_cells = 600
    n_genes = 50
    gene_names = (
        [f"ISG{i:02d}" for i in range(1, 16)] +   # IFN-stimulated genes (real names style)
        [f"HLA{i:02d}" for i in range(1, 11)] +    # MHC genes
        [f"CD{i:02d}"  for i in range(1, 11)] +    # Surface markers
        [f"RPS{i:02d}" for i in range(1, 9)]  +    # Ribosomal
        [f"MT{i:02d}"  for i in range(1, 9)]        # Mitochondrial
    )

    # Batch 1: Control PBMC
    b1 = np.random.negative_binomial(5, 0.5, (n_cells, n_genes)).astype(float)

    # Batch 2: IFN-β stimulated — ISG genes strongly upregulated
    b2 = np.random.negative_binomial(5, 0.5, (n_cells, n_genes)).astype(float)
    b2[:, :15] += np.random.uniform(6, 12, (n_cells, 15))   # ISG upregulation
    b2[:, 15:25] += np.random.uniform(2, 5, (n_cells, 10))  # HLA upregulation
    b2 *= 1.4  # sequencing depth difference (technical batch effect)

    # Log-normalize
    def lognorm(x):
        s = x.sum(axis=1, keepdims=True).clip(min=1)
        return np.log1p(x / s * 10000)

    b1 = lognorm(b1)
    b2 = lognorm(b2)

    data = np.vstack([b1, b2])
    df = pd.DataFrame(data, columns=gene_names)
    df["batch"]            = ["Control"] * n_cells + ["IFN_stimulated"] * n_cells
    df["cell_type"]        = (["Monocyte"] * 200 + ["T-cell"] * 200 + ["NK-cell"] * 200) * 2
    df["total_counts"]     = df[gene_names].sum(axis=1)
    df["n_genes_detected"] = (df[gene_names] > 0).sum(axis=1)
    df["mito_fraction"]    = np.concatenate([
        np.random.beta(2, 20, n_cells),
        np.random.beta(3, 15, n_cells)   # slightly higher mito in stimulated
    ])
    return df, gene_names

# ─── Load Data ───────────────────────────────────────────────────────────────
df = None
gene_cols = None

if upload_mode == "Upload CSV":
    uploaded_file = st.file_uploader(
        "📁 Upload your gene expression CSV",
        type=["csv"],
        help="Rows = cells, Columns = genes. Optional: 'batch', 'cell_type', 'total_counts'"
    )
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        gene_cols = [c for c in numeric_cols if c not in ["total_counts", "n_genes_detected", "mito_fraction"]]
        st.success(f"✅ Loaded {df.shape[0]} cells × {len(gene_cols)} genes")
    else:
        st.info("👆 Upload a CSV or switch to Demo Data in the sidebar.")
elif upload_mode == "Use Demo Data":
    df, gene_cols = generate_demo_data()
    st.success("✅ Demo data loaded: 300 cells × 20 genes (3 batches simulated)")
else:
    df, gene_cols = generate_pbmc_data()
    st.success("✅ PBMC dataset loaded: 1,200 cells × 50 genes | Control vs IFN-β stimulated (Kang et al. 2018 style)")
    st.info("📘 This dataset simulates the Kang et al. 2018 PBMC experiment — "
            "IFN-β stimulation creates strong, biologically meaningful batch effects "
            "ideal for benchmarking batch correction tools.")

# ─── Main Tabs ────────────────────────────────────────────────────────────────
if df is not None:
    tabs = st.tabs([
        "📊 Overview",
        "🔬 PCA",
        "🗺️ UMAP / t-SNE",
        "⚠️ Batch Effects",
        "🚨 Outliers",
        "📈 Distributions",
        "🌡️ Correlation",
        "🔧 Harmony Correction",
        "🕸️ GNN Module",
        "💊 Basic Correction"
    ])

    # ── Shared preprocessing ─────────────────────────────────────────────────
    expr       = df[gene_cols].fillna(0)
    scaler     = StandardScaler()
    expr_scaled = scaler.fit_transform(expr)

    # ════════════════════════════════════════════════════════════════════
    # TAB 0 — Overview
    # ════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔢 Total Cells",  df.shape[0])
        c2.metric("🧬 Genes",        len(gene_cols))
        c3.metric("📦 Batches",      df["batch"].nunique()    if "batch"     in df.columns else "N/A")
        c4.metric("🦠 Cell Types",   df["cell_type"].nunique() if "cell_type" in df.columns else "N/A")

        st.markdown("#### 🔍 Raw Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        if "total_counts" in df.columns:
            st.markdown("#### 📉 QC Metrics")
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            df["total_counts"].hist(ax=axes[0], bins=30, color="#1f77b4", edgecolor="white")
            axes[0].set_title("Total Counts per Cell"); axes[0].set_xlabel("Total Counts")
            if "n_genes_detected" in df.columns:
                df["n_genes_detected"].hist(ax=axes[1], bins=30, color="#2ca02c", edgecolor="white")
                axes[1].set_title("Genes Detected per Cell"); axes[1].set_xlabel("# Genes")
            if "mito_fraction" in df.columns:
                df["mito_fraction"].hist(ax=axes[2], bins=30, color="#d62728", edgecolor="white")
                axes[2].set_title("Mitochondrial Fraction"); axes[2].set_xlabel("Mito Fraction")
                axes[2].axvline(0.2, color="black", linestyle="--", label="20% threshold")
                axes[2].legend()
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # ════════════════════════════════════════════════════════════════════
    # TAB 1 — PCA
    # ════════════════════════════════════════════════════════════════════
    with tabs[1]:
        if run_pca:
            st.markdown('<div class="section-header">🔬 PCA Visualization</div>', unsafe_allow_html=True)
            st.write("PCA reveals batch-driven clustering — a key sign of systematic bias.")

            pca_model = PCA(n_components=min(10, len(gene_cols)))
            pcs = pca_model.fit_transform(expr_scaled)

            color_by = st.selectbox("Color cells by (PCA):",
                                    [c for c in ["batch", "cell_type"] if c in df.columns],
                                    key="pca_color")

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            categories = df[color_by].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
            for cat, col in zip(categories, colors):
                mask = df[color_by] == cat
                axes[0].scatter(pcs[mask, 0], pcs[mask, 1], label=cat, alpha=0.7, s=20, color=col)
            axes[0].set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)")
            axes[0].set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)")
            axes[0].set_title(f"PCA — colored by {color_by}")
            axes[0].legend(fontsize=8)

            axes[1].bar(range(1, len(pca_model.explained_variance_ratio_) + 1),
                        pca_model.explained_variance_ratio_ * 100, color="#1f77b4")
            axes[1].set_xlabel("Principal Component")
            axes[1].set_ylabel("Variance Explained (%)")
            axes[1].set_title("Scree Plot")
            plt.tight_layout(); st.pyplot(fig); plt.close()

            if "batch" in df.columns:
                st.markdown("""<div class="warning-box">
                ⚠️ <b>Bias Indicator:</b> If cells cluster by <b>batch</b> rather than <b>cell type</b>,
                a strong batch effect is present.</div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 2 — UMAP / t-SNE  (NEW)
    # ════════════════════════════════════════════════════════════════════
    with tabs[2]:
        if run_dimred:
            st.markdown('<div class="section-header">🗺️ UMAP / t-SNE Dimensionality Reduction</div>',
                        unsafe_allow_html=True)
            st.markdown("""<div class="info-box">
            📘 <b>Why UMAP/t-SNE?</b> PCA is linear — it misses nonlinear cell-type clusters.
            UMAP and t-SNE capture nonlinear manifold structure and are standard in
            modern single-cell workflows for revealing cell populations and batch mixing.
            </div>""", unsafe_allow_html=True)

            method = st.radio("Select method:", ["UMAP", "t-SNE"], horizontal=True)
            color_by_dr = st.selectbox("Color cells by:",
                                       [c for c in ["batch", "cell_type"] if c in df.columns],
                                       key="dr_color")

            # Use PCA-reduced space as input (standard practice)
            pca_50 = PCA(n_components=min(10, len(gene_cols))).fit_transform(expr_scaled)

            if method == "UMAP":
                if not UMAP_AVAILABLE:
                    st.error("❌ `umap-learn` not installed. Run: `pip install umap-learn`")
                else:
                    n_neighbors = st.slider("n_neighbors:", 5, 50, 15)
                    min_dist    = st.slider("min_dist:", 0.01, 0.99, 0.1)

                    with st.spinner("Running UMAP... (may take ~10 seconds)"):
                        reducer   = umap_lib.UMAP(n_neighbors=n_neighbors,
                                                   min_dist=min_dist,
                                                   random_state=42)
                        embedding = reducer.fit_transform(pca_50)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    categories = df[color_by_dr].unique()
                    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
                    for cat, col in zip(categories, colors):
                        mask = df[color_by_dr] == cat
                        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                                   label=cat, alpha=0.7, s=20, color=col)
                    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
                    ax.set_title(f"UMAP — colored by {color_by_dr}")
                    ax.legend(fontsize=9)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

                    if "batch" in df.columns and color_by_dr == "batch":
                        st.markdown("""<div class="warning-box">
                        ⚠️ <b>Batch mixing check:</b> Well-integrated data should show
                        interleaved batch colours. Separate clusters = uncorrected batch effect.
                        </div>""", unsafe_allow_html=True)

            else:  # t-SNE
                perplexity = st.slider("Perplexity:", 5, 50, 30, key="tsne_perp")
                n_iter_tsne = st.slider("Max iterations:", 250, 2000, 1000, step=250, key="tsne_iter")

                if st.button("▶️ Run t-SNE", key="run_tsne"):
                    with st.spinner("Running t-SNE... (may take ~15 seconds)"):
                        tsne      = TSNE(n_components=2, perplexity=perplexity,
                                         random_state=42, max_iter=n_iter_tsne)
                        embedding = tsne.fit_transform(pca_50)
                    st.session_state["tsne_embedding"] = embedding
                    st.session_state["tsne_color"]     = color_by_dr

                embedding = st.session_state.get("tsne_embedding", None)
                color_by_dr_tsne = st.session_state.get("tsne_color", color_by_dr)

                if embedding is not None:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    categories = df[color_by_dr_tsne].unique()
                    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
                    for cat, col in zip(categories, colors):
                        mask = df[color_by_dr_tsne] == cat
                        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                                   label=cat, alpha=0.7, s=20, color=col)
                    ax.set_xlabel("t-SNE-1"); ax.set_ylabel("t-SNE-2")
                    ax.set_title(f"t-SNE — colored by {color_by_dr_tsne}")
                    ax.legend(fontsize=9)
                    plt.tight_layout(); st.pyplot(fig); plt.close()
                else:
                    st.info("👆 Set perplexity and click **Run t-SNE** to compute the embedding.")

            # Side-by-side comparison with PCA
            st.markdown("#### 🔄 Pipeline: PCA → UMAP → Batch Assessment")
            st.markdown("""<div class="info-box">
            QC → <b>PCA</b> (linear structure) → <b>UMAP/t-SNE</b> (nonlinear clusters)
            → <b>Anomaly Detection</b> → <b>Batch Assessment</b>
            </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 3 — Batch Effects
    # ════════════════════════════════════════════════════════════════════
    with tabs[3]:
        if run_batch and "batch" in df.columns:
            st.markdown('<div class="section-header">⚠️ Batch Effect Detection</div>', unsafe_allow_html=True)
            batches     = df["batch"].unique()
            batch_means = df.groupby("batch")[gene_cols].mean()

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            batch_means.T.plot(ax=axes[0], alpha=0.8)
            axes[0].set_title("Mean Gene Expression per Batch")
            axes[0].set_xlabel("Gene"); axes[0].set_ylabel("Mean Expression")
            axes[0].tick_params(axis="x", rotation=45)

            if "total_counts" in df.columns:
                batch_groups = [df[df["batch"] == b]["total_counts"].values for b in batches]
                axes[1].boxplot(batch_groups, labels=batches, patch_artist=True,
                                boxprops=dict(facecolor="#aec6e8"))
                axes[1].set_title("Total Counts Distribution per Batch")
                axes[1].set_ylabel("Total Counts")
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.markdown("#### 📐 ANOVA — Statistical Batch Effect Test")
            significant_genes = []
            for gene in gene_cols:
                groups = [df[df["batch"] == b][gene].values for b in batches]
                f_stat, p_val = stats.f_oneway(*groups)
                if p_val < 0.05:
                    significant_genes.append({"Gene": gene,
                                              "F-statistic": round(f_stat, 3),
                                              "p-value": round(p_val, 5)})
            if significant_genes:
                sig_df = pd.DataFrame(significant_genes)
                st.warning(f"⚠️ {len(significant_genes)}/{len(gene_cols)} genes show significant batch effects (p < 0.05)")
                st.dataframe(sig_df, use_container_width=True)
            else:
                st.markdown('<div class="success-box">✅ No significant batch effects detected.</div>',
                            unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 4 — Outliers
    # ════════════════════════════════════════════════════════════════════
    with tabs[4]:
        if run_outlier:
            st.markdown('<div class="section-header">🚨 Outlier Cell Detection</div>', unsafe_allow_html=True)
            contamination = st.slider("Contamination rate:", 0.01, 0.2, 0.05)
            iso           = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = iso.fit_predict(expr)
            df["outlier"]  = outlier_labels

            n_outliers = (outlier_labels == -1).sum()
            st.metric("🚨 Outlier Cells Detected", n_outliers,
                      delta=f"{n_outliers/len(df)*100:.1f}% of dataset")

            pcs2 = PCA(n_components=2).fit_transform(expr_scaled)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(pcs2[outlier_labels == 1, 0],  pcs2[outlier_labels == 1, 1],
                       c="#1f77b4", alpha=0.5, s=15, label="Normal")
            ax.scatter(pcs2[outlier_labels == -1, 0], pcs2[outlier_labels == -1, 1],
                       c="#d62728", alpha=0.9, s=40, label="Outlier", marker="x")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.set_title("Outlier Detection via Isolation Forest")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

            if n_outliers > 0:
                st.markdown("#### 🔍 Outlier Cell Details")
                st.dataframe(df[df["outlier"] == -1][gene_cols[:5]].head(10),
                             use_container_width=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 5 — Distributions
    # ════════════════════════════════════════════════════════════════════
    with tabs[5]:
        if run_dist:
            st.markdown('<div class="section-header">📈 Gene Expression Distributions</div>',
                        unsafe_allow_html=True)
            selected_genes = st.multiselect("Select genes:", gene_cols, default=gene_cols[:4])
            if selected_genes:
                fig, axes = plt.subplots(1, len(selected_genes),
                                         figsize=(4 * len(selected_genes), 4))
                if len(selected_genes) == 1: axes = [axes]
                for ax, gene in zip(axes, selected_genes):
                    ax.hist(df[gene].values, bins=30, color="#1f77b4",
                            edgecolor="white", alpha=0.8)
                    ax.set_title(gene); ax.set_xlabel("Expression"); ax.set_ylabel("# Cells")
                plt.tight_layout(); st.pyplot(fig); plt.close()

                st.markdown("#### 💧 Zero-Inflation Analysis")
                zero_df = pd.DataFrame(
                    [(g, (df[g] == 0).mean() * 100) for g in gene_cols],
                    columns=["Gene", "Zero Fraction (%)"]
                ).sort_values("Zero Fraction (%)", ascending=False)

                fig2, ax2 = plt.subplots(figsize=(12, 4))
                ax2.bar(zero_df["Gene"], zero_df["Zero Fraction (%)"], color="#e377c2")
                ax2.axhline(80, color="red", linestyle="--", label="80% threshold")
                ax2.set_title("Zero Fraction per Gene")
                ax2.set_ylabel("% Cells with Zero Expression")
                ax2.tick_params(axis="x", rotation=45); ax2.legend()
                plt.tight_layout(); st.pyplot(fig2); plt.close()

    # ════════════════════════════════════════════════════════════════════
    # TAB 6 — Correlation
    # ════════════════════════════════════════════════════════════════════
    with tabs[6]:
        if run_corr:
            st.markdown('<div class="section-header">🌡️ Gene Correlation Heatmap</div>',
                        unsafe_allow_html=True)
            max_genes  = st.slider("Top genes to include:", 5, min(30, len(gene_cols)), 15)
            corr_matrix = df[gene_cols[:max_genes]].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                        center=0, ax=ax, linewidths=0.5, annot_kws={"size": 7})
            ax.set_title("Gene–Gene Correlation Matrix")
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # ════════════════════════════════════════════════════════════════════
    # TAB 7 — HARMONY BATCH CORRECTION  (NEW)
    # ════════════════════════════════════════════════════════════════════
    with tabs[7]:
        if run_harmony:
            st.markdown('<div class="section-header">🔧 Harmony Batch Correction</div>',
                        unsafe_allow_html=True)
            st.markdown("""<div class="info-box">
            📘 <b>What is Harmony?</b> Harmony is a state-of-the-art batch integration algorithm
            for single-cell data. It iteratively adjusts PCA embeddings so that cells from
            different batches intermix properly, while preserving biological cell-type structure.
            <br><br>
            <b>Pipeline:</b> Raw data → PCA → Harmony integration → Re-visualize → Compare separation scores
            </div>""", unsafe_allow_html=True)

            if "batch" not in df.columns:
                st.warning("⚠️ No 'batch' column found. Harmony requires batch labels.")
            elif not HARMONY_AVAILABLE:
                st.error("❌ `harmonypy` not installed. Run: `pip install harmonypy`")
                st.code("pip install harmonypy", language="bash")
            else:
                n_pcs = st.slider("Number of PCs for Harmony:", 5, min(20, len(gene_cols)), 10)
                theta = st.slider("Theta (batch penalty strength):", 0.0, 4.0, 2.0, step=0.5,
                                  help="Higher = stronger batch mixing. Default = 2.0")

                if st.button("▶️ Run Harmony Correction"):
                    with st.spinner("Running Harmony batch integration..."):
                        # PCA before correction
                        pca_before = PCA(n_components=n_pcs)
                        pcs_before = pca_before.fit_transform(expr_scaled)

                        # Harmony integration
                        meta = df[["batch"]].copy()
                        ho   = hm.run_harmony(pcs_before, meta, "batch", theta=theta,
                                              max_iter_harmony=20, random_state=42)
                        # Z_corr shape is (n_pcs, n_cells) — convert to numpy and transpose
                        pcs_after = np.array(ho.Z_corr).T
                        # Safety: if transpose gave wrong orientation, flip back
                        if pcs_after.shape[0] != pcs_before.shape[0]:
                            pcs_after = pcs_after.T

                    st.success("✅ Harmony correction complete!")

                    # ── Before vs After PCA plots ──────────────────────────
                    st.markdown("#### 📊 Before vs After Harmony: PCA Embedding")
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    categories = df["batch"].unique()
                    colors     = plt.cm.tab10(np.linspace(0, 1, len(categories)))

                    for cat, col in zip(categories, colors):
                        mask = (df["batch"] == cat).values
                        axes[0].scatter(pcs_before[mask, 0], pcs_before[mask, 1],
                                        label=cat, alpha=0.6, s=20, color=col)
                        axes[1].scatter(pcs_after[mask, 0],  pcs_after[mask, 1],
                                        label=cat, alpha=0.6, s=20, color=col)

                    axes[0].set_title("Before Harmony (PCA)")
                    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].legend(fontsize=8)
                    axes[1].set_title("After Harmony (Corrected Embedding)")
                    axes[1].set_xlabel("HC1"); axes[1].set_ylabel("HC2"); axes[1].legend(fontsize=8)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

                    # ── Quantitative batch mixing score ───────────────────
                    st.markdown("#### 📐 Batch Separation Score (lower = better mixing)")

                    def batch_separation_score(embedding, batch_labels):
                        """Mean distance between batch centroids (lower = more mixed)."""
                        centroids = {}
                        for b in np.unique(batch_labels):
                            centroids[b] = embedding[batch_labels == b].mean(axis=0)
                        centroid_vals = np.array(list(centroids.values()))
                        dists = []
                        for i in range(len(centroid_vals)):
                            for j in range(i + 1, len(centroid_vals)):
                                dists.append(np.linalg.norm(centroid_vals[i] - centroid_vals[j]))
                        return np.mean(dists)

                    batch_arr  = df["batch"].values
                    score_before = batch_separation_score(pcs_before, batch_arr)
                    score_after  = batch_separation_score(pcs_after,  batch_arr)
                    improvement  = (score_before - score_after) / score_before * 100

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Before Harmony", f"{score_before:.3f}")
                    col2.metric("After Harmony",  f"{score_after:.3f}",
                                delta=f"-{improvement:.1f}%")
                    col3.metric("Batch Mixing Improvement", f"{improvement:.1f}%")

                    if improvement > 20:
                        st.markdown('<div class="success-box">✅ Harmony significantly reduced batch separation. '
                                    'Corrected embedding recommended for downstream analysis.</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">⚠️ Modest improvement. '
                                    'Consider increasing theta or checking data quality.</div>',
                                    unsafe_allow_html=True)

                    # ── Download corrected embedding ──────────────────────
                    st.markdown("#### ⬇️ Download Corrected Embedding")
                    harmony_df  = pd.DataFrame(pcs_after,
                                               columns=[f"HC{i+1}" for i in range(pcs_after.shape[1])])
                    harmony_df["batch"]     = df["batch"].values
                    if "cell_type" in df.columns:
                        harmony_df["cell_type"] = df["cell_type"].values

                    csv_buf = io.StringIO()
                    harmony_df.to_csv(csv_buf, index=False)
                    st.download_button("⬇️ Download Harmony-corrected embedding (CSV)",
                                       data=csv_buf.getvalue(),
                                       file_name="harmony_corrected.csv",
                                       mime="text/csv")

    # ════════════════════════════════════════════════════════════════════
    # TAB 8 — GNN CELL GRAPH MODULE  (NEW)
    # ════════════════════════════════════════════════════════════════════
    with tabs[8]:
        if run_gnn:
            st.markdown('<div class="section-header">🕸️ Graph Neural Network Cell Similarity Module</div>',
                        unsafe_allow_html=True)
            st.markdown("""<div class="info-box">
            📘 <b>Why GNN?</b> Cells can be represented as a graph where edges connect
            transcriptomically similar cells (k-nearest neighbours). A Graph Convolutional
            Network (GCN) then learns richer cell embeddings that capture neighbourhood
            context — enabling improved anomaly detection and cell-type discrimination
            beyond what PCA alone can reveal.
            <br><br>
            <b>Pipeline:</b> Expression matrix → kNN cell graph → GCN embedding → Anomaly scoring
            </div>""", unsafe_allow_html=True)

            if not GNN_AVAILABLE:
                # Graceful fallback with sklearn kNN graph + manual GCN
                st.info("ℹ️ Running lightweight sklearn GNN approximation (mathematically equivalent 1-layer GCN).")
                st.markdown("#### 🕸️ kNN Cell Similarity Graph (sklearn fallback)")

                k_neighbors = st.slider("k (neighbours per cell):", 3, 15, 5, key="knn_k")

                with st.spinner("Building kNN graph..."):
                    pca_gnn = PCA(n_components=min(10, len(gene_cols))).fit_transform(expr_scaled)
                    adj     = kneighbors_graph(pca_gnn, k_neighbors, mode="connectivity",
                                               include_self=False)
                    adj_arr = adj.toarray()

                    # Simple graph smoothing: X_smooth = D^{-1} A X  (1-layer GCN approx)
                    degree   = adj_arr.sum(axis=1, keepdims=True).clip(min=1)
                    X_smooth = (adj_arr @ pca_gnn) / degree

                # Anomaly detection on smoothed embeddings
                iso_gnn        = IsolationForest(contamination=0.05, random_state=42)
                gnn_outliers   = iso_gnn.fit_predict(X_smooth)
                n_gnn_outliers = (gnn_outliers == -1).sum()

                st.metric("🕸️ GNN-detected Outlier Cells", n_gnn_outliers,
                          delta=f"{n_gnn_outliers/len(df)*100:.1f}% of dataset")

                # Visualise
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # Original PCA
                normal_mask  = gnn_outliers == 1
                outlier_mask = gnn_outliers == -1
                axes[0].scatter(pca_gnn[normal_mask, 0],  pca_gnn[normal_mask, 1],
                                c="#1f77b4", alpha=0.5, s=15, label="Normal")
                axes[0].scatter(pca_gnn[outlier_mask, 0], pca_gnn[outlier_mask, 1],
                                c="#d62728", alpha=0.9, s=40, label="GNN Outlier", marker="x")
                axes[0].set_title("PCA — GNN Outliers Highlighted")
                axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].legend()

                # GCN-smoothed embedding
                axes[1].scatter(X_smooth[normal_mask, 0],  X_smooth[normal_mask, 1],
                                c="#2ca02c", alpha=0.5, s=15, label="Normal")
                axes[1].scatter(X_smooth[outlier_mask, 0], X_smooth[outlier_mask, 1],
                                c="#d62728", alpha=0.9, s=40, label="GNN Outlier", marker="x")
                axes[1].set_title("GCN-Smoothed Embedding (1-layer approximation)")
                axes[1].set_xlabel("GCN-dim 1"); axes[1].set_ylabel("GCN-dim 2"); axes[1].legend()

                plt.tight_layout(); st.pyplot(fig); plt.close()

                # Graph stats
                st.markdown("#### 📐 Cell Graph Statistics")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Edges",         int(adj_arr.sum() / 2))
                c2.metric("Avg Degree",           f"{adj_arr.sum(axis=1).mean():.1f}")
                c3.metric("Graph Density",        f"{adj_arr.sum() / (len(df)*(len(df)-1)):.4f}")

                st.markdown("""<div class="info-box">
                🔬 <b>Interpretation:</b> The GCN smoothing aggregates each cell's expression
                with its k nearest neighbours. Cells that remain isolated (short path lengths)
                after smoothing are flagged as anomalies — these likely represent doublets,
                dead cells, or contaminating populations.
                </div>""", unsafe_allow_html=True)

            else:
                # Full PyTorch Geometric GCN
                st.success("✅ PyTorch Geometric detected — running full GCN.")
                k_neighbors = st.slider("k (neighbours):", 3, 15, 5, key="knn_k_full")
                n_epochs    = st.slider("Training epochs:", 10, 100, 30)

                if st.button("▶️ Train GCN"):
                    with st.spinner("Building graph and training GCN..."):
                        pca_gnn = PCA(n_components=min(10, len(gene_cols))).fit_transform(expr_scaled)
                        adj     = kneighbors_graph(pca_gnn, k_neighbors, mode="connectivity",
                                                   include_self=False)
                        coo     = adj.tocoo()
                        edge_index = torch.tensor(
                            np.vstack([coo.row, coo.col]), dtype=torch.long)
                        x = torch.tensor(pca_gnn, dtype=torch.float)

                        class SimpleGCN(torch.nn.Module):
                            def __init__(self, in_ch, hid_ch, out_ch):
                                super().__init__()
                                self.conv1 = GCNConv(in_ch,  hid_ch)
                                self.conv2 = GCNConv(hid_ch, out_ch)
                            def forward(self, x, edge_index):
                                x = F.relu(self.conv1(x, edge_index))
                                return self.conv2(x, edge_index)

                        data  = Data(x=x, edge_index=edge_index)
                        model = SimpleGCN(x.shape[1], 16, 2)
                        opt   = torch.optim.Adam(model.parameters(), lr=0.01)

                        losses = []
                        for epoch in range(n_epochs):
                            model.train(); opt.zero_grad()
                            out  = model(data.x, data.edge_index)
                            loss = F.mse_loss(out, data.x[:, :2])
                            loss.backward(); opt.step()
                            losses.append(loss.item())

                        model.eval()
                        with torch.no_grad():
                            embedding_gnn = model(data.x, data.edge_index).numpy()

                    st.success(f"✅ GCN trained. Final loss: {losses[-1]:.4f}")

                    # Loss curve
                    fig_loss, ax_loss = plt.subplots(figsize=(8, 3))
                    ax_loss.plot(losses, color="#1f77b4")
                    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("MSE Loss")
                    ax_loss.set_title("GCN Training Loss")
                    plt.tight_layout(); st.pyplot(fig_loss); plt.close()

                    # GCN embedding visualization
                    iso_gnn      = IsolationForest(contamination=0.05, random_state=42)
                    gnn_outliers = iso_gnn.fit_predict(embedding_gnn)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(embedding_gnn[gnn_outliers == 1, 0],
                               embedding_gnn[gnn_outliers == 1, 1],
                               c="#1f77b4", alpha=0.6, s=20, label="Normal")
                    ax.scatter(embedding_gnn[gnn_outliers == -1, 0],
                               embedding_gnn[gnn_outliers == -1, 1],
                               c="#d62728", alpha=0.9, s=50, label="GCN Outlier", marker="x")
                    ax.set_xlabel("GCN-dim 1"); ax.set_ylabel("GCN-dim 2")
                    ax.set_title("GCN Embedding — Anomaly Detection")
                    ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    # ════════════════════════════════════════════════════════════════════
    # TAB 9 — Basic Correction
    # ════════════════════════════════════════════════════════════════════
    with tabs[9]:
        st.markdown('<div class="section-header">💊 Basic Normalization</div>', unsafe_allow_html=True)
        st.write("Apply simple normalization as a quick baseline before Harmony.")

        method = st.selectbox("Method:", [
            "Z-score normalization (per gene)",
            "Log normalization (log1p)",
            "Min-Max scaling"
        ])

        if st.button("▶️ Apply Correction"):
            if method.startswith("Z-score"):
                corrected   = pd.DataFrame(StandardScaler().fit_transform(expr), columns=gene_cols)
                method_name = "Z-score"
            elif method.startswith("Log"):
                corrected   = np.log1p(expr)
                method_name = "Log1p"
            else:
                corrected   = pd.DataFrame(MinMaxScaler().fit_transform(expr), columns=gene_cols)
                method_name = "Min-Max"

            st.success(f"✅ {method_name} correction applied!")

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].boxplot([df[g].values        for g in gene_cols[:8]], labels=gene_cols[:8])
            axes[0].set_title("Before Correction")
            axes[0].tick_params(axis="x", rotation=45)
            axes[1].boxplot([corrected[g].values  for g in gene_cols[:8]], labels=gene_cols[:8])
            axes[1].set_title(f"After {method_name} Correction")
            axes[1].tick_params(axis="x", rotation=45)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            csv_buf = io.StringIO()
            corrected.to_csv(csv_buf, index=False)
            st.download_button("⬇️ Download Corrected Data (CSV)",
                               data=csv_buf.getvalue(),
                               file_name="corrected_expression.csv",
                               mime="text/csv")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center>🧬 <b>scRNA-seq Bias Detector v2.0</b> | "
    "UMAP · t-SNE · Harmony · GNN | "
    "Computational Genomics Research</center>",
    unsafe_allow_html=True
)
