import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
import io

# ─── Page Config ────────────────────────────────────────────────────────────
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
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
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
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧬 scRNA-seq Bias & Confounder Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect, visualize and correct systematic biases in single-cell transcriptomics data</div>', unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/DNА_model.png/240px-DNА_model.png", width=80)
    st.title("⚙️ Settings")

    st.markdown("### 📂 Data Input")
    upload_mode = st.radio("Choose input mode:", ["Upload CSV", "Use Demo Data"])

    st.markdown("### 🔬 Analysis Options")
    run_pca = st.checkbox("PCA Visualization", value=True)
    run_batch = st.checkbox("Batch Effect Detection", value=True)
    run_outlier = st.checkbox("Outlier Detection", value=True)
    run_dist = st.checkbox("Distribution Analysis", value=True)
    run_corr = st.checkbox("Correlation Heatmap", value=True)

    st.markdown("---")
    st.markdown("**About:** Built for BNITM PhD application — Computational Infection Biology")

# ─── Helper: Generate Demo Data ──────────────────────────────────────────────
@st.cache_data
def generate_demo_data():
    np.random.seed(42)
    n_cells = 300
    n_genes = 20

    gene_names = [f"Gene_{i}" for i in range(1, n_genes + 1)]

    # Simulate 3 batches with different expression profiles
    batch1 = np.random.negative_binomial(5, 0.4, (100, n_genes)).astype(float)
    batch2 = np.random.negative_binomial(8, 0.4, (100, n_genes)).astype(float) * 1.5  # batch effect
    batch3 = np.random.negative_binomial(5, 0.4, (100, n_genes)).astype(float) * 0.7

    data = np.vstack([batch1, batch2, batch3])
    df = pd.DataFrame(data, columns=gene_names)
    df["batch"] = ["Batch_1"] * 100 + ["Batch_2"] * 100 + ["Batch_3"] * 100
    df["cell_type"] = (["Macrophage"] * 50 + ["T-cell"] * 50) * 3
    df["total_counts"] = df[gene_names].sum(axis=1)
    df["n_genes_detected"] = (df[gene_names] > 0).sum(axis=1)
    df["mito_fraction"] = np.random.beta(2, 20, n_cells)  # mitochondrial gene fraction
    return df, gene_names

# ─── Load Data ───────────────────────────────────────────────────────────────
df = None
gene_cols = None

if upload_mode == "Upload CSV":
    uploaded_file = st.file_uploader(
        "📁 Upload your gene expression CSV file",
        type=["csv"],
        help="Rows = cells, Columns = genes. Optional columns: 'batch', 'cell_type', 'total_counts'"
    )
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        gene_cols = [c for c in numeric_cols if c not in ["total_counts", "n_genes_detected", "mito_fraction"]]
        st.success(f"✅ Loaded {df.shape[0]} cells × {len(gene_cols)} genes")
    else:
        st.info("👆 Please upload a CSV file, or switch to **Demo Data** in the sidebar.")

else:
    df, gene_cols = generate_demo_data()
    st.success("✅ Demo data loaded: 300 cells × 20 genes (3 batches simulated)")

# ─── Main Analysis ───────────────────────────────────────────────────────────
if df is not None:

    tabs = st.tabs(["📊 Overview", "🔬 PCA", "⚠️ Batch Effects", "🚨 Outliers", "📈 Distributions", "🌡️ Correlation", "💊 Correction"])

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔢 Total Cells", df.shape[0])
        c2.metric("🧬 Genes", len(gene_cols))
        c3.metric("📦 Batches", df["batch"].nunique() if "batch" in df.columns else "N/A")
        c4.metric("🦠 Cell Types", df["cell_type"].nunique() if "cell_type" in df.columns else "N/A")

        st.markdown("#### 🔍 Raw Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        if "total_counts" in df.columns:
            st.markdown("#### 📉 QC Metrics")
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            df["total_counts"].hist(ax=axes[0], bins=30, color="#1f77b4", edgecolor="white")
            axes[0].set_title("Total Counts per Cell")
            axes[0].set_xlabel("Total Counts")

            if "n_genes_detected" in df.columns:
                df["n_genes_detected"].hist(ax=axes[1], bins=30, color="#2ca02c", edgecolor="white")
                axes[1].set_title("Genes Detected per Cell")
                axes[1].set_xlabel("# Genes")

            if "mito_fraction" in df.columns:
                df["mito_fraction"].hist(ax=axes[2], bins=30, color="#d62728", edgecolor="white")
                axes[2].set_title("Mitochondrial Fraction")
                axes[2].set_xlabel("Mito Fraction")
                axes[2].axvline(0.2, color="black", linestyle="--", label="20% threshold")
                axes[2].legend()

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Tab 2: PCA ────────────────────────────────────────────────────────────
    with tabs[1]:
        if run_pca:
            st.markdown('<div class="section-header">🔬 PCA Visualization</div>', unsafe_allow_html=True)
            st.write("PCA helps reveal batch-driven clustering — a key sign of systematic bias.")

            expr = df[gene_cols].fillna(0)
            scaler = StandardScaler()
            scaled = scaler.fit_transform(expr)
            pca = PCA(n_components=min(10, len(gene_cols)))
            pcs = pca.fit_transform(scaled)

            color_by = st.selectbox("Color cells by:", [c for c in ["batch", "cell_type"] if c in df.columns])

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # PC1 vs PC2
            categories = df[color_by].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
            for cat, col in zip(categories, colors):
                mask = df[color_by] == cat
                axes[0].scatter(pcs[mask, 0], pcs[mask, 1], label=cat, alpha=0.7, s=20, color=col)
            axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            axes[0].set_title(f"PCA — colored by {color_by}")
            axes[0].legend(fontsize=8)

            # Scree plot
            axes[1].bar(range(1, len(pca.explained_variance_ratio_)+1),
                        pca.explained_variance_ratio_ * 100, color="#1f77b4")
            axes[1].set_xlabel("Principal Component")
            axes[1].set_ylabel("Variance Explained (%)")
            axes[1].set_title("Scree Plot")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            if "batch" in df.columns:
                st.markdown("""
                <div class="warning-box">
                ⚠️ <b>Bias Indicator:</b> If cells cluster primarily by <b>batch</b> rather than <b>cell type</b> in PCA, 
                this indicates a strong batch effect that needs correction.
                </div>""", unsafe_allow_html=True)

    # ── Tab 3: Batch Effects ──────────────────────────────────────────────────
    with tabs[2]:
        if run_batch and "batch" in df.columns:
            st.markdown('<div class="section-header">⚠️ Batch Effect Detection</div>', unsafe_allow_html=True)

            batches = df["batch"].unique()
            batch_means = df.groupby("batch")[gene_cols].mean()

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Mean expression per batch
            batch_means.T.plot(ax=axes[0], alpha=0.8)
            axes[0].set_title("Mean Gene Expression per Batch")
            axes[0].set_xlabel("Gene")
            axes[0].set_ylabel("Mean Expression")
            axes[0].tick_params(axis='x', rotation=45)

            # Boxplot of total counts per batch
            if "total_counts" in df.columns:
                batch_groups = [df[df["batch"] == b]["total_counts"].values for b in batches]
                axes[1].boxplot(batch_groups, labels=batches, patch_artist=True,
                                boxprops=dict(facecolor="#aec6e8"))
                axes[1].set_title("Total Counts Distribution per Batch")
                axes[1].set_ylabel("Total Counts")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ANOVA test for batch effects
            st.markdown("#### 📐 Statistical Test: ANOVA for Batch Effect")
            significant_genes = []
            for gene in gene_cols:
                groups = [df[df["batch"] == b][gene].values for b in batches]
                f_stat, p_val = stats.f_oneway(*groups)
                if p_val < 0.05:
                    significant_genes.append({"Gene": gene, "F-statistic": round(f_stat, 3), "p-value": round(p_val, 5)})

            if significant_genes:
                sig_df = pd.DataFrame(significant_genes)
                st.warning(f"⚠️ {len(significant_genes)} out of {len(gene_cols)} genes show significant batch effects (p < 0.05)")
                st.dataframe(sig_df, use_container_width=True)
            else:
                st.markdown('<div class="success-box">✅ No significant batch effects detected.</div>', unsafe_allow_html=True)

    # ── Tab 4: Outliers ───────────────────────────────────────────────────────
    with tabs[3]:
        if run_outlier:
            st.markdown('<div class="section-header">🚨 Outlier Cell Detection</div>', unsafe_allow_html=True)

            contamination = st.slider("Contamination rate (expected outlier fraction):", 0.01, 0.2, 0.05)
            expr = df[gene_cols].fillna(0)
            iso = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = iso.fit_predict(expr)
            df["outlier"] = outlier_labels

            n_outliers = (outlier_labels == -1).sum()
            st.metric("🚨 Outlier Cells Detected", n_outliers, delta=f"{n_outliers/len(df)*100:.1f}% of dataset")

            # Visualize on PCA
            scaler = StandardScaler()
            pcs = PCA(n_components=2).fit_transform(scaler.fit_transform(expr))

            fig, ax = plt.subplots(figsize=(8, 5))
            mask_in = outlier_labels == 1
            mask_out = outlier_labels == -1
            ax.scatter(pcs[mask_in, 0], pcs[mask_in, 1], c="#1f77b4", alpha=0.5, s=15, label="Normal")
            ax.scatter(pcs[mask_out, 0], pcs[mask_out, 1], c="#d62728", alpha=0.9, s=40, label="Outlier", marker="x")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Outlier Detection via Isolation Forest")
            ax.legend()
            st.pyplot(fig)
            plt.close()

            if n_outliers > 0:
                st.markdown("#### 🔍 Outlier Cell Details")
                st.dataframe(df[df["outlier"] == -1][gene_cols[:5]].head(10), use_container_width=True)

    # ── Tab 5: Distributions ──────────────────────────────────────────────────
    with tabs[4]:
        if run_dist:
            st.markdown('<div class="section-header">📈 Gene Expression Distributions</div>', unsafe_allow_html=True)

            selected_genes = st.multiselect("Select genes to plot:", gene_cols, default=gene_cols[:4])

            if selected_genes:
                fig, axes = plt.subplots(1, len(selected_genes), figsize=(4*len(selected_genes), 4))
                if len(selected_genes) == 1:
                    axes = [axes]
                for ax, gene in zip(axes, selected_genes):
                    vals = df[gene].values
                    ax.hist(vals, bins=30, color="#1f77b4", edgecolor="white", alpha=0.8)
                    ax.set_title(gene)
                    ax.set_xlabel("Expression")
                    ax.set_ylabel("# Cells")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Zero-inflation check
                st.markdown("#### 💧 Zero-Inflation Analysis")
                zero_fracs = [(g, (df[g] == 0).mean() * 100) for g in gene_cols]
                zero_df = pd.DataFrame(zero_fracs, columns=["Gene", "Zero Fraction (%)"])
                zero_df = zero_df.sort_values("Zero Fraction (%)", ascending=False)

                fig2, ax2 = plt.subplots(figsize=(12, 4))
                ax2.bar(zero_df["Gene"], zero_df["Zero Fraction (%)"], color="#e377c2")
                ax2.axhline(80, color="red", linestyle="--", label="80% threshold")
                ax2.set_title("Zero Fraction per Gene (Zero-Inflation Check)")
                ax2.set_ylabel("% Cells with Zero Expression")
                ax2.tick_params(axis='x', rotation=45)
                ax2.legend()
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

    # ── Tab 6: Correlation ────────────────────────────────────────────────────
    with tabs[5]:
        if run_corr:
            st.markdown('<div class="section-header">🌡️ Gene Correlation Heatmap</div>', unsafe_allow_html=True)

            max_genes = st.slider("Number of top genes to include:", 5, min(30, len(gene_cols)), 15)
            top_genes = gene_cols[:max_genes]

            corr_matrix = df[top_genes].corr()

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                        center=0, ax=ax, linewidths=0.5, annot_kws={"size": 7})
            ax.set_title("Gene-Gene Correlation Matrix")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Tab 7: Correction ─────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown('<div class="section-header">💊 Bias Correction (Z-score Normalization)</div>', unsafe_allow_html=True)
        st.write("Apply basic normalization to reduce batch-driven expression differences.")

        method = st.selectbox("Correction method:", [
            "Z-score normalization (per gene)",
            "Log normalization (log1p)",
            "Min-Max scaling"
        ])

        if st.button("▶️ Apply Correction"):
            expr = df[gene_cols].fillna(0)

            if method.startswith("Z-score"):
                corrected = pd.DataFrame(StandardScaler().fit_transform(expr), columns=gene_cols)
                method_name = "Z-score"
            elif method.startswith("Log"):
                corrected = np.log1p(expr)
                method_name = "Log1p"
            else:
                from sklearn.preprocessing import MinMaxScaler
                corrected = pd.DataFrame(MinMaxScaler().fit_transform(expr), columns=gene_cols)
                method_name = "Min-Max"

            st.success(f"✅ {method_name} correction applied!")

            # Before vs After
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].boxplot([df[g].values for g in gene_cols[:8]], labels=gene_cols[:8])
            axes[0].set_title("Before Correction")
            axes[0].tick_params(axis='x', rotation=45)

            axes[1].boxplot([corrected[g].values for g in gene_cols[:8]], labels=gene_cols[:8])
            axes[1].set_title(f"After {method_name} Correction")
            axes[1].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Download corrected data
            csv_buf = io.StringIO()
            corrected.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download Corrected Data (CSV)",
                data=csv_buf.getvalue(),
                file_name="corrected_expression.csv",
                mime="text/csv"
            )

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center>🧬 <b>scRNA-seq Bias Detector</b> | Built for Computational Infection Biology Research | "
    "Inspired by BNITM PhD Project</center>",
    unsafe_allow_html=True
)
