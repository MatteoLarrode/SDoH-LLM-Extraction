# ============================================
# ==== Helper functions for data analysis ====
# ============================================
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_merge_structured_and_predictions(
    structured_path: str,
    predictions_path: str,
    case_id_col: str = "case_ref",
    prediction_label_col: str = "generated_labels"
) -> pd.DataFrame:
    """
    Load and merge structured SDoH indicators and LLM-predicted labels.

    Parameters:
    - structured_path: Path to the CSV with structured data.
    - predictions_path: Path to the CSV with LLM output.
    - case_id_col: Unique ID column (usually 'case_ref').
    - prediction_label_col: Column containing <LIST> tags with predictions.

    Returns:
    - Merged DataFrame with parsed label lists.
    """
    df_structured = pd.read_csv(structured_path)
    df_predictions = pd.read_csv(predictions_path)

    df_predictions[prediction_label_col] = (
        df_predictions[prediction_label_col]
        .str.replace(r"</?LIST>", "", regex=True)
        .apply(lambda x: [label.strip().lower() for label in x.split(",") if label.strip()])
    )

    df_merged = pd.merge(df_structured, df_predictions[[case_id_col, prediction_label_col]], on=case_id_col)
    return df_merged

def create_exhaustiveness_long_format(
    df_merged: pd.DataFrame,
    structured_to_label: dict,
    case_id_col: str = "case_ref",
    prediction_label_col: str = "generated_labels"
) -> pd.DataFrame:
    """
    Construct long-format exhaustiveness classification table.

    Parameters:
    - df_merged: Merged DataFrame from load_and_merge_structured_and_predictions.
    - structured_to_label: Mapping from structured columns to free-text SDoH labels.
    - case_id_col: Unique identifier column.
    - prediction_label_col: Column with list of predicted labels.

    Returns:
    - Long-format DataFrame with documentation status per SDoH per case.
    """
    records = []

    for _, row in df_merged.iterrows():
        for structured_col, label in structured_to_label.items():
            experienced = row[structured_col] == 1.0
            documented = label.lower() in row[prediction_label_col]

            if experienced and documented:
                doc_status = "Experienced and documented in referral"
            elif experienced and not documented:
                doc_status = "Experienced but not documented in referral"
            elif not experienced and documented:
                doc_status = "Not experienced but documented in referral"
            else:
                doc_status = "Neither experienced nor documented in referral"

            records.append({
                case_id_col: row[case_id_col],
                "sdoh_category": label,
                "structured_experienced": experienced,
                "free_text_documented": documented,
                "documentation_status": doc_status
            })

    return pd.DataFrame(records)

def summarize_exhaustiveness(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize documentation status counts.

    Parameters:
    - df_long: Long-format exhaustiveness DataFrame.

    Returns:
    - Summary DataFrame with counts and recall score.
    """
    summary = df_long.groupby(["sdoh_category", "documentation_status"]).size().unstack(fill_value=0)
    return summary

def plot_exhaustiveness_bar_faceted(df_long, save_path="../results/figures/notes_exhaustiveness.png"):
    """
    Side-by-side bar charts of SDoH documentation exhaustiveness by dataset.
    Uses IFRC colour palette and labels consistent with visual framework.
    Food Access is excluded from HIU.
    
    Parameters:
    - df_long: Long-format DataFrame with 'sdoh_category', 'documentation_status', and 'dataset' columns.
    - save_path: File path to save the figure (PNG)
    """
    # Filter out Food Access from HIU
    df_long = df_long[~((df_long["dataset"] == "HIU") & (df_long["sdoh_category"] == "foodaccess"))]

    label_order = [
        "Experienced and documented in referral",
        "Experienced but not documented in referral",
        "Not experienced but documented in referral",
        "Neither experienced nor documented in referral"
    ]
    
    label_color_map = {
        "Experienced and documented in referral": colour_palette["medium_purple"],
        "Experienced but not documented in referral": colour_palette["ifrc_red"],
        "Not experienced but documented in referral": colour_palette["medium_blue"],
        "Neither experienced nor documented in referral": colour_palette["medium_green"],
    }

    label_map = {
        "housing": "Housing",
        "finances": "Finances",
        "loneliness": "Loneliness",
        "foodaccess": "Food Access"
    }

    df_long["sdoh_category"] = df_long["sdoh_category"].map(label_map)

    sns.set_style("ticks")
    datasets = df_long["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 6), sharey=True)

    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        df_subset = df_long[df_long["dataset"] == dataset]
        category_order = df_subset["sdoh_category"].unique()

        sns.countplot(
            data=df_subset,
            x="sdoh_category",
            hue="documentation_status",
            order=category_order,
            hue_order=label_order,
            palette=label_color_map,
            ax=ax,
            alpha=0.8,
            width=0.7
        )

        ax.set_title(f"Documentation in {dataset} referral notes", fontsize=14)
        ax.set_xlabel("", fontsize=12)
        ax.tick_params(axis='x', labelrotation=0, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylabel("Number of Service Users" if dataset == "HIU" else "", fontsize=12)
        ax.legend_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        #title="SDoH need",
        fontsize=13,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_note_recall_stacked_bar(df_long, save_path="../results/figures/note_recall_stacked_bar_horizontal.png"):
    """
    Horizontal stacked barplot of experienced SDoH needs, split by whether the need was documented
    in the referral note. Bar length = # experienced; split = documented / not documented.

    Parameters:
    - df_long: Long-format DataFrame with 'structured_experienced', 'free_text_documented',
               'sdoh_category', and 'dataset' columns.
    - save_path: File path to save the figure.
    """
    # Filter to experienced needs
    df_exp = df_long[df_long["structured_experienced"] == True].copy()
    df_exp = df_exp[~((df_exp["dataset"] == "HIU") & (df_exp["sdoh_category"] == "foodaccess"))]

    # Count by documented status
    counts = (
        df_exp
        .groupby(["dataset", "sdoh_category", "free_text_documented"])
        .size()
        .reset_index(name="count")
    )

    # Pivot to wide format
    counts_wide = counts.pivot_table(
        index=["dataset", "sdoh_category"],
        columns="free_text_documented",
        values="count",
        fill_value=0
    ).reset_index()

    # Rename columns
    counts_wide = counts_wide.rename(columns={
        True: "Documented",
        False: "Not Documented"
    })

    # Map SDoH labels
    label_map = {
        "housing": "Housing",
        "finances": "Finances",
        "loneliness": "Loneliness",
        "foodaccess": "Food Access"
    }
    counts_wide["SDoH Category"] = counts_wide["sdoh_category"].map(label_map)

    # Calculate recall (%)
    counts_wide["Total"] = counts_wide["Documented"] + counts_wide["Not Documented"]
    counts_wide["Recall"] = counts_wide["Documented"] / counts_wide["Total"] * 100

    # Set up horizontal bar chart
    sns.set_style("ticks")
    datasets = counts_wide["dataset"].unique()
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 6), sharex=False)

    if len(datasets) == 1:
        axes = [axes]

    bar_colors = {
        "Documented": colour_palette["medium_purple"],
        "Not Documented": colour_palette["ifrc_red"]
    }

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        df_subset = counts_wide[counts_wide["dataset"] == dataset].copy()

        y_labels = df_subset["SDoH Category"]
        nd = df_subset["Not Documented"]
        d = df_subset["Documented"]
        total = df_subset["Total"]

        # Base bar (Not Documented)
        ax.barh(
            y=y_labels,
            width=nd,
            label="Not Documented",
            color=bar_colors["Not Documented"],
            alpha=0.7,
            height=0.5
        )

        # Stacked bar (Documented)
        ax.barh(
            y=y_labels,
            width=d,
            left=nd,
            label="Documented",
            color=bar_colors["Documented"],
            alpha=0.7,
            height=0.5
        )

        # Annotate recall % properly
        for y_pos, row in enumerate(df_subset.itertuples()):
            total_val = row.Documented + row._3  # _3 = Not Documented (column 3)
            recall_pct = row.Recall
            ax.text(
                x=total_val + max(1, 0.02 * total_val),
                y=y_pos,
                s=f"{recall_pct:.0f}%",
                va="center",
                ha="left",
                fontsize=11
            )

        ax.set_title(f"{dataset} referrals", fontsize=14)
        ax.set_ylabel("", fontsize=12)
        ax.set_xlabel("Number of service users experiencing need", fontsize=12)
        ax.tick_params(axis='y', labelsize=11)
        ax.tick_params(axis='x', labelsize=11)

        sns.despine(ax=ax)

    # Legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
        fontsize=11
    )

    fig.subplots_adjust(hspace=0.6)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()