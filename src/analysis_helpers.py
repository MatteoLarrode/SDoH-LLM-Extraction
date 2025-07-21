# ============================================
# ==== Helper functions for data analysis ====
# ============================================
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Colour palette
# From https://brand.ifrc.org/ifrc-brand-system/basics/colour
colour_palette = {
    'ifrc_red': '#EE2435',
    'ifrc_darkblue': '#011E41',
    'dark_green': '#009775',
    'medium_green': '#00AB84',
    'light_green': '#47D7AC',
    'medium_blue': '#8DCDE2',
    'light_blue': '#CCf5FC',
    'medium_orange': '#FF8200',
    'light_orange': '#FFB25B',
    'medium_purple': '#512D6D',
    'light_purple': '#958DBE',
    'grey': '#A7A8AA',
}

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
            width=0.6
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

def plot_note_recall_by_sdoh(df_long, save_path="../results/figures/note_recall_by_sdoh.png"):
    """
    Plot recall of referral notes in documenting experienced SDoH needs.
    Recall = proportion of experienced needs that are documented in the note.

    Parameters:
    - df_long: Long-format DataFrame with 'structured_experienced', 'free_text_documented',
               'sdoh_category', and 'dataset' columns.
    - save_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Filter to experienced needs only
    df_exp = df_long[df_long["structured_experienced"] == True].copy()
    df_exp = df_exp[~((df_exp["dataset"] == "HIU") & (df_exp["sdoh_category"] == "foodaccess"))]

    # Compute recall per SDoH per dataset
    grouped = df_exp.groupby(["dataset", "sdoh_category"])
    recall_df = grouped["free_text_documented"].mean().reset_index()
    recall_df["Recall (%)"] = recall_df["free_text_documented"] * 100

    # Label mapping for display
    label_map = {
        "housing": "Housing",
        "finances": "Finances",
        "loneliness": "Loneliness",
        "foodaccess": "Food Access"
    }
    recall_df["SDoH Category"] = recall_df["sdoh_category"].map(label_map)

    # Define color palette (only purple and red used here)
    bar_colors = {
        True: colour_palette["medium_purple"],
        False: colour_palette["ifrc_red"]
    }

    # Plot
    sns.set_style("ticks")
    g = sns.catplot(
        data=recall_df,
        kind="bar",
        x="SDoH Category",
        y="Recall (%)",
        col="dataset",
        palette=[colour_palette["medium_purple"]],
        height=6,
        aspect=0.8
    )

    for ax in g.axes.flat:
        for bar in ax.patches:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10
            )
        ax.set_ylim(0, 100)
        ax.set_ylabel("Recall of referral notes (%)", fontsize=12)
        ax.set_xlabel("SDoH Category", fontsize=12)
        ax.tick_params(axis='x', labelrotation=0, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_title(ax.get_title().replace("dataset = ", ""), fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()