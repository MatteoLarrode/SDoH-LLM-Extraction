import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import re
from IPython.display import display, Markdown
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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

def set_visualization_style():
    plt.style.use('seaborn-v0_8-colorblind')
    #font_path = {include font path}
    #font_manager.fontManager.addfont(font_path)
    #prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = 'sans-serif'
    #plt.rcParams['font.sans-serif'] = prop.get_name()
    plt.rcParams.update({
        'text.usetex': False,
        #'font.family': 'serif',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 1.5,
        'lines.markersize': 8,
        'figure.figsize': (10, 6),
        'axes.grid': False, 
        'axes.spines.top': False,  # Remove top spine
        'axes.spines.right': False,  # Remove right spine
        # Add this line to use ASCII hyphen instead of Unicode minus
        'axes.unicode_minus': False
    })

def compute_proportions(df):
    df = df.copy()
    df["sdoh_category"] = df["sdoh_category"].map({
        "housing": "Housing",
        "finances": "Finances",
        "loneliness": "Loneliness",
        "foodaccess": "Food Access"
    })

    # Count totals per category+dataset to compute proportions
    count_df = df.groupby(["dataset", "sdoh_category", "documentation_status"]).size().reset_index(name="count")

    # Compute proportion within each dataset x category group
    count_df["total"] = count_df.groupby(["dataset", "sdoh_category"])["count"].transform("sum")
    count_df["proportion"] = count_df["count"] / count_df["total"]

    return count_df

def plot_single_dataset_proportion(df, dataset, save_path, figsize=(5, 6)):
    df = df[df["dataset"] == dataset].copy()

    if dataset == "HIU":
        df = df[df["sdoh_category"] != "Food Access"]
        title = "a) HIU"
        category_order = ["Loneliness", "Housing", "Finances"]
    else:
        title = "b) SCHH+SP"
        category_order = ["Loneliness", "Housing", "Finances", "Food Access"]

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

    sns.set_style("ticks")
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df,
        x="sdoh_category",
        y="proportion",
        hue="documentation_status",
        order=category_order,
        hue_order=label_order,
        palette=label_color_map,
        alpha=0.85
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y * 100)}%'))
    ax.tick_params(axis='x', labelrotation=0, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    if dataset == "HIU":
        ax.set_ylabel("Proportion of Service Users", fontsize=12)
        ax.legend_.remove()
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.legend_.set_title("")  # remove legend title
        ax.legend_.set_frame_on(False)
        ax.legend(fontsize=11)

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_stacked_horizontal_single(
    df_long,
    dataset,
    save_path,
    figsize=(6, 4)
):
    """
    Create a horizontal stacked barplot for a single dataset.
    Each bar represents experienced needs split by documentation in referral notes.
    """
    # Filter to experienced SDoH in this dataset
    df_exp = df_long[
        (df_long["structured_experienced"] == True) & (df_long["dataset"] == dataset)
    ].copy()

    if dataset == "HIU":
        df_exp = df_exp[df_exp["sdoh_category"] != "foodaccess"]

    # Count by SDoH and documentation status
    counts = (
        df_exp.groupby(["sdoh_category", "free_text_documented"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={True: "Documented", False: "Not Documented"})
        .reset_index()
    )

    # Label mapping
    label_map = {
        "housing": "Housing",
        "finances": "Finances",
        "loneliness": "Loneliness",
        "foodaccess": "Food"
    }
    counts["SDoH Category"] = counts["sdoh_category"].map(label_map)

    # Order
    ordered_sdoh = ["Loneliness", "Housing", "Finances"]
    if dataset != "HIU":
        ordered_sdoh.append("Food")

    counts = counts.set_index("SDoH Category").loc[ordered_sdoh].reset_index()

    # Compute recall %
    counts["Total"] = counts["Documented"] + counts["Not Documented"]
    counts["Recall"] = counts["Documented"] / counts["Total"] * 100

    bar_spacing = 0.6
    bar_pos = np.arange(len(counts)) * bar_spacing
    bar_height = 0.5

    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=figsize)

    # Reverse order: Documented first (left), then Not Documented (right)
    ax.barh(
        y=bar_pos,
        width=counts["Documented"],
        color=colour_palette["medium_purple"],
        label="Documented in referral note",
        alpha=0.7,
        height=bar_height
    )

    ax.barh(
        y=bar_pos,
        width=counts["Not Documented"],
        left=counts["Documented"],
        color=colour_palette["ifrc_red"],
        label="Undocumented in referral note",
        alpha=0.7,
        height=bar_height
    )

    # Add recall labels inside documented bar (white, centered)
    for y, row in zip(bar_pos, counts.itertuples()):
        doc_width = row.Documented
        recall_pct = f"{row.Recall:.0f}%"
        ax.text(
            x=doc_width/2,
            y=y,
            s=recall_pct,
            va="center",
            ha="center",
            fontsize=12,
            color="white"
        )

    ax.set_yticks(bar_pos)
    ax.set_yticklabels(ordered_sdoh, fontsize=11)

    # X-axis label only for SCHH+SP
    if dataset == "HIU":
        ax.set_xlabel("")
    else:
        ax.set_xlabel("Service users experiencing need", fontsize=12)

    ax.set_ylabel("")
    ax.set_title(
        "a) HIU" if dataset == "HIU" else "b) SCHH+SP", 
        fontsize=12,
        loc="left")
    ax.tick_params(axis='x', labelsize=11)

    sns.despine(ax=ax)

    # Legend for SCHH+SP only
    if dataset != "HIU":
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.35),  # pushed further down
            frameon=False,
            fontsize=12,
            ncol=2,
            title=""
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()