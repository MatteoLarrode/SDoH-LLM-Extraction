# ======================================================
# === HELPERS FOR INTER-ANNOTATOR AGREEMENT ANALYSIS ===
# ======================================================
import pandas as pd
import krippendorff
from typing import Tuple, List

def describe_mismatch(lab1: set, lab2: set) -> str:
    if lab1.issubset(lab2) or lab2.issubset(lab1):
        return "Partial match (subset)"
    return "Label disagreement"

def compare_annotations_krippendorff(df1: pd.DataFrame, df2: pd.DataFrame, label_col: str = "Sentence"):
    """
    Compare two annotation DataFrames and compute Krippendorff's alpha (nominal),
    plus return detailed sentence-level mismatches.

    Parameters:
    -----------
    df1, df2 : pd.DataFrame
        Long-format DataFrames with columns: Sentence, SDoH, Polarity, Annotator
    label_col : str
        Column used to match annotations (default: 'Sentence')

    Returns:
    --------
    alpha : float
        Krippendorff's alpha score for overall label agreement
    mismatches : pd.DataFrame
        Sentence-level rows with label differences between annotators
    """

    # Merge both annotator labels into one DataFrame
    df1["Label"] = df1["SDoH"] + " - " + df1["Polarity"].fillna("")
    df2["Label"] = df2["SDoH"] + " - " + df2["Polarity"].fillna("")
    annotators = df1["Annotator"].unique().tolist() + df2["Annotator"].unique().tolist()

    combined = pd.concat([df1, df2], ignore_index=True)
    all_labels = sorted(combined["Label"].unique())

    # Pivot to a binary presence matrix: [sentence x (annotator, label)]
    pivot = (
        combined.assign(value=1)
        .pivot_table(index=label_col, columns=["Annotator", "Label"], values="value", fill_value=0)
    )

    # Reconstruct per-sentence per-class binary rows (for Krippendorff)
    annotation_matrix = []

    for label in all_labels:
        ann1 = pivot.get((annotators[0], label), pd.Series(0, index=pivot.index))
        ann2 = pivot.get((annotators[1], label), pd.Series(0, index=pivot.index))

        # For each sentence, add a row with both annotators' judgments for this class
        for s in pivot.index:
            annotation_matrix.append([int(ann1.get(s, 0)), int(ann2.get(s, 0))])

    # Compute Krippendorff’s alpha over binary presence
    alpha = krippendorff.alpha(reliability_data=annotation_matrix, level_of_measurement="nominal")

    # Detect sentence-level mismatches
    grouped1 = df1.groupby(label_col)["Label"].apply(set)
    grouped2 = df2.groupby(label_col)["Label"].apply(set)
    shared_sentences = set(grouped1.index) & set(grouped2.index)

    mismatch_records = []
    for sent in shared_sentences:
        labels1 = grouped1.get(sent, set())
        labels2 = grouped2.get(sent, set())
        if labels1 != labels2:
            mismatch_records.append({
                "Sentence": sent,
                f"{annotators[0]} Labels": sorted(labels1),
                f"{annotators[1]} Labels": sorted(labels2),
                "Overlap": sorted(labels1 & labels2),
                "Mismatch Type": describe_mismatch(labels1, labels2)
            })

    mismatch_df = pd.DataFrame(mismatch_records)
    return alpha, mismatch_df

def compute_classwise_krippendorff(df1, df2, save_latex=False, latex_path="../results/latex_tables/classwise_krippendorff_round1.tex"):
    # Merge annotations and label decisions
    df_combined = pd.concat([df1, df2], ignore_index=True)
    df_combined["Annotator"] = df_combined["Annotator"].astype(str)

    all_sentences = df_combined["Sentence"].unique()
    annotators = df_combined["Annotator"].unique()
    sdoh_labels = sorted([s for s in df_combined["SDoH"].unique() if s != "No SDoH"])

    alpha_scores = []

    for sdoh in sdoh_labels:
        data = []
        for sentence in all_sentences:
            row = []
            for annotator in annotators:
                match = df_combined[
                    (df_combined["Sentence"] == sentence) &
                    (df_combined["SDoH"] == sdoh) &
                    (df_combined["Annotator"] == annotator)
                ]
                row.append(1 if not match.empty else 0)
            data.append(row)

        matrix = list(map(list, zip(*data)))  # rows = annotators, cols = sentences

        if len(matrix) > 1 and any(any(row) for row in matrix):
            alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement="nominal")
            alpha_scores.append((sdoh, round(alpha, 3)))
        else:
            alpha_scores.append((sdoh, "n/a"))

    alpha_df = pd.DataFrame(alpha_scores, columns=["SDoH Label", "Krippendorff's α (nominal)"])

    # Compute support: number of unique sentences where label applied by at least one annotator
    df_all = pd.concat([df1, df2])
    sdoh_counts = (
        df_all[df_all["SDoH"].isin(sdoh_labels)]
        .drop_duplicates(subset=["SDoH", "Sentence"])  # ← important line
        .groupby("SDoH")["Sentence"]
        .count()
        .reset_index(name="Support")
    )

    alpha_df = alpha_df.merge(sdoh_counts, left_on="SDoH Label", right_on="SDoH", how="left").drop(columns=["SDoH"])
    alpha_df = alpha_df[["SDoH Label", "Krippendorff's α (nominal)", "Support"]]

    if save_latex:
        latex_table = alpha_df.to_latex(index=False, float_format="%.2f")
        with open(latex_path, "w") as f:
            f.write(latex_table)

    return alpha_df