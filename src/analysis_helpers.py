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
                doc_status = "Documented & Experienced"
            elif experienced and not documented:
                doc_status = "Undocumented but Experienced"
            elif not experienced and documented:
                doc_status = "Documented but Not Experienced"
            else:
                doc_status = "Neither Documented Nor Experienced"

            records.append({
                case_id_col: row[case_id_col],
                "sdoh_category": label,
                "structured_experienced": experienced,
                "free_text_documented": documented,
                "documentation_status": doc_status
            })

    return pd.DataFrame(records)