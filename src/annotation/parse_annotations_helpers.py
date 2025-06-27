# ================================================
# === HELPER FUNCTIONS FOR PARSING ANNOTATIONS ===
# ================================================
import json
import pandas as pd
from typing import List, Dict, Any

def parse_labelstudio_json(json_path: str) -> pd.DataFrame:
    """
    Parse a Label Studio JSON export and return a flat DataFrame
    with one row per (sentence, label) pair.

    Parameters:
    -----------
    json_path : str
        Path to the JSON file exported from Label Studio.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with columns:
        Sentence, Case Reference, SDoH, Polarity, Annotator ID, Annotation ID,
        Task ID, Project ID, Lead Time, Created At, Updated At
    """
    with open(json_path, "r") as f:
        tasks = json.load(f)

    records = []

    for task in tasks:
        sentence = task["data"].get("text", "").strip()
        case_ref = task.get("meta", {}).get("case_reference", None)
        task_id = task.get("id")
        project_id = task.get("project")
        
        for annotation in task.get("annotations", []):
            annotator_id = annotation.get("completed_by")
            annotation_id = annotation.get("id")
            lead_time = annotation.get("lead_time")
            created_at = annotation.get("created_at")
            updated_at = annotation.get("updated_at")

            for result in annotation.get("result", []):
                from_name = result.get("from_name")
                choices = result.get("value", {}).get("choices", [])
                
                if from_name == "no_sdoh" and "True" in choices:
                    records.append({
                        "Sentence": sentence,
                        "Case Reference": case_ref,
                        "SDoH": "No SDoH",
                        "Polarity": None,
                        "Annotator": "Main_Author",
                        "Annotation ID": annotation_id,
                        "Task ID": task_id,
                        "Project ID": project_id,
                        "Lead Time": lead_time,
                        "Created At": created_at,
                        "Updated At": updated_at
                    })
                else:
                    for choice in choices:
                        records.append({
                            "Sentence": sentence,
                            "Case Reference": case_ref,
                            "SDoH": from_name.replace("_", " ").capitalize(),
                            "Polarity": choice,
                            "Annotator": "Main_Author",
                            "Annotation ID": annotation_id,
                            "Task ID": task_id,
                            "Project ID": project_id,
                            "Lead Time": lead_time,
                            "Created At": created_at,
                            "Updated At": updated_at
                        })

    return pd.DataFrame(records)

def parse_csv_annotations(filepath: str, annotator_name: str = "Cat") -> pd.DataFrame:
    """
    Parse CSV annotations into a long-format DataFrame.
    Applies SDoH label normalization for consistency.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file.
    annotator_name : str
        Name or ID of the annotator (default: 'Cat').

    Returns:
    --------
    pd.DataFrame with columns:
        Sentence, SDoH, Polarity, Annotator
    """
    df = pd.read_csv(filepath)

    # Label normalization map
    normalize_sdoh = {
        "Loneliness/Social support": "Loneliness",
        "Access to food": "Food",
        "Digital literacy": "Digital"
    }

    long_records = []

    for _, row in df.iterrows():
        sentence = row["Sentence"].strip()
        for i in range(1, 5):
            sdoh = row.get(f"SDoH {i}")
            polarity = row.get(f"Polarity {i}")
            if pd.notna(sdoh):
                sdoh_clean = normalize_sdoh.get(sdoh.strip(), sdoh.strip())
                if sdoh_clean == "No SDoH":
                    long_records.append({
                        "Sentence": sentence,
                        "SDoH": "No SDoH",
                        "Polarity": None,
                        "Annotator": annotator_name
                    })
                    break  # Stop after No SDoH
                elif pd.notna(polarity):
                    long_records.append({
                        "Sentence": sentence,
                        "SDoH": sdoh_clean,
                        "Polarity": polarity.strip(),
                        "Annotator": annotator_name
                    })

    return pd.DataFrame(long_records)