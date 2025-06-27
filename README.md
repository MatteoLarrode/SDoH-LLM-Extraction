# Social Determinants of Health (SDoH) Extraction Project

## Overview
This project focuses on extracting Social Determinants of Health (SDoH) from clinical notes using natural language processing and large language models, based on recent research advances in multi-institutional SDoH extraction.

## Project Structure
```
.
├── data/                        # All raw and processed datasets
│   ├── raw/                    # External or source data (e.g. BRC, MIMIC-III, Label Studio)
│   └── processed/              # Cleaned and merged data used in modeling and analysis
│       ├── annotations/       # Human-generated SDoH annotations
│       ├── brc-cleaned/       # Preprocessed BRC referral notes
│       └── merged/            # Joined datasets (e.g. SNAP/HIU + referrals)

├── notebooks/                  # Jupyter notebooks for experimentation and prototyping
│   ├── 01_data_cleaning.ipynb
│   ├── 02_annotation.ipynb
│   ├── 03_extraction_pipeline.ipynb
│   ├── 04_fine_tuning.ipynb
│   └── 05_analysis.ipynb

├── src/                        # Core source code for annotation, modeling, and evaluation
│   ├── annotation/            # Annotation parsing, adjudication, inter-annotator agreement
│   ├── classification/        # Prompting, model setup, classification helpers
│   ├── config/                # Constants, global parameters, prompt configs
│   ├── data_cleaning/         # Text preprocessing and referral cleaning functions
│   └── evaluation/            # Metric calculations for SDoH extraction performance

├── scripts/                    # Scripts for batch processing and automated runs
│   ├── sdoh-extraction/       # Scripts to extract SDoH with different models/settings
│   └── evaluation/            # Scripts to evaluate predictions against annotations

├── results/                    # Outputs from experiments and model evaluation
│   ├── annotation_evaluation/ # Detailed metrics for level 1 and 2 annotation performance
│   ├── batch_results/         # Outputs from multi-model or multi-batch evaluations
│   └── figures/               # Saved visualizations (e.g. plots, charts)

├── streamlit_app/             # Streamlit interface for visualizing and interacting with results
│   ├── app.py                 # Main Streamlit app entrypoint
│   ├── config.py              # Streamlit app configuration
│   └── components/           # Modular UI components (e.g. data browser, model runner, tabs)

├── environment.yml            # Conda environment definition
├── README.md                  # Project documentation and structure
├── .gitignore                 # Files and folders to exclude from version control
└── tree.ipynb                 # Diagnostic notebook to print repo structure
```

## Research Foundation
This project builds upon recent advances in SDoH extraction, particularly:
- Multi-institutional approaches for better generalizability (Keloth et al., 2025)
- Large language models for clinical text processing
- Comprehensive annotation of 21+ SDoH factors
- Cross-dataset evaluation and transfer learning

## Setup
1. Clone this repository
2. Source conda: `source /opt/anaconda/etc/profile.d/conda.sh`
3. Create environment: `conda create -n sdoh python=3.10`
4. Activate environment: `conda activate sdoh`
5. Install dependencies: `pip install -r requirements.txt`

## SDoH Factors Covered
Based on recent literature, this project aims to extract:
- Demographics: Sex, Race, Age
- Socioeconomic: Employment, Education, Financial issues, Insurance
- Social: Marital status, Living status, Social support, Isolation
- Behavioral: Alcohol, Drugs, Smoking
- Environmental: Housing, Geographic location
- Health-related: Adverse childhood experiences, Physical/sexual abuse

## Usage
[To be developed as project progresses]

## Author
Matteo Larrodé - Oxford Internet Institute