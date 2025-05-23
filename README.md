# Social Determinants of Health (SDoH) Extraction Project

## Overview
This project focuses on extracting Social Determinants of Health (SDoH) from clinical notes using natural language processing and large language models, based on recent research advances in multi-institutional SDoH extraction.

## Project Structure
```
sdoh-extraction/
├── data/
│   ├── raw/           # Original, immutable data
│   ├── processed/     # Cleaned and preprocessed data
│   └── external/      # External datasets and references
├── models/
│   ├── trained/       # Trained model files
│   └── checkpoints/   # Model checkpoints during training
├── notebooks/         # Jupyter notebooks for exploration and analysis
├── src/              # Source code for the project
│   └── sdoh_extraction/  # Main package
├── scripts/          # Standalone scripts
├── configs/          # Configuration files
├── results/
│   ├── figures/      # Generated plots and visualizations
│   └── outputs/      # Model outputs and results
└── docs/             # Documentation
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