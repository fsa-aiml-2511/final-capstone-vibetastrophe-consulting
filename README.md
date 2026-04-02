[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/wtyFW8Ul)
# Final Capstone Project — AI Consulting Team

## Project Structure

```
├── data/                          # Your datasets (DO NOT commit large files)
│   ├── raw/                       # Original, unmodified data
│   └── processed/                 # Cleaned, feature-engineered data
│
├── models/                        # One folder per model — keep it clean
│   ├── model1_traditional_ml/     # Traditional ML (e.g., XGBoost, Random Forest)
│   │   ├── train.py               # Training script
│   │   ├── predict.py             # Prediction script (loads saved model, outputs CSV)
│   │   └── saved_model/           # Serialized model files (.joblib, .pkl)
│   │
│   ├── model2_deep_learning/      # Deep Neural Network (TensorFlow/Keras)
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── saved_model/
│   │
│   ├── model3_cnn/                # Convolutional Neural Network (image classification)
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── saved_model/
│   │
│   ├── model4_nlp_classification/ # NLP text classification
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── saved_model/
│   │
│   └── model5_innovation/         # Your team's innovation model
│       ├── train.py
│       ├── predict.py
│       └── saved_model/
│
├── notebooks/                     # EDA, experimentation, prototyping ONLY
│
├── pipelines/                     # Data preprocessing & feature engineering
│   └── data_pipeline.py           # Shared data loading and cleaning functions
│
├── webapp/                        # Your deployed Streamlit web application
│   └── app.py                     # Main application (streamlit run webapp/app.py)
│
├── test_data/                     # Instructor test data goes here (DO NOT MODIFY)
│
├── output_templates/              # REQUIRED output format for each model
│   ├── model1_results_template.csv
│   ├── model2_results_template.csv
│   ├── model3_results_template.csv
│   ├── model4_results_template.csv
│   └── model5_results_template.csv
│
├── bulk_test.py                   # Test script for all models (--model N or --all)
├── requirements.txt               # All dependencies (pip install -r requirements.txt)
├── weekly-sprint-template.md      # Template for weekly check-ins (submit via Slack)
├── .gitignore                     # Ignore large files, data, model weights
└── README.md                      # This file — update with your project details
```

### What Each Folder Is For

| Folder | What Goes Here | Who Uses It |
|--------|---------------|-------------|
| `data/raw/` | Original CSV files and images downloaded from the shared Google Drive link. Do not modify these files. | Everyone |
| `data/processed/` | Cleaned, transformed, feature-engineered versions of the raw data. This is what your models actually train on. | Data Engineering Lead |
| `models/model1_*/` through `model5_*/` | Each model gets its own folder. `train.py` trains the model, `predict.py` loads and runs it, `saved_model/` stores the trained weights. | Each model's lead owner |
| `notebooks/` | Jupyter notebooks for exploratory data analysis (EDA) and experimentation. Notebooks are for exploration only — final code goes in `.py` files. | Everyone |
| `pipelines/data_pipeline.py` | Shared functions that all models use — data loading, cleaning, feature engineering, train/test splitting. Write it once, import everywhere. | Data Engineering Lead |
| `webapp/` | Your deployed Streamlit web app that integrates all 5 models. Run with `streamlit run webapp/app.py`. Deploy to [Streamlit Community Cloud](https://streamlit.io/cloud) for free. | Everyone (shared responsibility) |
| `test_data/` | **Do not touch.** The instructor will place test data here during evaluation. Your `predict.py` scripts read from this folder. | Instructor only |
| `output_templates/` | Example CSVs showing the **exact** output format each model must produce. Open these and match them precisely. | Reference — do not modify |

---

## Critical Rules

### 1. Output Format is Non-Negotiable

Each model's `predict.py` must output a CSV that **exactly matches** the template in `output_templates/`. Same column names. Same data types. Same order. No exceptions.

Your evaluation script will fail if your output doesn't match. This is intentional — in the real world, your client's systems expect data in a specific format.

### Output Format Reference

Each model has a different output CSV format. Use this table as a quick reference — your `predict.py` must produce columns that **exactly match** these.

| Model | Column 1 | Column 2 | Column 3 | Column 4 | Column 5 |
|-------|----------|----------|----------|----------|----------|
| **Model 1** (Traditional ML) | `id` | `prediction` | `probability` | `confidence` | — |
| **Model 2** (Deep Learning) | `id` | `prediction` | `probability` | `confidence` | — |
| **Model 3** (CNN) | `image_id` | `predicted_class` | `confidence` | — | — |
| **Model 4** (NLP) | `id` | `predicted_class` | `confidence` | — | — |
| **Model 5** (Innovation) | `id` | `prediction` | `confidence` | `metric_name` | `metric_value` |

**Key differences to watch for:**
- **Model 3 uses `image_id`**, not `id` — because predictions correspond to image files (e.g., `img_0001.png`), not numeric record IDs.
- **Models 3 and 4 use `predicted_class`**, not `prediction` — because they output categorical class labels rather than binary/numeric predictions.
- **Model 5 includes extra columns** (`metric_name`, `metric_value`) for your custom evaluation metric.
- **Models 1 and 2 include `probability`** (the raw model output probability) in addition to `confidence`.

**Note:** The example values in the templates are illustrative. Use your scenario's actual class labels (e.g., severity 1-4 for smart city, 0/1 for healthcare binary predictions, actual category names for NLP).

Always check `output_templates/` for the authoritative format. If your output doesn't match, you will lose 5 points per model.

### 2. No Notebooks as Final Deliverables

Notebooks are for exploration. Your final model code must be in `.py` files inside the `models/` directory. Models submitted only as notebooks will receive a **-5 point deduction**.

### 3. Models Must Be Saved and Loadable

Your `predict.py` scripts must:
- Load a pre-trained model from `saved_model/`
- Accept test data as input
- Output predictions to `test_data/modelN_results.csv`

Do NOT retrain during prediction. If your model requires retraining to make predictions, it's not production-ready.

### 4. File Naming Conventions

- Use `snake_case` for all file and folder names
- Model artifacts: `model_name.joblib`, `model_name.h5`, `model_name.keras`
- No spaces in file names. Ever.
- No duplicate versions (`model_v2_final_FINAL.py` — don't do this)

### 5. Data Stays Out of Git

Large data files and model weights should NOT be committed to git. Use `.gitignore`. Your `data/raw/` folder should contain only a README explaining where to get the data.

---

## How Your Models Will Be Evaluated

1. I will place test data in your `test_data/` folder
2. I will run `python bulk_test.py --all`
3. The script calls each model's `predict.py` with the test data
4. Your predictions are compared against ground truth I hold back
5. Metrics are calculated automatically
6. **If your script crashes, that model scores 0**

### What "Runs" Means

```bash
pip install -r requirements.txt    # Must work without errors
python bulk_test.py --all          # Must produce test_data/model{1-5}_results.csv
```

No manual steps. No "you need to run this notebook first." No "change this path." It either runs or it doesn't.

---

## Required Models (5 Total)

| # | Model Type | What You Build | Key Metric |
|---|-----------|----------------|------------|
| 1 | **Traditional ML** | Classical ML algorithm (XGBoost, Random Forest, etc.) | See scenario spec |
| 2 | **Deep Learning** | Neural network on tabular/structured data (TensorFlow/Keras) | See scenario spec |
| 3 | **CNN** | Image classification with convolutional neural network | See scenario spec |
| 4 | **NLP Classification** | Text classification using NLP techniques | See scenario spec |
| 5 | **Innovation** | Your team's choice — surprise us | Your defined metric |

---

## Team Members

| Role | Name | GitHub Username |
|------|------|-----------------|
| Data Engineering Lead | Kalpana | |
| ML / DNN Lead | | |
| CNN Lead | | |
| NLP Lead | | |

---

## Getting Started

1. **Read your scenario spec** — understand the business problem and datasets
2. **Read `preprocessing_hints.py` carefully** — this file is in your scenario folder (healthcare/ or smart_city/), not in this repo. It contains critical data preparation steps including target variable creation, category mapping, and class imbalance warnings. Skipping this will cost you significant points.
3. **Explore the data** — use `notebooks/eda.ipynb` to understand what you're working with
4. **Build your data pipeline** — `pipelines/data_pipeline.py` for shared preprocessing
5. **Develop models** — one team member leads each, everyone contributes
6. **Test your outputs** — make sure they match the templates BEFORE submission
7. **Build the web app** — integrate all models into a single interface
8. **Deploy** — deploy to [Streamlit Community Cloud](https://streamlit.io/cloud) (free). Push your repo to GitHub, connect it to Streamlit Cloud, and your app gets a public URL

---

## Team Roles

| Role | Primary Responsibility | Models |
|------|----------------------|--------|
| **Data Engineering Lead** | Data cleaning, feature engineering, pipelines | Supports all |
| **ML / DNN Lead** | Traditional ML + Deep Learning models | Model 1 & 2 |
| **CNN Lead** | Image classification, computer vision | Model 3 |
| **NLP Lead** | Text classification | Model 4 |

**Model 5 (Innovation)** is a shared team responsibility. **Everyone** contributes to the web app and presentation.

---

## Weekly Sprint Check-Ins

Every week, your team must submit a **sprint check-in** via Slack before your weekly meeting with the instructor. Use the `weekly-sprint-template.md` included in this repo.

This includes:
- What you accomplished that week
- What you're working on next
- Any blockers or challenges
- Updated model status tracker

These check-ins are part of your **Collaboration & Process** grade. Consistent, honest updates show professionalism. Silence until demo day does not.

---

## Submission Checklist

Before your final push, verify:

- [ ] `pip install -r requirements.txt` works cleanly
- [ ] `python bulk_test.py --all` runs without errors
- [ ] All 5 `test_data/modelN_results.csv` files are generated
- [ ] Output CSVs match the templates exactly (column names, types)
- [ ] No hardcoded absolute paths (use relative paths or config)
- [ ] No data leakage between train/test splits
- [ ] Models load from saved files (no retraining during prediction)
- [ ] Web app runs locally with `streamlit run webapp/app.py`
- [ ] README updated with your project-specific details
- [ ] `.gitignore` excludes large data and model files



## Installing `UV`

Install `uv` on the Mac:

```sh
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows PowerShell:

```powershell
$ powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installing dependencies

Use `uv` to install Python 3.11 and install a vistual enviroment.

```sh
$ uv venv --python 3.11 .venv
$ uv pip install -r requirements.txt
```

Start the virtual enviroment on the Mac:

```sh
$ source .venv/bin/activate
```

on Windows PowerShell:

```sh
$ .venv\Scripts\Activate.ps1
```

## Deactivate

To deactivate on a Mac, use the following command:

```sh
deactivate
```

On Windows:

```sh
venv/Scripts/activate.ps1 
```