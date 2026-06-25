# Electric Vehicle Price Prediction in Vietnam

End-to-end pipeline for predicting electric vehicle prices in the Vietnamese market, from web scraping to regression modeling. Built as part of the AIL301m course at FPT University.

## Project Structure

```
ev_car/
├── configs/              # LLM extraction configs (Ollama, OpenAI)
├── data/
│   ├── raw/              # Scraped CSVs per website
│   ├── interim/          # Harmonized, cleaned, LLM-extracted data
│   └── processed/        # Model-ready train/test splits, scaler
├── docs/
│   └── research/         # IEEE paper, Beamer presentation, figures
├── notebooks/
│   ├── prep_eda.ipynb    # Data preparation overview
│   ├── eda.ipynb         # Exploratory data analysis
│   ├── feature_engineering.ipynb  # Feature pipeline walkthrough
│   └── project_report.ipynb      # Full project report with plots
├── reports/              # Generated PDFs from notebooks & models
├── scripts/
│   ├── harmonize_datasets.py     # Merge raw CSVs into unified schema
│   ├── preprocess_rule_based.py  # Rule-based cleaning
│   ├── llm_extraction_local.py   # Local LLM feature extraction
│   └── ...               # Scrapers (bonbanh, chotot, otodien, VFluot)
└── src/
    ├── features/
    │   ├── build_features.py     # Feature engineering pipeline
    │   ├── target_encoder.py     # LOO target encoder with smoothing
    │   └── ev_specs.py           # EV specs lookup from otodien.vn
    ├── models/
    │   ├── run_all.py            # Train & benchmark all models
    │   ├── linear_regression.py  # Ridge regression
    │   ├── svr.py                # Support vector regression
    │   ├── random_forest.py      # Random forest
    │   ├── xgboost_model.py      # XGBoost
    │   ├── benchmark.py          # Cross-model comparison
    │   └── plotting.py           # Model evaluation plots
    └── viz/
        └── style.py              # Shared plot style (palette, rcParams)
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync --dev

# Copy environment variables (for LLM API keys if using cloud extraction)
cp .env.example .env
```

LaTeX compilation requires a TeX distribution (e.g., [MiKTeX](https://miktex.org/) on Windows).

## Pipeline

Run from the project root, in order:

```bash
# 1. Merge raw scraped data into unified schema
uv run python scripts/harmonize_datasets.py

# 2. Rule-based cleaning (standardize columns, fix types)
uv run python scripts/preprocess_rule_based.py

# 3. LLM feature extraction (optional — requires Ollama with Qwen 2.5)
uv run python scripts/llm_extraction_local.py

# 4. Feature engineering (clean → encode → split → scale)
uv run python -m src.features.build_features

# 5. Train all models and run benchmark
uv run python -m src.models.run_all --models linear_regression svr random_forest xgboost --benchmark
```

## Notebooks

Execute after the pipeline completes:

```bash
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 notebooks/prep_eda.ipynb --output prep_eda.ipynb
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 notebooks/eda.ipynb --output eda.ipynb
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 notebooks/feature_engineering.ipynb --output feature_engineering.ipynb
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 notebooks/project_report.ipynb --output project_report.ipynb
```

Or open them interactively:

```bash
uv run jupyter lab
```

## Paper & Presentation

After running the pipeline and notebooks, copy figures and compile:

```cmd
:: Copy figures to paper directory
copy reports\project_report\before_after_comparison.pdf docs\research\figures\
copy reports\project_report\chotot_price_comparison.pdf docs\research\figures\
copy reports\project_report\model_r2_comparison.pdf docs\research\figures\
copy reports\project_report\model_rmse_comparison.pdf docs\research\figures\
copy reports\project_report\records_per_source.pdf docs\research\figures\
copy reports\project_report\segment_rmse.pdf docs\research\figures\
copy reports\feature_engineering\byd_removal.pdf docs\research\figures\
copy reports\feature_engineering\dedup_analysis.pdf docs\research\figures\
copy reports\feature_engineering\encoding_comparison.pdf docs\research\figures\
copy reports\feature_engineering\ev_specs_coverage.pdf docs\research\figures\
copy reports\feature_engineering\feature_correlation_matrix.pdf docs\research\figures\
copy reports\feature_engineering\feature_importance_summary.pdf docs\research\figures\
copy reports\model_benchmark\random_forest_actual_vs_predicted.pdf docs\research\figures\
copy reports\model_benchmark\random_forest_feature_importance.pdf docs\research\figures\
copy reports\model_benchmark\random_forest_residual_distribution.pdf docs\research\figures\

:: Compile paper (run twice for cross-references)
cd docs\research
pdflatex -interaction=nonstopmode bare_jrnl_new_sample4.tex
pdflatex -interaction=nonstopmode bare_jrnl_new_sample4.tex

:: Compile presentation
pdflatex -interaction=nonstopmode presentation.tex
pdflatex -interaction=nonstopmode presentation.tex
```

## Key Results

| Model | RMSE (M VND) | MAE (M VND) | R² |
|---|---|---|---|
| Ridge | 381 | 115 | 0.636 |
| SVR | 369 | 94 | 0.660 |
| **Random Forest** | **332** | **89** | **0.725** |
| XGBoost | 383 | 103 | 0.633 |

Data quality fixes (chotot.com price correction, deduplication, BYD removal) reduced RMSE by **67%** (1.02B → 332M VND) — far more impactful than model selection (15% difference between best and worst).
