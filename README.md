# Electric Vehicle Price Prediction in Vietnam

End-to-end pipeline for predicting electric vehicle prices in the Vietnamese market, from web scraping to regression modeling. Built as part of the AIL301m course at FPT University.

> **v2 Feature Engineering** — The car pipeline has been upgraded with brand/model normalization (80+ mappings), interaction features, reduced sparsity via target encoding, and improved data quality. See [Feature Engineering v2](#feature-engineering-v2-cars) for details.

## Project Structure

```
ev_car/
├── configs/              # LLM extraction configs (Ollama, OpenAI)
├── data/
│   ├── raw/              # Scraped CSVs and JSONs per website
│   ├── interim/          # Harmonized, cleaned, LLM-extracted data
│   └── processed/        # Model-ready train/test splits, scaler (cars & two-wheelers)
├── docs/
│   └── research/         # IEEE paper, Beamer presentation, figures
├── notebooks/
│   ├── prep_eda.ipynb    # Data preparation overview
│   ├── eda.ipynb         # Exploratory data analysis
│   ├── feature_engineering.ipynb  # Feature pipeline walkthrough
│   ├── project_report.ipynb      # Full car project report with plots
│   └── two_wheelers_analysis.ipynb # Full two-wheeler project report with plots
├── reports/              # Generated PDFs/CSVs from notebooks & models (cars & two-wheelers)
├── scripts/
│   ├── harmonize_datasets.py     # Merge raw car CSVs into unified schema
│   ├── preprocess_rule_based.py  # Rule-based cleaning for cars
│   ├── llm_extraction_local.py   # Local LLM feature extraction for cars
│   ├── two_wheelers_pipeline.py  # Merge, clean, and align raw two-wheeler JSONs
│   ├── two_wheelers_llm_extract.py # LLM extraction for two-wheelers
│   └── ...               # Scrapers (bonbanh, chotot, otodien, VFluot, etc.)
└── src/
    ├── features/
    │   ├── build_features.py     # Feature engineering pipeline for cars (v2)
    │   ├── build_features_twowheeler.py # Feature engineering pipeline for two-wheelers
    │   ├── ev_specs.py           # EV specs lookup builder (battery, range, power)
    │   └── target_encoder.py     # LOO target encoder with smoothing
    └── models/
        ├── run_all.py            # Train & benchmark all car models
        ├── train_twowheeler.py   # Train & benchmark all two-wheeler models
        ├── linear_regression.py  # Ridge regression
        ├── svr.py                # Support vector regression
        ├── random_forest.py      # Random forest
        ├── xgboost_model.py      # XGBoost
        ├── benchmark.py          # Cross-model comparison for cars
        └── plotting.py           # Model evaluation plots for cars
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

## Pipelines

### A. Electric Cars Pipeline
Run from the project root, in order:

```bash
# 1. Merge raw scraped data into unified schema
uv run python scripts/harmonize_datasets.py

# 2. Rule-based cleaning (standardize columns, fix types)
uv run python scripts/preprocess_rule_based.py

# 3. LLM feature extraction (optional — requires Ollama with Qwen 2.5)
uv run python scripts/llm_extraction_local.py

# 4. Feature engineering v2 (normalize → clean → encode → split → scale)
uv run python -m src.features.build_features

# 5. Train all car models and run benchmark (without LightGBM)
uv run python -m src.models.run_all --models linear_regression svr random_forest xgboost --benchmark
```

### B. Two-Wheelers (Bicycles & Motorbikes) Pipeline
Run from the project root, in order:

```bash
# 1. Load and clean raw two-wheelers scraped JSONs
uv run python scripts/two_wheelers_pipeline.py

# 2. Merge cleaned listings with LLM-extracted features
uv run python scripts/two_wheelers_pipeline.py --merge-only

# 3. Feature engineering (ratio-to-median outlier scaling, encoding, split)
uv run python -m src.features.build_features_twowheeler

# 4. Train and benchmark all two-wheeler models
uv run python -m src.models.train_twowheeler
```

## Notebooks

Execute after the pipelines complete to pre-render output:

```bash
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 notebooks/prep_eda.ipynb --output prep_eda.ipynb
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 notebooks/eda.ipynb --output eda.ipynb
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 notebooks/feature_engineering.ipynb --output feature_engineering.ipynb
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 notebooks/project_report.ipynb --output project_report.ipynb
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 notebooks/two_wheelers_analysis.ipynb --output two_wheelers_analysis.ipynb
```

Or open them interactively:

```bash
uv run jupyter lab
```

## Paper & Presentation (Cars)

After running the pipeline and notebooks, copy figures and compile:

```cmd
:: Copy figures to paper directory
copy reports\project_report\before_after_comparison.png docs\research\figures\
copy reports\project_report\chotot_price_comparison.png docs\research\figures\
copy reports\project_report\model_r2_comparison.png docs\research\figures\
copy reports\project_report\model_rmse_comparison.png docs\research\figures\
copy reports\project_report\records_per_source.png docs\research\figures\
copy reports\project_report\segment_rmse.png docs\research\figures\
copy reports\feature_engineering\byd_removal.png docs\research\figures\
copy reports\feature_engineering\dedup_analysis.png docs\research\figures\
copy reports\feature_engineering\encoding_comparison.png docs\research\figures\
copy reports\feature_engineering\ev_specs_coverage.png docs\research\figures\
copy reports\feature_engineering\feature_correlation_matrix.png docs\research\figures\
copy reports\feature_engineering\feature_importance_summary.png docs\research\figures\
copy reports\model_benchmark\random_forest_actual_vs_predicted.png docs\research\figures\
copy reports\model_benchmark\random_forest_feature_importance.png docs\research\figures\
copy reports\model_benchmark\random_forest_residual_distribution.png docs\research\figures\

:: Compile paper (run twice for cross-references)
cd docs\research
pdflatex -interaction=nonstopmode bare_jrnl_new_sample4.tex
pdflatex -interaction=nonstopmode bare_jrnl_new_sample4.tex

:: Compile presentation
pdflatex -interaction=nonstopmode presentation.tex
pdflatex -interaction=nonstopmode presentation.tex
```

## Feature Engineering v2 (Cars)

The car feature engineering pipeline was upgraded to address data quality issues and reduce feature sparsity:

| Improvement | Before (v1) | After (v2) |
|---|---|---|
| Brand normalization | None (`VinFast`/`Vinfast` treated as different) | 15+ mappings applied before dedup |
| Base model normalization | None (`VFe34`/`VF e34`/`E34` all separate) | 80+ mappings unifying variants |
| `base_model` encoding | One-hot (22 sparse columns) | Target encoding (1 dense column) |
| `exterior_color` encoding | One-hot (11 columns) | Grouped into 4 categories (3 columns) |
| `brand_body` interaction | Created then dropped | Target-encoded (1 column) |
| `seats` column | Dropped | Kept (imputed per base_model) |
| Interaction features | None | 5 new: `mileage_x_age`, `range_per_kwh`, `hp_per_kwh`, `is_luxury_brand`, `battery_tier` |
| `has_aftermarket_mods` | Kept (very low signal) | Dropped |
| Estimated total features | ~67 | ~30 (denser, higher signal) |

## Key Results

### Electric Cars Benchmarking
| Model | RMSE (M VND) | MAE (M VND) | R² | Adj R² |
|---|---|---|---|---|
| **Random Forest** | **325** | **152** | **0.682** | **0.668** |
| XGBoost | 361 | 180 | 0.610 | 0.592 |
| SVR | 396 | 136 | 0.530 | 0.509 |
| Ridge | 330B | 14.3B | -328K | - |

Data quality fixes (chotot.com price correction, deduplication, BYD removal) reduced RMSE by **67%** (1.02B → 325M VND) — far more impactful than model selection.

> **Note:** Results above are from the v1 feature set. Re-running with v2 features is expected to further improve R² and reduce RMSE/MAE thanks to brand/model normalization, reduced sparsity, and interaction features.

### Two-Wheelers Benchmarking
| Model | Test RMSE (M VND) | Test MAE (M VND) | Test R² | Test MAPE (%) |
|---|---|---|---|---|
| Linear Regression | 6.97 | 4.50 | 0.620 | 57.47% |
| SVR | 5.54 | 3.82 | 0.760 | 57.03% |
| Random Forest | 5.61 | 3.93 | 0.754 | 61.00% |
| **XGBoost** | **5.29** | **3.69** | **0.782** | **57.51%** |


For two-wheelers, **XGBoost** achieves the highest performance with a Test R² of **0.782**, closely followed by SVR and LightGBM.
All two-wheeler evaluation plots are saved to `reports/two_wheelers/`.
