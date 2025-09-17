# SaleAnalysisApp

Retail Sales Intelligence dashboard for exploring Quebec store performance, customer behaviour, promotions, and retention trends. Built as a portfolio-ready DA/BA project with reusable data preparation utilities and storytelling baked into the Streamlit UI.

## Problem Statement
- Provide decision-ready insights on store and online sales using the January 2023 transaction dataset.
- Diagnose which locations, customer segments, and promotions contribute most to revenue and volume.
- Surface retention patterns so commercial teams can prioritise marketing follow-up and operational improvements.

## Datasets
- `combined_data_202301.csv` — line-level transaction log with customer, product, pricing, promotion, and transaction value fields.
- `qc_cities_coordinates.csv` — latitude/longitude reference used for mapping sales and retention metrics.

## Streamlit Experience
The app (`retail_app.py`) now contains seven tabs, each tied to business questions:
- **Sales Overview** – KPI deck, trend line, monthly revenue bars, and a geospatial sales footprint.
- **Customer Insights** – customer value segmentation, promotion usage, and top-customer leaderboards.
- **Transaction Value** – distribution of order totals and geographic variation in average basket size.
- **RFM Analysis** – automated RFM scoring with segment summaries and spatial lift analysis.
- **Promotion Impact** – promotion vs. non-promotion comparison, top-performing promo codes, and store profitability snapshot.
- **Retention Cohorts** – cohort matrix and retention trend charts for the first six months after acquisition.
- **Data Quality & Notes** – dynamic report on missing/zero values plus a reminder of active filters.

## Data Pipeline
Reusable data-prep lives in `data_pipeline.py`:
- Validates raw CSV schema and converts comma decimal values to numerics.
- Generates derived metrics (line-level revenue, discounts, promotion flags, customer age, etc.).
- Supplies helper tables: KPI highlights, promotion summaries, profitability views, cohort matrices, and data-quality diagnostics.
- The `RetailDataPipeline` class is available if you prefer a class-based interface for notebooks or scripts.

## Getting Started
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run retail_app.py
```

Streamlit will print a local URL (default `http://localhost:8501`). Open it in your browser to interact with the dashboard.

## Analyst Workflow
1. **Explore** — Use the Streamlit filters to slice by year/quarter/month, category, store, or city. Export filtered tables for deeper modelling if needed.
2. **Diagnose** — Leverage the KPI callouts and narrative bullets at the top of the Sales tab; cross-check with promotion and RFM tabs to understand drivers.
3. **Validate** — Jump to the Data Quality tab for missing-value checks; cohort and promotion tabs will warn when the filtered slice is too thin to analyse.
4. **Extend** — Build new notebooks/scripts on top of `data_pipeline.py` to run what-if simulations, forecasting, or incremental modelling.

## Suggested Extensions
- Add forecasting or anomaly detection on top of the monthly revenue series.
- Introduce profitability/discount thresholds to flag margin risk by store or promotion.
- Load additional months to strengthen cohort stability and support year-over-year comparisons.
- Pair with a lightweight slide deck that screenshots each tab and summarises key takeaways for stakeholders.

## Repository Structure
- `data_pipeline.py` – data loading, cleaning, enrichment, and analytic helper routines.
- `retail_app.py` – Streamlit application leveraging the pipeline and presenting business storytelling.
- `requirements.txt` – Python dependencies for the dashboard environment.
- `README.md` – project overview and usage guide (this file).
- CSV files – local offline datasets required by the app.

## Support
If you encounter schema changes in future datasets, update `REQUIRED_SALES_COLUMNS` or the relevant cleaners in `data_pipeline.py`, then rerun the app to ensure the validation checks pass.
