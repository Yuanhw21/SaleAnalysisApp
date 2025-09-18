# Alberta Retail Performance Analytics

## 1. Background and Objectives
This project builds an end-to-end business analytics workflow for the Alberta retail portfolio. It:
- Consolidates and anonymises transaction fact tables plus supporting dimensions so analysts can work from a reusable dataset (`src/data_pipeline.py`).
- Produces core insights on monthly KPIs, category mix, promotion efficiency, store profitability, customer value, and retention (`BA_DA_Report.ipynb`, `src/report_utils.py`).
- Surfaces results as interactive dashboards (`retail_app.py`) and a narrative BA/DA notebook for decision support.

## 2. Data Coverage and Sources
- **Sales window**: the report currently loads all available months between January and December 2023 (`BA_DA_Report.ipynb`).
- **Reference tables**: contacts, items, promotions, and stores are normalised and joined during ingestion (`src/data_pipeline.py`).
- **File organisation**: the pipeline scans annual folders under `data/`, selecting the best file per month and preferring CSV over XLSX when both exist (`src/data_pipeline.py`).
- **Open-source dataset**: each month is limited to 1,000 anonymised transactions with hashed identifiers; reference tables only retain rows referenced by the sample so the project remains lightweight for cloning and exploration.

## 3. Processing and Privacy Hardening
The pipeline performs multiple layers of cleanup before analytics:
- **Identifier normalisation**: trims whitespace, collapses placeholder values ("na", "null", etc.), and standardises primary keys across fact and dimension tables (`src/data_pipeline.py`).
- **Store obfuscation & geocoding**: builds an Alberta-specific lookup to rewrite store IDs, names, postal codes, and coordinates, while keeping online stores grouped under `ABWEB##` (`src/data_pipeline.py`, `src/anonymize_datasets.py`).
- **Identifier hashing**: transaction, customer, item, and promotion IDs are replaced with prefixed SHA-1 hashes to preserve structure but prevent reversal (`src/data_pipeline.py`, `src/anonymize_datasets.py`).
- **Numeric & date cleansing**: converts pricing/quantity fields, derives line-level revenue, original price, discount, and ensures `Date_transaction` is parsed correctly (`src/data_pipeline.py`).
- **Promotion flag**: derives a reliable `Promo_Flag` from the cleaned promotion identifiers for downstream comparisons (`src/data_pipeline.py`).
- **Customer attributes**: normalises boolean fields such as `Has_Web_Account` before analysis (`src/data_pipeline.py`).

> **Note**: `promotion_activity_timeline` currently triggers a Pandas 3.0 chained-assignment warning in `src/report_utils.py`. Replace inplace fills with explicit assignments (e.g. `frame['END_DATE'] = frame['END_DATE'].fillna(...)`) to silence the warning.

## 4. Data Quality Assessment
`load_and_prepare_data` exposes helper utilities to review schema, data types, missing values, zero counts, and cardinality (`src/data_pipeline.py`). Use this report to prioritise governance items with high business impact.

## 5. Metrics and Analytical Methods
Key reusable routines include:
- **Monthly KPI summary**: revenue, orders, units, unique customers, and average order value per month (`src/report_utils.py`).
- **Category contribution**: top categories and Pareto shares (`src/report_utils.py`, amplified in the notebook).
- **Promotion lifecycle**: monthly active-promotion timeline; missing `END_DATE` values default to "still active" (`src/report_utils.py`).
- **Promotion effectiveness**: transaction-level promo vs. non-promo comparisons, OLS lift model, and Welch t-test to validate incremental value (`BA_DA_Report.ipynb`, `src/report_utils.py`).
- **Store profitability**: revenue, volume, and discount structure by store/city (`src/data_pipeline.py`).
- **Network expansion**: historical store openings and trend visualisations (`src/report_utils.py`).
- **Customer engagement**: web-account adoption, cohort retention, and new RFM scoring blocks (`BA_DA_Report.ipynb`, `src/report_utils.py`).
- **One-page summaries**: overall KPIs, top cities, online share, and other headline stats (`src/data_pipeline.py`).

## 6. Insights Delivered
The HTML/Notebook report packages the following perspectives (values rendered when the notebook is executed):
- **Temporal coverage**: confirms the 2023 monthly window via the loader cells.
- **Promotion cadence & lift**: charts active promotions over time and quantifies order lift vs. discount depth.
- **Store & city structure**: contrasts performance and discount intensity across anonymised stores and geographies.
- **Online performance**: online stores remap to dedicated IDs, enabling clean splits between physical and digital trade.
- **Category & SKU Pareto**: identifies the 80/20 mix for planning, SKU rationalisation, and merchandising.
- **RFM segmentation**: classifies customers into Champions/Loyal/Potentials/At Risk with category preferences and recency buckets.

## 7. Recommendations
- **Promotion governance**: retain campaigns with statistically significant lift and manageable discount rates; sunset the rest (`BA_DA_Report.ipynb`, `src/report_utils.py`).
- **Store optimisation**: target stores with weak GMV but heavy discounting for operational review (`src/data_pipeline.py`).
- **Channel strategy**: monitor online share vs. promotion cadence to synchronise campaigns across BOPIS/online journeys (`src/report_utils.py`).
- **Retention programmes**: activate offers within the first 1–3 months after first purchase using cohort and RFM outputs (`BA_DA_Report.ipynb`).
- **Data governance**: feed DQA findings back to source owners, especially for high-value customer or product attributes (`src/data_pipeline.py`).

## 8. Technical Risks & Improvements
- **Pandas chained assignment**: refactor inplace fills as highlighted above to stay compliant with Pandas 3.0 behaviour (`src/report_utils.py`).
- **Store/city mapping**: ensure upstream data providers honour standard naming to reduce fallback mappings to default cities (`src/data_pipeline.py`).
- **Hashing strategy**: SHA-1 + prefixes meet confidentiality requirements; retain salt management procedures if cross-system reconciliation is needed (`src/anonymize_datasets.py`).

## 9. Conclusion
The repository now ships a reproducible analytics stack that ingests sensitive retail data, anonymises it, and surfaces time, category, promotion, store, and customer value insights. It is suitable for ongoing decision-making and future productisation (dashboards, experimentation platforms, CLV/price models).

## 10. References & Traceability
- Data ingestion, joins, and cleaning: `src/data_pipeline.py`.
- Anonymisation helpers and geography mapping: `src/anonymize_datasets.py`.
- KPI, promotion, and visualisation utilities: `src/report_utils.py`.
- Business analytics notebook: `BA_DA_Report.ipynb`.
- Streamlit app shell: `retail_app.py`.

## Getting Started
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run retail_app.py
```
Use Jupyter, VS Code, or similar to execute `BA_DA_Report.ipynb` if you prefer the storytelling notebook format.
