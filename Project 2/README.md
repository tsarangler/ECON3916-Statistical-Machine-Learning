# WTI Crude Oil Price Predictor
### ECON 3916 — ML Prediction Project — Spring 2026

**Prediction question:** Can we predict next-day WTI crude oil spot price from macroeconomic indicators?

**Stakeholder:** Energy portfolio managers deciding whether to increase or reduce crude oil exposure.

**Live app:** https://yourapp.streamlit.app *(update after deployment)*

---

## Repository Structure

```
├── app.py
├── requirements.txt
├── README.md
├── Project 2/
│   └── 3916-final-project-starter.ipynb
└── data/
    └── README_data.md
```

---

## Data Acquisition

All data from FRED (fred.stlouisfed.org). Download as CSV into a `data/` folder:

| File | FRED Series | Description |
|------|-------------|-------------|
| DCOILWTICO.csv | DCOILWTICO | WTI Crude Oil Price (daily) |
| DFF.csv | DFF | Federal Funds Rate (daily) |
| DTWEXBGS.csv | DTWEXBGS | Broad USD Index (daily) |
| DGS10.csv | DGS10 | 10Y Treasury Yield (daily) |
| CPIAUCSL.csv | CPIAUCSL | CPI All Urban (monthly) |
| WCRSTUS1w.csv | EIA | US Crude Inventories (weekly) |

Access date: April 19, 2026

---

## Environment Setup

Requires Python 3.10+.

```bash
git clone https://github.com/tsarangler/ECON3916-Statistical-Machine-Learning.git
cd ECON3916-Statistical-Machine-Learning
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

---

## Running the Notebook

```bash
jupyter notebook "Project 2/3916-final-project-starter.ipynb"
```

Runs end-to-end sequentially (Parts 0-7). Update the os.chdir path in cell 2.1 to your local data/ folder. All random ops use random_state=42.

---

## Running the Streamlit App Locally

```bash
streamlit run app.py
```

Opens at http://localhost:8501. No data files needed — app trains on a synthetic dataset reproducing the FRED data's statistical properties (same seed and distributions).

---

## Deploying to Streamlit Community Cloud

1. Push app.py and requirements.txt to GitHub
2. Go to streamlit.io/cloud
3. Sign in with GitHub, click New app
4. Repo: tsarangler/ECON3916-Statistical-Machine-Learning | Branch: main | File: app.py
5. Click Deploy
6. Submit the permanent URL on Canvas

---

## Model Summary

| Model | CV R2 (5-fold) | Test RMSE |
|-------|----------------|-----------|
| Ridge Regression (baseline) | 0.4851 +/- 0.0230 | $9.66/bbl |
| Random Forest (200 trees) | 0.9746 +/- 0.0042 | $2.46/bbl |

Features: fed_funds, usd_index, t10y, cpi, crude_inv
Target: next-day WTI price via .shift(-1)
Split: 80/20, random_state=42

---

## Citation

Federal Reserve Bank of St. Louis. FRED Economic Data. https://fred.stlouisfed.org. Accessed April 19, 2026.
