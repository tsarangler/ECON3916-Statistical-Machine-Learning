# 📉 Lab 2: Deflating History with FRED

**The Illusion of Growth & The Composition Effect**

> *Analyzing 50 years of wage stagnation and exposing the statistical mirage of the 2020 "wage boom."*

---

## 🎯 Objective

Build an automated Python pipeline to ingest live economic data from the **Federal Reserve Economic Data (FRED) API** in order to:

- Calculate inflation-adjusted ("real") wages and visualize long-term stagnation
- Identify and correct for the **Composition Effect**—a statistical bias that distorted wage data during the COVID-19 pandemic

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `Python 3.x` | Core language |
| `fredapi` | FRED API client |
| `pandas` | Data manipulation & time series alignment |
| `matplotlib` | Visualization |

---

## 📊 Methodology

### 1. Data Ingestion
| Series ID | Description |
|-----------|-------------|
| `AHETPI` | Average Hourly Earnings of All Employees (Total Private) |
| `CPIAUCSL` | Consumer Price Index for All Urban Consumers |
| `ECIWAG` | Employment Cost Index (Wages & Salaries) |

### 2. Inflation Adjustment
Deflated nominal wages (`AHETPI`) against CPI to compute **Real Wages**, revealing true purchasing power over time.

### 3. Anomaly Detection
Identified a sharp upward spike in Q2 2020—average wages surged during an economic contraction. This counterintuitive result triggered further investigation.

### 4. Composition Effect Correction
Introduced the **Employment Cost Index (ECI)** as a control. Unlike raw averages, the ECI tracks wage changes for a *fixed basket of occupations*, neutralizing shifts in workforce composition.

---

## 🔍 Key Findings

### The Money Illusion
![Status](https://img.shields.io/badge/Finding-Confirmed-blue)

Real wages have remained **effectively flat for over 50 years**. Despite nominal earnings appearing to grow, workers' purchasing power has stagnated—a classic demonstration of the "Money Illusion."

---

### The Pandemic Paradox
![Status](https://img.shields.io/badge/Finding-Artifact_Detected-orange)

The 2020 "wage boom" was **not** a sign of increased labor demand.

**What happened:**
- Pandemic layoffs disproportionately hit low-wage sectors (hospitality, retail, services)
- The remaining workforce skewed toward higher earners
- The *average* rose because lower earners exited the sample—not because anyone received a raise

**The proof:**  
The ECI, which holds occupation mix constant, shows **no corresponding spike**. The apparent wage surge occurred precisely when workers had the *least* leverage.

---

## 💡 Conclusion

Headline economic statistics can mislead without proper context. By leveraging the FRED API and applying compositional controls, this project distinguishes genuine economic phenomena from **artifacts of measurement**.

---

<p align="center">
  <i>Data sourced from the Federal Reserve Bank of St. Louis</i>
</p>
