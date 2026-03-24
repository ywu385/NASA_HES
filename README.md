# GLOBE Water Transparency — Bayesian Anomaly Detection Pipeline

Data quality framework for NASA's GLOBE citizen science dataset. Validates sparse, uncurated water transparency measurements using Bayesian hierarchical models and random forest, across 160,000+ observations from 4,500 sites spanning three decades.

**Collaborators:** Harvard University Capstone — Team 5 (NASA Team Blue)  
**Sponsor:** NASA GLOBE Program

---

## Quick orientation

| What | Where |
|------|-------|
| Data cleaning & constraint rules | `data_processing/` |
| Bayesian hierarchical models (PyMC3) | `models/bayesian/` |
| Random forest model | `models/random_forest/` |
| Continuous data curation framework | `curation/` → [framework repo](https://github.com/ywu385) |
| Site clustering & spatial analysis | `clustering/` |
| Full methodology & results | [`capstone_report.pdf`](./capstone_report.pdf) |

---

## What this does

Three-component validation framework:

1. **Physical constraints** — rules-based cleaning using hydrology domain knowledge (censored data handling, saturation flags, geo-proximity checks)
2. **Statistical modeling** — Bayesian hierarchical models quantify uncertainty across sites/seasons/water body types; random forest for classification
3. **Continuous curation** — autonomous framework validates new observations against current model, triggers retraining when anomalies detected

---

## Stack

`Python` · `PyMC3` · `scikit-learn` · `pandas` · `ArviZ`

**Key numbers:** 159,146 observations · 470 validated sites · 3 hierarchical model architectures · R-hat = 1.00 across all parameters
