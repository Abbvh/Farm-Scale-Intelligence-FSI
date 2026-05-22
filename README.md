# Farm Scale Intelligence (FSI)

> The AI intelligence layer for biological farming systems.

## Vision
FSI is a research-driven AI company building unified intelligence 
for animals, crops, and farm environments modeled on DeepMind's 
dual research + product philosophy.

---

## FSI v1.0 — Mastitis Early Detection

Current focus: Early detection of bovine mastitis using real-time 
continuous sensor signals with explainable AI (SHAP).

### Key scientific finding
> Mastitis can be predicted with 99.74% balanced accuracy from 
> continuous real-time farm signals alone without requiring 
> laboratory SCC testing.

### Pipeline completed
- Data loading and exploratory data analysis (Mendeley dataset)
- Dataset limitation analysis clinical signs identified as leakage
- Synthetic biologically-validated dataset generation (n=10,000)
- Feature engineering using animal science domain knowledge
- Random Forest baseline model
- XGBoost real-time early warning model
- 5-fold cross validation
- SHAP explainability, global and individual cow level

### Model performance — XGBoost (5-fold cross validation)

| Metric | Score |
|--------|-------|
| Balanced Accuracy | 0.9974 ± 0.0016 |
| AUC-ROC | 1.0000 ± 0.0000 |
| F1 Score | 0.9948 ± 0.0040 |
| Mastitis Recall | 0.99 |
| Mastitis Precision | 0.99 |

### Dataset
- 10,000 synthetic records (8,000 healthy / 2,000 mastitis)
- Generated from published dairy research parameters
- 80/20 train/test split, stratified

### Features used (real-time signals only)
| Feature | Source |
|---------|--------|
| milk_yield_kg | Automatic milking system |
| electrical_conductivity | Milking machine sensor |
| activity_steps | Pedometer / collar |
| rumination_time | Smart collar |
| body_temp | Temperature sensor |
| milk_fat_pct | Milking machine |
| milk_protein_pct | Milking machine |
| parity | Farm records |
| days_in_milk | Farm records |
| breed | Farm records |
| stress_score | Engineered feature |
| early_lactation | Engineered feature |
| yield_deviation | Engineered feature |

### Top predictors (SHAP analysis)
1. Electrical conductivity
2. Stress score (composite activity + rumination + temperature)
3. Body temperature
4. Rumination time
5. Milk yield

### Features excluded
- SCC (somatic cell count) — requires lab testing, not real-time
- This exclusion is FSI's core scientific contribution

---

## FSI Version Roadmap

| Version | Module | Status |
|---------|--------|--------|
| v1.0 | Animal health monitoring | ✅ Complete |
| v1.5 | Herd management system | 🔄 Planning |
| v2.0 | Genomics integration | 📋 Roadmap |
| v2.5 | Hardware layer | 📋 Roadmap |
| v3.0 | Crop intelligence | 📋 Roadmap |
| v4.0 | Systems intelligence | 📋 Roadmap |

---

## Tech stack
Python · scikit-learn · XGBoost · SHAP · pandas · matplotlib · joblib

## Repository structure
FSI/
├── notebooks/     — Jupyter notebooks
├── outputs/       — Plots, model files, performance JSON
├── data/          — Datasets
└── README.md
## Founded by
Murtala Kabir · 2026 
