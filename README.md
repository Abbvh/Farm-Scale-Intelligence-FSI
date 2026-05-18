# Farm Scale Intelligence (FSI)

> The AI intelligence layer for biological farming systems.

## Vision
FSI is a research-driven AI company building unified intelligence 
for animals, crops, and farm environments.Modeled on DeepMind's 
dual research + product philosophy.

## FSI v1.0 — Mastitis Early Detection
Current focus: Early detection of bovine mastitis using sensor 
and production data with explainable AI (SHAP).

### Pipeline completed
- Data loading and exploratory data analysis
- Preprocessing and feature engineering  
- Random Forest baseline model
- XGBoost core model
- SHAP explainability — global and individual cow level

### Results
| Metric | Random Forest | XGBoost |
|--------|--------------|---------|
| Balanced Accuracy | 0.9932 | 0.9928 |
| AUC-ROC | 0.9968 | 0.9987 |
| Mastitis Recall | 0.99 | 0.99 |

### Key finding
Dataset limitations identified clinical signs encoded as 
features inflate performance. Next step: acquire longitudinal 
pre-clinical sensor dataset for honest early warning modeling.

## FSI Version Roadmap
- v1.0 — Animal health monitoring (current)
- v1.5 — Herd management system
- v2.0 — Genomics integration
- v2.5 — Hardware layer
- v3.0 — Crop intelligence
- v4.0 — Systems intelligence

## Tech stack
Python · scikit-learn · XGBoost · SHAP · pandas · matplotlib

## Founded by
Murtala Kabir· 2025
