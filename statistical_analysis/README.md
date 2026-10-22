# OnSight Pathology: statistical analysis code

This folder contains the code for the statistical analyses and figure panels of "OnSight Pathology as a
Real Time Platform Agnostic Computational Pathology Companion for Histopathology". 

## How to run

```bash
pip install -r requirements.txt
jupyter notebook
```

Start Jupyter in this folder, open a notebook and choose Run All. All paths are relative to this folder.

| Notebook | Figures and tables | Runtime |
|---|---|---|
| `main_figures.ipynb` | Figs 2b, 3c–h, 4b, 4c, 5a–k, 6c–f, 7e–g | about 1 min |
| `supplementary_figures.ipynb` | Supplementary Figs 1, 2, 7, 13–21, 24, 25, 28–30, 32, 33 and Supplementary Tables 1–3 | about 1 min |

Each notebook prints the statistics reported in the figure legends next to the panel that uses them and saves every
panel to `outputs/` (PNG at 300 dpi and PDF). 


## Contents

```
main_figures.ipynb
supplementary_figures.ipynb
source_data/
  Main_Figure/fig2 ... fig7               Source Data of the main figures
  Supplementary_Figure/fig1 ... fig33     Source Data of the supplementary figures (folder number = figure number)
  Supplementary_Table/                    Source Data of Supplementary Table 2
survival/
  survival_models.Rmd                     XGBoost survival models with leave-one-out cross-validation (R)
  patch_dilution_simulation.py            simulation behind Fig 5d
figure_assets/                            image tiles of Supplementary Fig 16a (not data; used only for the layout)
requirements.txt
```



## Survival models (R)

`survival/survival_models.Rmd` fits the XGBoost survival models of Fig 5 and the single-feature model of
Supplementary Fig 17. It needs R with survival, survminer, xgboost, dplyr and ggplot2, and reads the
patient-level features in `source_data/Main_Figure/fig5`. 

`survival/patch_dilution_simulation.py` contains the simulation behind Fig 5d. 

## Software

Tested with Python 3.11.9 and the package versions in `requirements.txt`.
