# How To Run the Code
## Download dataset
1. [CICAndMal2017](https://www.unb.ca/cic/datasets/andmal2017.html)
```
mkdir -p /scratch/Malware/CICAndMal
wget -c -i data/CICAndMal2017/cic_url.txt -P /scratch/Malware/CICAndMal
```
2. [IoT23](https://www.stratosphereips.org/datasets-iot23)
```
mkdir -p /scratch/Malware/iot23
wget -c -i data/IoT23/iot23_url.txt \
     -P /scratch/Malware/iot23/mal \
     -x -nH --cut-dirs=3
```

## Data Preprocess
CICAndAMl2017 : data/CICAndMal2017/CIC_preprocess.py (saves to /scratch/Malware/CICAndMal/processed_data/)
IoT23 : data/IoT23/store_by_capture.py -> capture_preprocess.py (saves to /scratch/Malware/iot23/data/)

## Usage
- training with default arguments(kmeans exemplar selection)
```
python main_all.py 
```

## Refactored layout (automated)
After reorganization the repository follows this layout:

- `data/` : original dataset-related folders.
- `data/resources/` : small data artefacts (feature encoders, label classes) moved here.
- `results/` : JSON iteration / experiment summaries (`iteration_results_*.json`, `xgb_iteration_results_*.json`).
- `models/` : trained model files and exported model JSONs (`model-iteration*.pt`, `xgboost_model.json`).
- `plots/` : plotting scripts and generated figures. Plot scripts now read from `results/` and save output into `plots/`.

Example: run the comparison plots from the `plots/` folder:

```bash
python plots/plot_results.py
python plots/viz.py
```