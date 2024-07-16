# Link prediction heuristics
This repository contains the code for reproducing the results in the paper "Link prediction heuristics for temporal graph benchmark", to appear in ESANN 2024.

## Reproducibility
To run a certain heuristic on a dataset, you can use the following script:
```
dataset="tgbl-wiki"
heuristic="PA"

python run.py --dataset $dataset --heuristic $heuristic
```
The above script runs and evaluates Preferential Attachment on the `tgbl-wiki` dataset.
