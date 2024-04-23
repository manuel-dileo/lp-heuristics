# lp-heuristics
This repository contains the code for reproducing the results in the paper "On the importance of link prediction heuristics for temporal graph benchmark".

## Reproducibility
To run a certain heuristic on a dataset, you can use the following script:
```
dataset="tgbl-wiki"
heuristic="PA"

python run.py --dataset_name "$dataset" --heuristic "$heuristic"
```
The above script runs and evaluates Preferential Attachment on the `tgbl-wiki` dataset.
