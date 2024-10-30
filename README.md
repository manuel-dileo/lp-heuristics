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

## Cite
If you use the code of this repository for your project or you find the work interesting, please cite the following work:  

```bibtex
@article{Dileo2024LinkPH,
  title={Link prediction heuristics for temporal graph benchmark},
  author={Manuel Dileo and Matteo Zignani},
  journal={ESANN 2024 proceedings},
  year={2024},
  url={https://www.esann.org/proceedings/2024#526},
  doi={https://doi.org/10.14428/esann/2024.ES2024-141}
}
```
