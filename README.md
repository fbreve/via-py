# via-py — Visually Impaired Aid with CNNs, Transfer Learning, and Ensembles

Research code and experimental results associated with two publications on image-based obstacle recognition for visually impaired assistance:

1. **IJCNN 2020:** Breve, Fabricio Aparecido, and Carlos Norberto Fischer. *Visually Impaired Aid using Convolutional Neural Networks, Transfer Learning, and Particle Competition and Cooperation.* 2020 International Joint Conference on Neural Networks (IJCNN). [Published article (DOI)](https://doi.org/10.1109/IJCNN48605.2020.9207606) · [arXiv preprint](https://arxiv.org/abs/2005.04473) · [Author-hosted manuscript (PDF)](https://www.fabriciobreve.com/artigos/2020/wcci2020.pdf)
2. **ICCSA 2023:** Breve, Fabricio Aparecido. *Convolutional Neural Networks and Ensembles for Visually Impaired Aid.* In *Computational Science and Its Applications — ICCSA 2023*, Lecture Notes in Computer Science, vol. 13956, pp. 520–534. [Published chapter (DOI)](https://doi.org/10.1007/978-3-031-36805-9_34) · [Author-hosted manuscript (PDF)](https://www.fabriciobreve.com/artigos/2023/iccsa2023.pdf)

## Overview

The research investigates convolutional neural networks (CNNs), transfer learning, and ensemble approaches for recognizing potential obstacles in images. The 2020 study also uses the graph-based **Particle Competition and Cooperation (PCC)** method for semi-supervised learning.

The code reflects different stages of the research and includes experimental scripts, configuration files, saved predictions, results, and MATLAB feature/PCC data. It is a **research archive**, not a packaged application or a tested end-user assistive system.

## Dataset

The image dataset is published separately at **[fbreve/via-dataset](https://github.com/fbreve/via-dataset)**.

The newer transfer-learning scripts expect this folder layout by default:

```text
project-directory/
├── via-py/
│   └── visually_impaired_aid_tl.py
└── via-dataset/
    └── images/
```

That corresponds to the path `../via-dataset/images/` configured in `visually_impaired_aid_tl_config.py` and `visually_impaired_aid_tl_ensemble_config.py`. Update `DATASET_PATH` if your folders are arranged differently.

The dataset-loading function in these scripts assigns the class `clear` to filenames whose prefix before the first dot is `clear`, and `non-clear` to other filenames. Preserve the dataset's original filenames and check its separate documentation and usage rights before redistributing it.

## Main files and results

| Path | Purpose |
| --- | --- |
| `visually_impaired_aid_tl.py` | CNN transfer-learning experiments across model architectures |
| `visually_impaired_aid_tl_config.py` | Architecture ranges, training parameters, and machine-specific configurations |
| `visually_impaired_aid_tl_ensemble.py` | Ensemble-oriented CNN experiments |
| `visually_impaired_aid_tl_ensemble_config.py` | Ensemble experiment configuration |
| `visually_impaired_aid_tl_ensemble_tables.py` | Aggregation of ensemble results |
| `visually-impaired-aid-*.py`, `old-visually-impaired-aid-tl.py` | Earlier and alternative experimental implementations |
| `via-pcc/` | MATLAB scripts, extracted features, and PCC experimental outputs associated with the earlier study |
| `pred/` and `pred_ensemble/` | Saved per-experiment prediction CSV files |
| `results/` | CNN results, logs, spreadsheets, and experiment summaries |
| `results (IJCNN2020)/` | Archived results associated with the earlier study |
| `model_plot.png` | Model diagram |

The MATLAB experiments in `via-pcc/` use the separate [Particle Competition and Cooperation implementation](https://github.com/fbreve/Particle-Competition-and-Cooperation). The `.mat` files and recorded predictions are retained as scientific artifacts, not as installation dependencies for every Python script.

## Environment and reproducibility

The Python scripts use research-era **TensorFlow/Keras**, NumPy, pandas, scikit-learn, Pillow, and, in the principal training scripts, **TensorFlow Addons** (`tensorflow_addons`). Some older scripts have different requirements. No complete, historically verified dependency lockfile is provided.

**Important compatibility note:** TensorFlow Addons is discontinued and its [repository is archived](https://github.com/tensorflow/addons). The principal scripts import it even when `MULTI_OPTIMIZER` is disabled. Installing today's newest TensorFlow/Keras alongside TensorFlow Addons is not a verified way to reproduce these experiments. Prefer an isolated environment with mutually compatible versions chosen for the relevant historical code, and record them with your results.

The original runs used workstation-specific profiles named `DONALD`, `PRECISION`, and `SNOOPY`. The current configuration modules preserve their parameter selections, and also provide a **portable fallback** for other hostnames:

- The main transfer-learning fallback selects only the first listed CNN architecture by default and disables fine-tuning and multi-optimizer training.
- The ensemble fallback selects MobileNet and a locally identified configuration.
- These defaults **are not the experimental settings of either publication**. Inspect and edit the configuration before running an experiment.

The original profiles for `DONALD` and `PRECISION` lacked `MODEL_TYPE_END`, even though the main training script requires it. The current configuration supplies the list length as their default end index; `SNOOPY` retains its explicit end index.

Both primary scripts include repeated cross-validation and can require substantial GPU time and memory. A single selected model can still entail many training runs. The experiments were **not rerun** as part of this documentation update.

### Running the scripts

From inside `via-py/`, after preparing an appropriate historical Python environment and obtaining the dataset in the expected sibling directory, review the selected `*_config.py` file and then run the experiment of interest:

```bash
python visually_impaired_aid_tl.py
```

For the ensemble experiments:

```bash
python visually_impaired_aid_tl_ensemble.py
```

These commands invoke scientific experiment scripts, not a lightweight demonstration. Do not assume that the latest Python/TensorFlow versions are compatible. Logs, predictions, and experiment settings should be archived together for reproducibility.

## Citations

Please cite the relevant publication(s) if you use the corresponding implementation or results.

### IJCNN 2020

```bibtex
@inproceedings{breve2020visually,
  author    = {Breve, Fabricio Aparecido and Fischer, Carlos Norberto},
  title     = {Visually Impaired Aid Using Convolutional Neural Networks, Transfer Learning, and Particle Competition and Cooperation},
  booktitle = {2020 International Joint Conference on Neural Networks (IJCNN)},
  year      = {2020},
  doi       = {10.1109/IJCNN48605.2020.9207606}
}
```

### ICCSA 2023

```bibtex
@incollection{breve2023ensembles,
  author    = {Breve, Fabricio Aparecido},
  title     = {Convolutional Neural Networks and Ensembles for Visually Impaired Aid},
  booktitle = {Computational Science and Its Applications -- ICCSA 2023},
  series    = {Lecture Notes in Computer Science},
  volume    = {13956},
  pages     = {520--534},
  publisher = {Springer},
  year      = {2023},
  doi       = {10.1007/978-3-031-36805-9_34}
}
```

## License and third-party materials

The original source code in this repository is distributed under the [Apache License, Version 2.0](LICENSE).

**The dataset is maintained in a separate repository** and may be governed by distinct terms. The code license does not automatically grant rights to redistribute third-party images, datasets, pretrained weights, or other external materials represented in the experimental artifacts.
