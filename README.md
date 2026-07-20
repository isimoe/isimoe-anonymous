# ISI-MoE: Importance-Aware Sparse Interaction Mixture-of-Experts

## Overview

Multimodal fusion integrates heterogeneous modalities to improve prediction and reasoning beyond unimodal representations. However, multimodal collaboration is inherently asymmetric: different modalities contribute unequally across samples and require different interaction directions and strengths. Existing fusion methods often rely on dense or weakly controlled cross-modal interactions, which can introduce redundant information exchange and obscure modality-specific and synergistic information roles. To address this issue, we propose Importance-Aware Sparse Interaction Mixture-of-Experts (ISI-MoE), an interpretable framework for asymmetric multimodal collaboration. ISI-MoE estimates sample-level modality importance using the Mutual Information Rate (MIR) and introduces AMSS to dynamically calibrate modality-strength contributions across different training stages. The calibrated importance signal guides sparse incremental cross-attention for controllable token-level cross-modal exchange. Grounded in Partial Information Decomposition (PID), ISI-MoE further introduces S2IM to constrain expert discrepancies and enforce bidirectional alignment, encouraging experts to specialize in modality-specific and cross-modal synergistic information. Extensive experiments on four real-world multimodal datasets show that ISI-MoE achieves the best performance among compared methods while maintaining computational efficiency.

![ISI-MoE architecture](assets/ISI-MoE.png)

## Environment setup

```shell
conda create -n isimoe python=3.10 -y
conda activate isimoe
pip install -r requirements.txt
```

## Data Directory

Create a data directory under `./data`.

```shell
mkdir -p data
```

## Reproduce Experiment Results

### ADNI

For ADNI data access and preprocessing, follow the [Flex-MoE ADNI guide](https://github.com/UNITES-Lab/Flex-MoE/blob/main/data/adni/README.md).

### MM-IMDB, MOSI, and Enrico

Download MM-IMDB, MOSI, and Enrico using the dataset links provided by [MultiBench](https://arxiv.org/abs/2107.07502). MOSI regression uses the same MOSI data.

## Train Models

### Train ISI-MoE models

- Supported fusion methods: `<fusion>` in `transformer`, `interpretcc`, `moepp`, `switchgate`.
- Supported datasets: `<dataset>` in `adni`, `mmimdb`, `mosi`, `mosi_regression`, `enrico`.

```shell
source scripts/train_scripts/isimoe/<fusion>/run_<dataset>.sh
```

## License

The code is released under the [MIT License](LICENSE).
