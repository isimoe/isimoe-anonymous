# ISI-MoE: Adaptive Interaction-Aware Mixture-of-Experts for Multimodal Learning

![Anonymous Review](https://img.shields.io/badge/review-anonymous-137f78)
![Reproducibility](https://img.shields.io/badge/artifact-reproducibility--ready-111d2e)
![GitHub Pages](https://img.shields.io/badge/site-GitHub%20Pages-dff472)
![License: MIT](https://img.shields.io/badge/license-MIT-6b7280)

Anonymous implementation and review artifact for **ISI-MoE**, an interaction-aware mixture-of-experts framework for heterogeneous and incomplete multimodal inputs.

> **Double-blind review notice.** Author names, affiliations, personal accounts, contact details and identity-bearing links are intentionally omitted. Please do not add a citation or acknowledgements section until the review period has ended.

- Project page: [https://isimoe.github.io/isimoe-anonymous/](https://isimoe.github.io/isimoe-anonymous/)
- Reproduction guide: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
- Anonymous publishing guide: [`English`](PUBLISHING.md) · [`中文`](PUBLISHING_CN.md)
- Reference scripts: [`scripts/train_scripts/isimoe/`](scripts/train_scripts/isimoe/)

## Overview

Multimodal fusion must distinguish information that is unique to one modality from evidence that emerges through cross-modal interaction. ISI-MoE addresses this with four components:

1. **Modality encoders** map heterogeneous inputs into a shared representation space.
2. **Adaptive sparse delta cross-attention** selects informative context tokens and injects cross-modal residuals.
3. **Modality-specialized and shared experts** separate modality-specific evidence from common multimodal evidence.
4. **Sample-wise routing** combines expert outputs according to the evidence available for each example.

The training implementation supports task, interaction, routing, auxiliary and orthogonality objectives. Each term is controlled by the supplied reference recipes.

![ISI-MoE architecture](assets/ISI-MoE.png)

## Environment setup

```shell
conda create -n isimoe python=3.10 -y
conda activate isimoe
pip install -r requirements.txt
```

The core Transformer path can run without sparse fusion. FasterMoE is required only when `--fusion_sparse True`; follow its CUDA-specific installation notes for that configuration.

## Data directory

Create a `data` directory under the repository root. Raw datasets are not redistributed because their original licenses and, for clinical datasets, data-use agreements continue to apply.

```text
data/
├── adni/
│   ├── label.csv
│   ├── PTID_splits.json
│   ├── image/
│   ├── genomic/
│   ├── clinical/
│   └── biospecimen/
├── cmu-mosi/
│   └── mosi_data.pkl
├── mm-imdb/
│   └── multimodal_imdb.hdf5
└── enrico/
    ├── design_topics.csv
    ├── screenshots/
    └── wireframes/
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the exact file-level layout and access constraints.

## Train ISI-MoE

Supported fusion backbones include `transformer`, `interpretcc`, `moepp` and `switchgate`. Supported datasets include `adni`, `mmimdb`, `mosi`, `mosi_regression` and `enrico`.

```shell
# Example: Transformer backbone on CMU-MOSI
python src/isimoe/train_transformer.py \
  --data mosi \
  --modality TVA \
  --fusion_sparse False \
  --amss_enabled True \
  --seed 0

# Reference sweep
bash scripts/train_scripts/isimoe/transformer/run_amss_mosi.sh
```

Logs, checkpoints, predictions, routing weights and MIR trajectories are written under `logs/`, `saves/` and `outputs/`.

## Baselines and ablations

Baseline training entry points are under [`src/baseline/`](src/baseline/). Ablation implementations and launchers are under [`src/ablation/`](src/ablation/) and `scripts/train_scripts/`.

The release includes variants for:

- no interaction loss;
- a single interaction expert;
- mean, zero-vector and random perturbation controls;
- latent contrastive interaction;
- synergy/redundancy-only objectives;
- simplified expert aggregation.

## Anonymous artifact checklist

- [x] No author names, affiliations, email addresses or personal repository links in the review landing page.
- [x] No analytics, trackers or third-party fonts on the GitHub Pages site.
- [x] Environment and dataset layouts are documented.
- [x] Training commands, seeds and output locations are documented.
- [ ] Re-run the identity scan immediately before submission.
- [ ] Verify the final PDF and supplementary archive metadata separately.
- [ ] Publish from an anonymous GitHub account or organization created only for review.

## License and citation

The code is released under the [MIT License](LICENSE). Citation authorship and BibTeX are intentionally omitted from this anonymous draft; add them only after confirming that doing so is compatible with the double-blind review policy.
