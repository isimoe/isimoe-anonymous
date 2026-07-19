# ISI-MoE Reproducibility Guide

This document describes the anonymous review artifact. It does not redistribute raw third-party datasets or credentials.

## 1. System assumptions

- Python 3.10
- Linux is recommended for the supplied shell launchers
- NVIDIA GPU and a compatible CUDA toolchain are recommended for full experiments
- CPU-only execution may require disabling sparse fusion and reducing the model/data configuration

Install the Python environment with `pip install -r requirements.txt`. For `--fusion_sparse True`, install FasterMoE separately against the exact PyTorch/CUDA combination used by the machine.

## 2. Data root

Run commands from the repository root. By default, loaders expect `./data`. Dataset access remains subject to the original provider's terms.

### ADNI

```text
data/adni/
├── label.csv
├── PTID_splits.json
├── image/
│   ├── UCSFFSX7_29Jan2026.csv
│   ├── ADNI_G.npy                         # only for non-preprocessed imaging
│   ├── ADNI_subj.txt                      # only for non-preprocessed imaging
│   └── BLSA_SPGR+MPRAGE_averagetemplate_muse_seg_DS222.nii.gz
├── genomic/genomic_merged.h5ad
├── clinical/
│   ├── clinical_merged.csv
│   └── clinical_merged_mean.csv
└── biospecimen/
    ├── biospecimen_merged.csv
    └── biospecimen_merged_mean.csv
```

### CMU-MOSI

```text
data/cmu-mosi/mosi_data.pkl
```

The pickle must contain `train`, `valid` and `test` dictionaries with `vision`, `audio`, `text`, `labels` and `id` fields.

### MM-IMDb

```text
data/mm-imdb/multimodal_imdb.hdf5
```

The loader expects the keys `imdb_ids`, `features`, `vgg_features` and `genres`.

### Enrico

```text
data/enrico/
├── design_topics.csv
├── screenshots/
├── wireframes/
└── hierarchies/
```

## 3. Reference run

Start with the non-sparse Transformer configuration to validate the environment and data layout:

```shell
python src/isimoe/train_transformer.py \
  --data mosi \
  --modality TVA \
  --train_epochs 2 \
  --batch_size 32 \
  --hidden_dim 64 \
  --num_patches 4 \
  --num_heads 4 \
  --fusion_sparse False \
  --amss_enabled True \
  --use_interaction True \
  --track_mir_curve True \
  --seed 0 \
  --n_runs 1 \
  --save False
```

Full hyperparameter settings are encoded in `scripts/train_scripts/isimoe/`. Run each script from the repository root and record the commit/archive checksum with the resulting logs.

## 4. Outputs

```text
logs/isimoe/                  textual run logs
saves/isimoe/                 model checkpoints when --save True
outputs/isimoe/               predictions and expert/routing visualizations
outputs/mir_curves/isimoe/    MIR CSV files and trajectory figures
```

Report mean and standard deviation across the same seed set used in the paper. Do not compare a single-seed rerun with a multi-seed paper result.

## 5. Review-time verification

Before uploading or making the repository public:

1. Search all tracked text files for author names, email addresses, institution names, usernames, personal paths and repository URLs.
2. Inspect Git commit author names/emails and squash or recreate history in the anonymous account if necessary.
3. Remove `.idea`, editor metadata, checkpoints, logs, caches and raw data.
4. Inspect PDF, image and archive metadata.
5. Open the public Pages URL in a signed-out/private browser session.
6. Confirm that code and documentation are available at submission time and do not promise a future release as the only evidence.
