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

The core Transformer path can run without sparse fusion. FasterMoE is required only when `--fusion_sparse True`; follow its CUDA-specific installation notes for that configuration.

## Data Directory

Create a data directory under `./data`.

在项目根目录下创建 `./data` 数据目录。

```shell
mkdir -p data
```

## Reproduce Experiment Results

### ADNI

To access the ADNI dataset, please first visit the [ADNI website](https://adni.loni.usc.edu/) and [apply for data access](https://ida.loni.usc.edu/collaboration/access/appApply.jsp?project=ADNI).

要访问 ADNI 数据集，请先访问 [ADNI 网站](https://adni.loni.usc.edu/)，并在[此处申请数据访问权限](https://ida.loni.usc.edu/collaboration/access/appApply.jsp?project=ADNI)。

Once you obtain access, log in to IDA and download the necessary files for each modality.

获得访问权限后，请登录 IDA，并下载每种模态所需的文件。

**Steps / 步骤：**

`Search & Download` → `Study Collections` → `Study Files` → `Imaging`

`搜索与下载` → `研究集合` → `研究文件` → `成像`

Download **“UCSF - Cross-Sectional FreeSurfer (7.x) [ADNI1, GO, 2, 3, 4]”**.

下载 **“UCSF - 横断面 FreeSurfer (7.x) [ADNI1, GO, 2, 3, 4]”**。

Further details are available in the [Flex-MoE ADNI preprocessing guide](https://github.com/UNITES-Lab/Flex-MoE/blob/main/data/adni/README.md#1-2-image-mri-preprocessing).

更详细的信息请参阅 [Flex-MoE 的 ADNI 预处理说明](https://github.com/UNITES-Lab/Flex-MoE/blob/main/data/adni/README.md#1-2-image-mri-preprocessing)。

The ISI-MoE training entry point and ADNI data loader support loading preprocessed data. Use the argument `--preprocessed True` for this purpose.

ISI-MoE 的训练入口和 ADNI 数据加载器支持读取预处理数据，可使用参数 `--preprocessed True`。

### MM-IMDB, MOSI, and Enrico

Download MM-IMDB, MOSI, and Enrico using the dataset links provided by [MultiBench](https://arxiv.org/abs/2107.07502). MOSI regression uses the same MOSI data.

请通过 [MultiBench](https://arxiv.org/abs/2107.07502) 提供的数据集链接下载 MM-IMDB、MOSI 和 Enrico；MOSI 回归任务使用相同的 MOSI 数据。

## Train Models

### Train ISI-MoE models

- Supported fusion methods: `<fusion>` in `transformer`, `interpretcc`, `moepp`, `switchgate`.
- Supported datasets: `<dataset>` in `adni`, `mmimdb`, `mosi`, `mosi_regression`, `enrico`.

```shell
source scripts/train_scripts/isimoe/<fusion>/run_<dataset>.sh
```

## License

The code is released under the [MIT License](LICENSE).
