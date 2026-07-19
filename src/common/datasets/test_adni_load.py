import torch
import numpy as np
from adni import load_and_preprocess_data_adni


# Provide the minimal arguments required by the ADNI loader.
class Args:
    def __init__(self):
        self.modality = "I"  # Test the image modality only.
        self.initial_filling = "mean"
        self.patch = False  # Exercise the MLP path.
        self.hidden_dim = 128
        self.num_layers_enc = 2
        self.device = torch.device("cpu")
        self.num_patches = 16  # Used when patch mode is enabled.


def test_loading():
    args = Args()
    print("开始加载 ADNI 数据集...")

    try:
        (
            data_dict,
            encoder_dict,
            labels,
            train_idxs,
            valid_idxs,
            test_idxs,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            mc_idx_dict,
            mc_num_to_mc,
        ) = load_and_preprocess_data_adni(args)

        print("\n--- 加载成功！数据概览 ---")

        # Validate image feature dimensions.
        if "image" in data_dict:
            img_data = data_dict["image"]
            print(f"图像特征矩阵形状: {img_data.shape} (总人数 x 特征数)")
            print(f"每个样本的特征维度: {input_dims['image']}")

            # Confirm that at least one row contains observed data.
            real_data_count = np.sum(img_data[:, 0] != -2)
            print(f"拥有图像模态数据的样本数: {real_data_count}")

            # Inspect the expected min-max scaled range.
            valid_vals = img_data[img_data[:, 0] != -2]
            print(f"特征数值范围: [{valid_vals.min():.2f}, {valid_vals.max():.2f}]")

        # Inspect dataset split sizes.
        print(f"\n训练集样本数: {len(train_idxs)}")
        print(f"验证集样本数: {len(valid_idxs)}")
        print(f"测试集样本数: {len(test_idxs)}")

        # Inspect the configured encoder.
        print(f"\n图像编码器结构:\n{encoder_dict['image']}")

        # Validate labels.
        print(f"分类类别数: {n_labels}")
        print(f"标签前 10 个值: {labels[:10]}")

    except Exception as e:
        print(f"\n加载失败！错误信息: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_loading()
