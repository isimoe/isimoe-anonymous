import torch
import numpy as np
from adni import load_and_preprocess_data_adni


# 1. 模拟 argparse 对象，设置必要的参数
class Args:
    def __init__(self):
        self.modality = "I"  # 只测试图像模态 (Image)
        self.initial_filling = "mean"
        self.patch = False  # 测试 MLP 模式
        self.hidden_dim = 128
        self.num_layers_enc = 2
        self.device = torch.device("cpu")
        self.num_patches = 16  # 如果 patch 为 True 时使用


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

        # 2. 检查图像特征维度
        if "image" in data_dict:
            img_data = data_dict["image"]
            print(f"图像特征矩阵形状: {img_data.shape} (总人数 x 特征数)")
            print(f"每个样本的特征维度: {input_dims['image']}")

            # 检查是否有实际数据被填入（非 -2 的行）
            real_data_count = np.sum(img_data[:, 0] != -2)
            print(f"拥有图像模态数据的样本数: {real_data_count}")

            # 检查数据范围（应该是 MinMaxScaler 处理后的 -1 到 1）
            valid_vals = img_data[img_data[:, 0] != -2]
            print(f"特征数值范围: [{valid_vals.min():.2f}, {valid_vals.max():.2f}]")

        # 3. 检查划分情况
        print(f"\n训练集样本数: {len(train_idxs)}")
        print(f"验证集样本数: {len(valid_idxs)}")
        print(f"测试集样本数: {len(test_idxs)}")

        # 4. 检查编码器
        print(f"\n图像编码器结构:\n{encoder_dict['image']}")

        # 5. 验证标签
        print(f"分类类别数: {n_labels}")
        print(f"标签前 10 个值: {labels[:10]}")

    except Exception as e:
        print(f"\n加载失败！错误信息: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_loading()