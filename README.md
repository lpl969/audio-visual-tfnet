# Audio-Visual TFNet

用于音视频联合分类与三维定位的完整网络及训练实现。仓库仅包含网络、必要的数据加载与训练代码，不包含数据集、权重、历史日志、论文、消融或对比实验。

## 网络结构

1. 四通道音频及其四阶球谐投影（25 通道）分别做 STFT，幅度谱经过稳定门控融合。
2. 四通道融合幅度、四通道 SCP 特征及参考通道相位组成 9 通道输入。这里使用的是 STFT 幅度谱，部分历史变量名中的 `mel` 不表示 Mel 滤波。
3. DenseEncoder 提取音频特征；4 个时频模块串行处理时间与频率维度，再进行全局/局部门控及交叉注意力融合。
4. 时间 Conformer 使用 ReLU 路由、4 个低秩专家与因果深度卷积。频率 Conformer 使用普通 FFN。`top_k=2` 是辅助约束目标，并非硬 Top-2 路由；执行所有正分数专家。
5. DarkNet 风格图像编码器产生 14×14 个 token，经音频上下文条件化、FFN 与音视频交叉注意力融合。
6. 音频及图像的池化特征映射到双曲空间，距离转换为相似度，与切空间特征及融合特征拼接后，分别预测类别 logits 和三维坐标。

```python
import torch
from models import TFNet

model = TFNet(num_cls=5).eval()
with torch.no_grad():
    logits, position = model(
        torch.randn(1, 4, 9600),
        torch.randn(1, 25, 9600),  # 示例；训练时由音频计算球谐投影
        torch.rand(1, 3, 224, 224),
    )
# logits: (1, 5), position: (1, 3)
```

## 安装与训练

使用 Python 3.10 或更新版本。GPU 训练请先安装与本机 CUDA 匹配的 PyTorch。

依赖中限制 SciPy `<1.17`，因为当前 spaudiopy 球谐接口使用新版 SciPy 已移除的 `sph_harm`。

```bash
python -m pip install -r requirements.txt
python train.py --root ./data --gpu cuda:0 --batch_size 32 --train_epoch 300
```

无 CUDA 时自动使用 CPU。`--gpu cpu` 可显式选择 CPU。Windows 遇到数据加载进程问题时可使用 `--num_workers 0`。

默认训练设置：Adam，学习率 0.0001，weight decay 0，batch size 32，300 epochs，seed 42。未启用学习率调度、混合精度、梯度裁剪或预训练图像权重。

损失为 `CrossEntropy(logits, label) + 1.5 × L1(valid_position)`。位置无效的样本只参与分类训练；L1 在有效样本的三个坐标上取均值。验证使用独立的 `val` 列表，最佳权重按有效样本的平均欧氏定位误差 APE 选择。没有有效验证位置时明确报错。坐标单位沿用输入标签。

路由器保留辅助损失实现，但原训练入口未收集或加入这些项，本发布版同样不启用。不能将其描述为已参与默认训练的负载均衡正则；其中基于硬布尔路由统计的负载项也不提供路由梯度。

`output/best.pth` 保存模型、优化器、epoch、最佳 APE 和数值训练配置；不保存本地数据及输出路径。训练日志和 `train_config.json` 也不记录这些路径。

## 数据接口

自行准备以下结构；仓库不提供任何真实样本或划分列表：

```text
data/
  annotation/V3/train.txt
  annotation/V3/val.txt
  audio_npy/0/sample.npy
  visual_data/image_npy/0/sample.npy
  label/0/sample.npy
  gt/0/sample.npy
```

每行一个相对路径，例如 `0/sample.npy`，首级目录为整数类别编号。文件对应关系：

| 文件 | 内容 |
| --- | --- |
| audio_npy | 浮点音频数组 `(4, T)` 或 `(T, 4)`；裁剪或补零到 9600 点 |
| visual_data/image_npy | RGB 图像 HWC 或 CHW；单通道会复制；0–255 输入除以 255，缩放为 224×224 |
| label | 首个数值为类别 ID，范围 `[0, num_cls)` |
| gt | 前三个数值为三维坐标；不足三个或含 NaN/Inf 时不参与位置损失 |

音频代码不执行重采样，输入须使用一致采样率。球谐投影默认采用平面十字阵列，四个位置为 `(±0.4, 0, 0)` 和 `(0, ±0.4, 0)`，通道顺序见 `dataloader/data_process.py` 的 `MIC_POSITIONS`；实际阵列不同时须同步修改。25 个球谐特征是四通道观测的投影，并不意味着拥有 25 个独立传感器。

可通过 `--annotation_version`、`--audio_dir`、`--image_dir`、`--label_dir`、`--gt_dir` 指定相对目录。`--fold N` 优先读取 `foldN_train.txt` / `foldN_val.txt`，也支持 `_shuffled.txt` 命名；未指定 fold 时优先读取 `train_shuffled.txt` / `val_shuffled.txt`。

## 实现保留与清理

保留原完整网络的参数名称和前向计算，包括图像残差块的两次残差相加，以及最后一个时频块的中间输出用于下游融合。这些属于当前实现，未借发布清理改变网络行为。

去除了不被完整网络调用的频带 MoE、其他专家变体、统计可视化收集器，以及仅供其他模型使用的音频谱和 256 像素图像分支。移除未用的 PESQ、joblib、torchaudio 导入；更新 autocast 接口；修正无效位置屏蔽和空验证位置的处理。

项目根目录仅放置本目录内容。数据、检查点、日志及本地配置由 `.gitignore` 排除，历史 Git 元数据不随本代码包提供。

## 验证范围

已用合成数据验证默认完整网络的输出形状、有限数值、清理前后的前向结果一致性、Adam 单步反向更新、验证及检查点保存。默认网络参数量为 42,916,623。验证环境为 Python 3.13、PyTorch 2.14.1 CPU、NumPy 2.5.4、SciPy 1.16.3、spaudiopy 0.2.0。未执行真实数据集的完整训练，也未验证 CUDA 性能或复现论文指标。
