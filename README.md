# D2NN：衍射深度神经网络 MNIST 分类

基于 PyTorch 数值仿真的 5 层衍射深度神经网络（Diffractive Deep Neural Network），对 MNIST 手写数字（0–9）做全光分类。复现并对齐 Lin et al., *Science* 361, 1004–1008 (2018) 的太赫兹分类器配置。

## 特点

- 自由空间传播采用**严格瑞利-索末菲角谱法**（精确 `kz = sqrt(k²−kx²−ky²)`，正确处理大角度与倏逝波），传递函数在初始化时预计算。
- 相位调制采用 **sigmoid 约束**，相位严格限制在 `[0, π]`，相位-only。
- 训练结果：测试集准确率 **92.89%**（论文数值仿真约 91.75%）。

## 目录结构

```
train/
  helper.py              光学模型：传播层、调制层、成像层、完整网络
  torch_prepro.py        MNIST 预处理，生成 200×200 光学输入场
  train_main.py          训练脚本
  weights_large_uniform.pt   已训练权重（5 层）
predict/
  helper.py              预测侧模型（额外提供 forward_with_intensity）
  predict.py             单样本推理与可视化
  combined_figure.png    输入场 + 成像强度示例
```

## 关键参数

| 参数 | 值 |
|---|---|
| 工作频率 / 波长 | 0.4 THz / 0.75 mm |
| 平面边长 / 神经元数 | 8 cm / 200×200（像素间距 0.4 mm） |
| 层数 / 层间距 | 5 层 / 30 mm |
| 探测器边长 | 12 像素（约 4.8 mm） |
| batch / epochs / 学习率 | 8 / 10 / 0.001 |
| 优化器 / 损失 | Adam / MSE |

## 环境依赖

- Python 3
- torch
- numpy
- matplotlib
- tqdm
- torchvision

可按需安装：

```bash
pip install torch numpy matplotlib tqdm torchvision
```

## 使用方法

按顺序执行预处理 → 训练 → 预测：

```bash
# 1) 预处理：首次运行会自动下载 MNIST，输出到 train/preprocessed
python train/torch_prepro.py

# 2) 训练：权重保存到 train/weights_large_uniform.pt
python train/train_main.py

# 3) 预测：输入 0–9999 的样本索引，输出分类结果并保存 predict/combined_figure.png
python predict/predict.py
```

> 全量预处理数组约 10 GB，建议在内存充足的机器上运行（预处理固定使用 CPU）。

## 参考文献

Lin, X. et al. "All-optical machine learning using diffractive deep neural networks." *Science* 361, 1004–1008 (2018).
