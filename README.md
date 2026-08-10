# 基于EEG信号的运动想象脑机接口分类研究：机器学习与深度学习方法比较

## 1. 项目简介

脑机接口（Brain-Computer Interface, BCI）是一种利用脑电信号（Electroencephalography, EEG）实现人脑与外部设备直接通信的技术。其中，运动想象（Motor Imagery, MI）脑机接口通过识别用户想象不同肢体运动时产生的脑电模式，实现无需实际运动的控制方式。

本项目基于 **BCI Competition IV Dataset 2a** 数据集，开展运动想象EEG信号分类研究，对比传统机器学习方法与深度学习方法在脑机接口分类任务中的性能差异。

研究内容包括：

* EEG原始信号读取与分析
* EEG信号预处理
* CSP（Common Spatial Pattern）特征提取
* 传统机器学习分类模型构建
* 深度学习模型构建
* 不同方法性能比较与结果分析

---

# 2. 项目流程

整体实验流程如下：

```
EEG原始数据
        |
        ↓
数据理解与事件解析
        |
        ↓
信号预处理
        |
        ├── 去除EOG通道
        ├── 8-30Hz带通滤波
        └── 标准化
        |
        ↓
特征提取
        |
        └── CSP
        |
        ↓
机器学习分类
        |
        ├── LDA
        ├── SVM
        └── Random Forest
        |
        ↓
深度学习分类
        |
        ├── CNN
        └── EEGNet
        |
        ↓
模型比较与结果分析
        |
        ↓
科研报告
```

---

# 3. 数据集介绍

## BCI Competition IV Dataset 2a

数据集包含：

* 9名被试
* 22个EEG通道
* 3个EOG通道
* 采样频率250Hz
* 四分类运动想象任务

分类类别：

| 标签  | 运动想象类别 |
| --- | ------ |
| 769 | 左手运动想象 |
| 770 | 右手运动想象 |
| 771 | 双脚运动想象 |
| 772 | 舌头运动想象 |

本项目实验使用：

```
A01T.gdf
```

单个被试数据进行模型验证。

---

# 4. 实验环境

## Hardware

GPU:

```
NVIDIA GeForce RTX 4060 Laptop GPU
```

CUDA:

```
CUDA 12.6
```

## Software

```
Python 3.13

PyTorch 2.13.0+cu126

MNE
Scikit-learn
NumPy
Matplotlib
Pandas
```

---

# 5. 数据预处理

## 5.1 GDF数据读取

使用MNE库读取BCI Competition IV Dataset 2a的GDF格式文件。

## 5.2 Epoch划分

根据事件标记提取运动想象时间段：

时间窗口：

```
0.5s - 4s
```

最终数据：

```
X shape:
(288, 22, 876)

y shape:
(288,)
```

## 5.3 EOG去除

删除：

```
EOG-left
EOG-central
EOG-right
```

处理后：

```
22 EEG channels
```

## 5.4 Band-pass滤波

滤波范围：

```
8-30Hz
```

用于提取运动想象相关的：

* μ节律
* β节律

## 5.5 标准化

采用Z-score标准化：

[
X'=\frac{X-\mu}{\sigma}
]

---

# 6. 分类方法

## 6.1 CSP特征提取

Common Spatial Pattern用于增强不同运动想象类别之间的空间差异。

提取：

```
6 CSP features
```

---

# 6.2 传统机器学习模型

## LDA

线性判别分析。

## SVM

支持向量机。

## Random Forest

随机森林分类器。

输入：

```
CSP features
```

---

# 6.3 深度学习模型

## CNN

直接输入原始EEG信号：

```
(1,22,876)
```

## EEGNet

针对EEG信号设计的轻量级深度学习网络。

结构包括：

* Temporal convolution
* Depthwise convolution
* Separable convolution

---

# 7. 实验结果

## 模型分类准确率比较

| 模型                | 输入      | Accuracy       |
| ----------------- | ------- | -------------- |
| LDA+CSP           | CSP     | 67.24%         |
| SVM+CSP           | CSP     | 75.86%         |
| Random Forest+CSP | CSP     | 77.59%         |
| CNN               | Raw EEG | 43.10%         |
| EEGNet            | Raw EEG | 72.76% ± 4.93% |

---

# 8. 实验分析

实验结果表明：

1. CSP结合传统机器学习方法在小样本EEG分类任务中具有较好的性能。

2. Random Forest结合CSP特征取得最高分类准确率，达到77.59%。

3. 普通CNN由于参数量较大，在有限EEG样本下容易出现泛化能力不足的问题。

4. EEGNet通过结合EEG信号特点设计网络结构，在无需人工特征提取的情况下取得72.76%的平均准确率，表现出较好的应用潜力。

---

# 9. 项目目录结构

```
EEG_BCI_Project

├── data
│   ├── raw
│   └── processed
│

├── scripts
│   ├── preprocessing
│   ├── features
│   ├── models
│   └── utils
│

├── results
│   ├── models
│   ├── figures
│   └── logs
│

├── notebooks

└── README.md
```

---

# 10. 后续工作

未来可以进一步研究：

* 多被试联合训练
* FBCSP特征优化
* Transformer-based EEG模型
* 数据增强方法
* 跨被试迁移学习
* 在线BCI系统实现

---

# 作者  lyy ; 23159100076@stu.xidian.edu.cn

本科科研项目

题目：

《基于EEG信号的运动想象脑机接口分类研究：机器学习与深度学习方法比较》
