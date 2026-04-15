# 微博互动量预测项目

本项目为**北京理工大学（BIT）《社交网络分析2026》课程期末项目**。

## 项目成员

顾昌宇，刘炳臻，任珍妮，安毅，沙希德。

## 项目简介

本项目的目标是预测微博博文发布一周后的互动量，包括：

- `forward_count`：转发数
- `comment_count`：评论数
- `like_count`：点赞数

给定历史博文数据（`uid`、`mid`、`time`、`content` 以及对应标签），模型需要预测未来博文在发布一周后的互动情况。

当前效果最好的版本基于以下思路：

- 用户历史行为统计特征
- 用户近期时序统计特征
- 预训练句向量特征
- 两阶段建模策略
  - 第一阶段：判断目标是否为 0
  - 第二阶段：对正值样本进行回归预测

这种设计比直接做普通回归更符合比赛的评估指标。

## 环境准备

### 推荐环境

- Python 3.9 或 3.10
- Linux
- CUDA 可选，但推荐使用（主要用于句向量生成）
- GPU 不是必须，但有 GPU 时运行更快

### 创建环境

建议使用 `conda`：

```bash
conda create -n weibo python=3.10 -y
conda activate weibo
````

安装依赖：

```bash
pip install numpy pandas scipy scikit-learn lightgbm joblib torch sentence-transformers
```

## 数据准备

请将数据文件放在本地目录中，例如：

```text
data/
├── weibo_train_data.txt
└── weibo_predict_data.txt
```

### 训练集格式

训练集每一行应包含：

```text
uid    mid    time    forward_count    comment_count    like_count    content
```

### 测试集格式

测试集每一行应包含：

```text
uid    mid    time    content
```

文件使用 **tab 分隔**。

## 主脚本

当前推荐使用的主脚本为：

```text
weibo_lgbm_baseline_v5.py
```

这是目前最推荐使用的版本。

## 配置方式

本项目**不依赖命令行参数**。
所有配置都通过修改 Python 文件底部的 `CONFIG` 字典完成。

示例：

```python
CONFIG = {
    "mode": "validate",
    "train_path": "data/weibo_train_data.txt",
    "test_path": "data/weibo_predict_data.txt",
    "valid_start": "2015-07-01",
    "cv_valid_starts": ["2015-06-01", "2015-07-01"],
    "output_dir": "./outputs/weibo_baseline_v5",
    "submission_name": "submission.txt",

    "use_sentence_embeddings": True,
    "sentence_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "embedding_device": "auto",
    "embedding_batch_size": 128,
    "embedding_normalize": True,
    "embedding_max_chars": 192,
    "embedding_max_seq_length": 192,
    "embedding_text_prefix": "",
    "embedding_cache_dir": "./outputs/weibo_baseline_v5/embedding_cache",

    "recent_ks": [3, 5, 10],
    "recent_days": [7, 14, 30],

    "seed": 42,
    "num_threads": 8,
    "validate_config": None,
}
```

## 使用方法

### 1. 单折验证

用于单次本地验证与快速调参。

将配置中的：

```python
"mode": "validate"
```

然后运行：

```bash
python weibo_lgbm_baseline_v5.py
```

### 2. 严格滚动交叉验证（CV）

用于评估模型在不同时间段上的泛化能力。

将配置中的：

```python
"mode": "cv"
```

然后运行：

```bash
python weibo_lgbm_baseline_v5.py
```

该模式采用按月不重叠切分，例如：

* 使用 2 月到 5 月训练，6 月验证
* 使用 2 月到 6 月训练，7 月验证

相比单折验证，这种方式更适合估计线上表现。

### 3. 全量训练并生成提交文件

用于使用全部训练集进行训练，并对测试集生成最终预测结果。

将配置中的：

```python
"mode": "full"
```

同时设置 `validate_config`，例如：

```python
"validate_config": "./outputs/weibo_baseline_v5/fold_2_2015-07-01/fold_summary.json"
```

然后运行：

```bash
python weibo_lgbm_baseline_v5.py
```

## 输出文件

脚本会将结果保存到 `output_dir` 下。

常见输出包括：

* `validate_config.json`
* `fold_summary.json`
* `cv_summary.csv`
* `cv_summary.json`
* `submission.txt`

### 提交文件格式

最终提交文件格式如下：

```text
uid<TAB>mid<TAB>forward_count,comment_count,like_count
```

其中三个预测值必须为**整数**。

## 推荐使用流程

建议按以下流程使用本项目：

1. 先运行 `validate` 做快速实验
2. 再运行 `cv` 检查不同月份上的稳定性
3. 选择表现最稳的一组配置
4. 最后运行 `full` 生成提交文件

## 说明

* 第一次运行时，程序可能会从 Hugging Face 下载预训练句向量模型
* 句向量会缓存在本地，因此后续重复运行会更快
* GPU 主要用于句向量编码，LightGBM 主体训练主要依赖 CPU
* 如果内存不足，可以适当减小 `embedding_batch_size`
