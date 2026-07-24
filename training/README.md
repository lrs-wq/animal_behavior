# 离线模型训练工具

本目录保存检测、花色分类和猫姿态模型的离线训练入口。它不是正式推理运行时的一部分，`src/eating_behavior/` 不应导入这里的代码，生产 worker 也不需要携带训练数据、W&B 日志或训练输出。

## 工具范围

| 入口 | 用途 |
| --- | --- |
| `train.py` | 训练 `cat` / `feeding_head` 检测模型 |
| `train_cls.py` | 划分数据并训练花色图像分类模型 |
| `train_pose.py` | 训练猫 17 关键点姿态模型，支持 W&B sweep |
| `train_pose_noaug_overfit.py` | 无增强小数据过拟合诊断，不作为正式训练配置 |
| `tune.py` | 检测模型超参数搜索 |
| `predict.py` | 用候选检测权重生成验证集预测结果 |

姿态标签定义和主训练集准入规则见 `../docs/cat_pose_annotation_guide.md`。

## 环境

从仓库根目录运行脚本，并使用已经安装 PyTorch、Ultralytics 和相应 GPU 运行时的隔离环境：

```bash
cd /root/Eating_behavior
conda activate yolo
```

不要依赖脚本中的默认数据路径；服务器数据集统一通过显式参数指向 `/data/Eating_behavior/...`。

## 运行前检查

所有主要训练入口均支持 `--dry-run`。先检查数据、模型和输出路径，再开始长时间任务：

```bash
python training/train.py \
  --dataset /data/Eating_behavior/v9_yolo26 \
  --model training/yolo26n.pt \
  --dry-run
```

检测训练示例：

```bash
python training/train.py \
  --dataset /data/Eating_behavior/v9_yolo26 \
  --model training/yolo26n.pt \
  --epochs 100 \
  --device 0 \
  --name cat_feeding_candidate
```

花色分类训练示例：

```bash
python training/train_cls.py \
  --source /data/Eating_behavior/cls_v1 \
  --dataset /data/Eating_behavior/cls_v1_yolo26_cls \
  --model yolo26n-cls.yaml \
  --dry-run
```

姿态训练或单次诊断示例：

```bash
python training/train_pose.py \
  --dataset /data/Eating_behavior/cat_pose_v2 \
  --model training/yolo11n-pose.pt \
  --mode train \
  --wandb-mode offline \
  --dry-run
```

## W&B 行为

- `train.py` 默认关闭 W&B，只有传入 `--enable-wandb` 才启用。
- `train_cls.py` 和 `predict.py` 强制关闭 W&B。
- `train_pose.py` 默认使用在线 W&B；不需要联网时必须显式传入 `--wandb-mode offline` 或 `--wandb-mode disabled`。
- `tune.py` 会启用在线 W&B，运行前确认账号、项目名和网络策略。

W&B 目录、训练运行目录、预测图片和自动生成的数据 YAML 均属于本地实验产物，已由 `.gitignore` 排除。

## 模型进入正式推理的流程

训练结束不代表模型可以直接上线。候选模型必须依次完成：

1. 在独立验证/测试数据上完成指标、错误样本和业务事件回归评估。
2. 记录训练数据版本、脚本参数、基础权重和评估报告。
3. 将批准的权重复制到受控模型目录，例如 `/root/Eating_behavior/models/`。
4. 计算模型文件 SHA-256。
5. 显式更新 `configs/models.yaml` 中的路径和哈希。
6. 运行正式推理测试后再部署 worker。

不得把 `training/runs*`、`training/wandb`、数据集软链接或未经批准的权重批量加入 Git。
