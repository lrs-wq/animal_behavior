# 已发布的精选权重

本目录包含当前检测、毛色分类、RTMPose 权重，以及 v9 验证 mAP50-95 最优检测候选。

- [当前生产检测](detection_best.pt)
- [v9 验证最优检测候选](detection_v9_map5095_best.pt)
- [毛色分类](coat_classifier_best.pt)
- [RTMPose 真实机位适配](rtmpose_m_camera05_realbbox_stage2_epoch140-56b9fb51_20260715.pth)

[选型依据与使用方法](../../docs/model_weights.md) · [指标与完整校验清单](manifest.json) · [SHA256SUMS](SHA256SUMS)

`configs/models.github.yaml` 使用现行生产模型组合；`configs/models.github.best-map5095.yaml`
使用检测验证候选。两份配置均从仓库根目录运行，保留所有 SHA-256 和允许目录检查。
