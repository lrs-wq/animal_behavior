# 架构决策记录

这里保存会长期影响 Eating Behavior 推理、身份、事件和模型治理的设计选择。

应新增 ADR 的情况：

- 改变 `source_id / household_id / cat` 身份作用域；
- 改变 Re-ID、事件状态机或姿态状态语义；
- 更换检测、姿态或身份模型职责；
- 改变训练产物进入生产的准入流程；
- 改变 worker 调度、查看器或部署拓扑。

文件名格式：`NNNN-short-title.md`。使用同目录的 `TEMPLATE.md`。
