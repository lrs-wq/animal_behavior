# 0005：整数后端用户 ID 与用户名分离

- 状态：accepted
- 日期：2026-09-07
- 范围：SQLite 用户登记、已有队列及后端 JSON 类型

## 决策

1. users 表使用 user_id INTEGER PRIMARY KEY，新增 username TEXT NOT NULL UNIQUE；sources/outbox 的 user_id 使用整数。userId 为后端业务主键，username 仅用于管理展示，二者不能互换。
2. 外发继续使用 eventId/userId/eventType 三字段 JSON，userId 必须为正整数 number。登记入口及发送器均拒绝用户名、数字字符串、浮点数和布尔值；用户名不外发。
3. v1 不自动推断数字用户 ID。通过 migrate-users 的标准输入显式提供完整映射，迁移持有发送器锁，先做 SQLite 一致性备份，在同一写事务中重建表及检查外键，失败回滚。
4. 迁移是同一真实用户的字段语义纠正，保持源版本、事件 ID、顺序、读取游标及暂停状态；只修改未尝试或明确鉴权拒绝的活跃载荷，拒绝改写结果不确定的活跃请求。终态原始载荷保留作历史证据，不能自动复活。
5. 用户名更新不影响同一数字 ID 的视频源绑定与待发载荷。真正改变视频源归属仍沿用版本更新、旧队列取消和生产者重启规则。
6. 所有真实映射、队列及备份继续只位于 /data；源码在 /root，凭据在 /etc。回退需匹配代码与数据库版本并对账新增数据。

## 验证

- tests/test_alerts.py：数字 JSON、数据库字段类型、旧库迁移与备份、错误回滚、不确定投递拒绝迁移、用户名变更与冲突隔离。
- tests/test_live_inference_interval.py：数字用户登记后直播入口正常解析来源。
- docs/backend_event_delivery.md：登记、迁移、发送恢复和回退说明。
