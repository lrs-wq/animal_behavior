# 源码同步与演示素材范围

阿里云 Codeup 继续作为默认代码跟踪仓库：`origin` 指向既有的
`Eating_behavior_formal_project.git`，默认推送和分支 upstream 均使用 `origin`。
GitHub `lrs-wq/animal_behavior` 通过独立 `github` remote 接收经过核对的源码提交。

2026-09-30 首次同步在 `codex/github-sync-20260930` 分支汇总服务器正式版本与本地
统一可视化代码、采集文档，并将同一提交发布到 Codeup 同名分支和 GitHub `main`。
服务器 `video-stream-test` 保留既有阿里云跟踪；本地研发分支和同步分支均跟踪 `origin`。
同步分支不改变已运行服务的工作目录或配置。

## 上传范围

- 推理与离线训练源码、测试、依赖声明、配置模板、部署模板、文档和知识库。
- `docs/demo/` 内明确列出的一个标注 MP4 和两张 JPEG，用于直接查看输出样式。
- `.gitignore` 默认排除其他图片和视频；新增演示素材需逐文件扩展白名单并检查内容。
- 用户随后授权的四个精选模型权重在 `models/weights/` 逐文件分发，清单与依据见 `docs/model_weights.md`。
- 其余模型权重、数据集、原始媒体、抽帧结果、依赖目录、缓存、日志、运行事件、SQLite、
  客户对应关系和凭据均通过仓库外的数据目录管理。

生产注册表中的模型路径和 SHA-256 保持受控资产约定。首次克隆后可用
`configs/models.github.yaml` 加载仓库内精选副本；在仓库根目录校验清单并安装对应依赖。
发布副本不自动替换服务器生产模型；离线演示媒体可以直接播放。

## 后续同步

先查看当前分支、工作区和跟踪关系，核对代码、知识库、测试与素材范围。
保留已有 `origin` URL，按需增加 GitHub remote：

```bash
git remote -v
git branch -vv
git remote add github git@github.com:lrs-wq/animal_behavior.git
git config remote.pushDefault origin
```

`github` 已存在时无需再次添加。完成审查与提交后，先推送当前发布分支到 Codeup，
再将同一提交推送到 GitHub `main`。以下命令在已确认的发布分支执行：

```bash
git push --set-upstream origin HEAD
git push github HEAD:main
git rev-parse HEAD
git ls-remote origin "refs/heads/$(git branch --show-current)"
git ls-remote github refs/heads/main
git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'
```

最后核对三个提交 SHA 一致，upstream 为 `origin/<发布分支>`。GitHub 推送不使用
`--set-upstream`；不使用全仓库 mirror 或强制推送。若 GitHub `main` 已有新的提交，
先检查并合并差异，再进行普通推送。

## 当前本机 Git 连接

本机没有独立的 Codeup 公钥权限，因此本项目的 Git SSH 请求通过已授权的
阿里云服务器 `39.106.120.253` 转发，使用服务器既有 Codeup/GitHub SSH 认证。
本机私有 Git 元数据中的 `ssh-via-aliyun.py` 由 `core.sshCommand` 调用；
私钥留在服务器，`origin` 的 URL 仍指向 Codeup。连接需要该服务器可用。

此配置只记录在当前本机仓库的 `.git` 中，不随源码分发。检查配置：

```bash
git config --local --get core.sshCommand
git config --local --get remote.pushDefault
git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'
```

其他机器可配置自己的 Git SSH 权限，或从已授权服务器执行前述同步命令。
