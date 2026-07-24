#!/usr/bin/env python3
"""Publish the unified cat-pose instructions to Label Studio project 9."""

from __future__ import annotations

import argparse
import json
import sqlite3
import urllib.request
from pathlib import Path


INSTRUCTIONS = """# 猫姿态 17 点统一规则（v1.0）

1. **左右按猫自身方向**，不是画面左右。
2. **看到什么就标什么**：清楚可见的点不得因难标而省略；同名点只能标一次。
3. 可见性：`2=清楚可见`；`1=被遮挡但能可靠推定`；完全不可判断或画面外的点不标（导出为 `0 0 0`）。
4. 关键点固定顺序：鼻尖、左眼、右眼、颈背、尾根、左前肩/肘/爪、右前肩/肘/爪、左后髋/膝/爪、右后髋/膝/爪。
5. 颈背取颈部与躯干连接处的背侧中线；尾根取尾巴与骨盆连接处，**不是尾尖**。
6. 肢体沿“肩/髋 → 肘/膝 → 爪”检查；三点都可见时不得只标其中一个。
7. 主数据集准入：至少 5 个有效点、覆盖头部/躯干轴/前肢/后肢中的至少 3 个区域。未达到时先提交复核，不要为了凑数猜点。

完整规则保存在服务器：`/root/Eating_behavior/docs/cat_pose_annotation_guide.md`。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--project-id", type=int, default=9)
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    owner = connection.execute(
        "select created_by_id from project where id=?", (args.project_id,)
    ).fetchone()
    if owner is None:
        raise LookupError(f"Label Studio project not found: {args.project_id}")
    token_row = connection.execute(
        "select key from authtoken_token where user_id=?", (owner[0],)
    ).fetchone()
    connection.close()
    if token_row is None:
        raise LookupError(f"No API token found for project owner {owner[0]}")

    payload = json.dumps(
        {
            "description": "Cat pose 17-keypoint annotation. Unified policy v1.0 (2026-07-13).",
            "expert_instruction": INSTRUCTIONS,
            "show_instruction": True,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{args.url.rstrip('/')}/api/projects/{args.project_id}",
        data=payload,
        method="PATCH",
        headers={
            "Authorization": f"Token {token_row[0]}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.load(response)
    print(
        json.dumps(
            {
                "id": result.get("id"),
                "title": result.get("title"),
                "show_instruction": result.get("show_instruction"),
                "description": result.get("description"),
                "instruction_characters": len(result.get("expert_instruction") or ""),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
