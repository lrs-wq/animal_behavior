#!/usr/bin/env python3
"""Read-only browser viewer linked to Label Studio editing tasks.

The viewer never writes labels. Each item is mapped to its Label Studio task,
where edits are performed separately. Images and labels stay on the server.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import sqlite3
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class ViewerItem:
    item_id: str
    status: str
    split: str
    source: str
    label_relative: str
    image_path: Path
    filename: str
    present_count: int
    covered_regions: int
    reasons: str
    bbox: tuple[float, float, float, float]
    keypoints: tuple[tuple[float, float, int], ...]
    label_studio_project: int
    label_studio_task: int
    label_studio_url: str

    def public_dict(self) -> dict[str, object]:
        return {
            "id": self.item_id,
            "status": self.status,
            "split": self.split,
            "source": self.source,
            "label": self.label_relative,
            "filename": self.filename,
            "present_count": self.present_count,
            "covered_regions": self.covered_regions,
            "reasons": self.reasons,
            "bbox": self.bbox,
            "keypoints": self.keypoints,
            "label_studio_project": self.label_studio_project,
            "label_studio_task": self.label_studio_task,
            "label_studio_url": self.label_studio_url,
        }


def find_image(dataset: Path, label_relative: Path) -> Path:
    image_base = dataset / "images" / Path(*label_relative.parts[1:])
    for suffix in IMAGE_SUFFIXES:
        candidate = image_base.with_suffix(suffix)
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No image for {dataset / label_relative}")


def read_pose(label: Path) -> tuple[tuple[float, float, float, float], tuple[tuple[float, float, int], ...]]:
    lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Expected one pose instance in {label}, found {len(lines)}")
    values = [float(token) for token in lines[0].split()]
    if len(values) != 56:
        raise ValueError(f"Expected 56 columns in {label}, found {len(values)}")
    bbox = tuple(values[1:5])
    keypoints = tuple(
        (values[5 + index * 3], values[6 + index * 3], int(round(values[7 + index * 3])))
        for index in range(17)
    )
    return bbox, keypoints


def covered_region_count(keypoints: tuple[tuple[float, float, int], ...]) -> int:
    regions = ((0, 1, 2), (3, 4), (5, 6, 7, 8, 9, 10), (11, 12, 13, 14, 15, 16))
    return sum(any(keypoints[index][2] > 0 for index in region) for region in regions)


def load_label_studio_tasks(database: Path) -> dict[tuple[int, str], int]:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    mapping: dict[tuple[int, str], int] = {}
    duplicates: set[tuple[int, str]] = set()
    for task_id, project_id, raw_data in connection.execute(
        "select id, project_id, data from task where project_id in (8, 9)"
    ):
        data = json.loads(raw_data)
        filename = data.get("filename")
        if not isinstance(filename, str) or not filename:
            continue
        key = (int(project_id), filename)
        if key in mapping:
            duplicates.add(key)
        else:
            mapping[key] = int(task_id)
    connection.close()
    for key in duplicates:
        mapping.pop(key, None)
    return mapping


def project_for_label(label_relative: Path) -> int:
    # Project 9 is the 800-image cat/ batch; all color-class folders came from project 8.
    return 9 if len(label_relative.parts) > 2 and label_relative.parts[2] == "cat" else 8


def load_csv_items(
    csv_path: Path,
    dataset: Path,
    status: str,
    id_prefix: str,
    task_map: dict[tuple[int, str], int],
    label_studio_base_url: str,
) -> list[ViewerItem]:
    if not csv_path.exists():
        return []
    result = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows):
        label_relative = Path(row["path"])
        label_path = dataset / label_relative
        bbox, keypoints = read_pose(label_path)
        image_path = find_image(dataset, label_relative)
        project_id = project_for_label(label_relative)
        filename = label_relative.with_suffix(image_path.suffix.lower()).name
        task_id = task_map.get((project_id, filename))
        if task_id is None:
            raise LookupError(f"No unique Label Studio task for project {project_id}: {filename}")
        edit_url = (
            f"{label_studio_base_url.rstrip('/')}/projects/{project_id}/data"
            f"?task={quote(str(task_id))}"
        )
        result.append(
            ViewerItem(
                item_id=f"{id_prefix}{index}",
                status=status,
                split=row.get("split", ""),
                source=row.get("source", ""),
                label_relative=label_relative.as_posix(),
                image_path=image_path,
                filename=image_path.name,
                present_count=sum(point[2] > 0 for point in keypoints),
                covered_regions=covered_region_count(keypoints),
                reasons=row.get("decision_reasons", ""),
                bbox=bbox,
                keypoints=keypoints,
                label_studio_project=project_id,
                label_studio_task=task_id,
                label_studio_url=edit_url,
            )
        )
    return result


def load_items(
    dataset: Path,
    source_dataset: Path,
    task_map: dict[tuple[int, str], int],
    label_studio_base_url: str,
) -> list[ViewerItem]:
    accepted = load_csv_items(
        dataset / "manifest.csv", dataset, "accepted", "a", task_map, label_studio_base_url
    )
    review = load_csv_items(
        dataset / "review_queue.csv", source_dataset, "review", "r", task_map, label_studio_base_url
    )
    items = accepted + review
    items.sort(key=lambda item: (item.status, item.split, item.source, item.filename))
    return items


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>cat_pose_v2 标注查看器</title>
  <style>
    :root { color-scheme:dark; --bg:#0e1116; --panel:#171c24; --line:#2b3442; --text:#e8edf4; --muted:#9aa7b8; --green:#42f57b; --orange:#ffae42; --blue:#65a8ff; }
    * { box-sizing:border-box; } body { margin:0; font-family:system-ui,"Microsoft YaHei",sans-serif; background:var(--bg); color:var(--text); }
    header { position:sticky; top:0; z-index:10; padding:14px 18px; background:rgba(14,17,22,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }
    h1 { margin:0 0 10px; font-size:19px; } .toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
    select,input,button,.button-link { background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:7px; padding:7px 9px; }
    input[type=search] { min-width:260px; } button,.button-link { cursor:pointer; } button:hover,.button-link:hover { border-color:#60728b; }
    .button-link { display:inline-block; color:#d9eaff; border-color:#496c9e; text-decoration:none; white-space:nowrap; }
    .summary { margin-left:auto; color:var(--muted); font-size:13px; } main { padding:16px; }
    .notice { margin:0 0 12px; padding:10px 12px; border:1px solid #44546a; border-radius:8px; background:#172130; color:#cad8e9; font-size:13px; }
    .legend { color:var(--muted); font-size:13px; margin:0 0 12px; } .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin:0 4px 0 12px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:14px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; } .card.review { border-color:#76582a; }
    canvas { display:block; width:100%; height:auto; background:#090b0f; cursor:zoom-in; } .meta { padding:10px 12px; }
    .name { font-size:13px; overflow-wrap:anywhere; } .tags { display:flex; flex-wrap:wrap; gap:6px; margin:7px 0; }
    .tag { padding:2px 6px; border-radius:5px; background:#252d39; color:#bdc8d6; font-size:11px; } .tag.review { background:#533a19; color:#ffd79b; }
    .reason { margin:6px 0; color:#ffca79; font-size:12px; overflow-wrap:anywhere; }
    .pager { display:flex; justify-content:center; align-items:center; gap:10px; margin:18px 0 4px; }
    dialog { width:min(96vw,1500px); max-height:96vh; padding:0; border:1px solid var(--line); border-radius:12px; background:var(--panel); color:var(--text); }
    dialog::backdrop { background:rgba(0,0,0,.82); } .modal-head { display:flex; gap:8px; align-items:center; padding:10px 12px; border-bottom:1px solid var(--line); }
    .modal-head strong { overflow-wrap:anywhere; } .close { margin-left:auto; } .modal-body { display:grid; grid-template-columns:minmax(0,1fr) 310px; max-height:calc(96vh - 54px); }
    .canvas-wrap { overflow:auto; display:grid; place-items:center; background:#090b0f; } #modalCanvas { width:auto; max-width:100%; max-height:calc(96vh - 58px); cursor:default; }
    .details { overflow:auto; padding:12px; border-left:1px solid var(--line); } table { width:100%; border-collapse:collapse; font-size:12px; }
    th,td { padding:5px; border-bottom:1px solid var(--line); text-align:left; } .empty { padding:60px; text-align:center; color:var(--muted); }
    @media (max-width:850px) { .modal-body { grid-template-columns:1fr; } .details { border-left:0; border-top:1px solid var(--line); } .summary { width:100%; margin-left:0; } }
  </style>
</head>
<body>
<header><h1>cat_pose_v2 远程标注查看器</h1><div class="toolbar">
  <select id="status"><option value="all">全部状态</option><option value="accepted">主数据集</option><option value="review">待复核</option></select>
  <select id="split"><option value="all">全部 split</option><option>train</option><option>val</option><option>test</option></select>
  <select id="source"><option value="all">全部来源</option><option>v4</option><option>v9</option></select>
  <select id="minKpts"><option value="0">不限关键点数</option><option value="5">≥ 5 点</option><option value="9">≥ 9 点</option><option value="13">≥ 13 点</option></select>
  <input id="search" type="search" placeholder="搜索文件名"><label><input id="showEdges" type="checkbox" checked> 骨架</label><label><input id="showIds" type="checkbox" checked> 点编号</label>
  <button id="refresh">刷新</button><span class="summary" id="summary">正在加载…</span>
</div></header>
<main>
  <p class="notice">本页面只读。点击“在 Label Studio 中编辑”会打开对应项目与任务；修改不会同步回本查看器，需后续重新导出并更新数据集。</p>
  <p class="legend">绿色点：v=2 清楚可见 <span class="dot" style="background:var(--orange)"></span>橙色点：v=1 遮挡可推断；单击图片查看大图。</p>
  <div id="grid" class="grid"></div><div class="pager"><button id="prev">上一页</button><span id="pageInfo"></span><button id="next">下一页</button></div>
</main>
<dialog id="modal"><div class="modal-head"><strong id="modalTitle"></strong><a class="button-link" id="modalEdit" target="label_studio_editor">在 Label Studio 中编辑</a><button class="close" id="closeModal">关闭</button></div>
  <div class="modal-body"><div class="canvas-wrap"><canvas id="modalCanvas"></canvas></div><div class="details" id="details"></div></div>
</dialog>
<script>
const NAMES=['鼻尖','左眼','右眼','颈背','尾根','左前肩','左前肘','左前爪','右前肩','右前肘','右前爪','左后髋','左后膝','左后爪','右后髋','右后膝','右后爪'];
const EDGES=[[0,1],[0,2],[1,3],[2,3],[3,4],[3,5],[5,6],[6,7],[3,8],[8,9],[9,10],[4,11],[11,12],[12,13],[4,14],[14,15],[15,16]];
const state={page:1,limit:24,total:0,pages:1,items:[]}; const $=id=>document.getElementById(id);
function query(){return new URLSearchParams({status:$('status').value,split:$('split').value,source:$('source').value,min_kpts:$('minKpts').value,search:$('search').value,page:state.page,limit:state.limit}).toString()}
async function load(){ $('summary').textContent='正在加载…'; const response=await fetch('/api/items?'+query()); const data=await response.json(); Object.assign(state,data); render() }
function render(){
  const grid=$('grid');grid.replaceChildren();$('summary').textContent=`共 ${state.total} 张；当前 ${state.page}/${state.pages} 页`;$('pageInfo').textContent=`第 ${state.page} / ${state.pages} 页`;
  $('prev').disabled=state.page<=1;$('next').disabled=state.page>=state.pages;if(!state.items.length){grid.innerHTML='<div class="empty">没有符合条件的图片</div>';return}
  for(const item of state.items){const card=document.createElement('article');card.className='card '+item.status;const canvas=document.createElement('canvas');canvas.title='单击查看大图';canvas.onclick=()=>openModal(item);card.appendChild(canvas);draw(canvas,item,760);
    const meta=document.createElement('div');meta.className='meta';const name=document.createElement('div');name.className='name';name.textContent=item.filename;meta.appendChild(name);
    const tags=document.createElement('div');tags.className='tags';for(const text of [item.status==='accepted'?'主数据集':'待复核',item.split,item.source,`${item.present_count} 点`,`${item.covered_regions} 区域`,`LS P${item.label_studio_project}/T${item.label_studio_task}`]){const tag=document.createElement('span');tag.className='tag '+(item.status==='review'&&text==='待复核'?'review':'');tag.textContent=text;tags.appendChild(tag)}meta.appendChild(tags);
    if(item.reasons){const reason=document.createElement('div');reason.className='reason';reason.textContent=item.reasons;meta.appendChild(reason)}const edit=document.createElement('a');edit.className='button-link';edit.target='label_studio_editor';edit.href=item.label_studio_url;edit.textContent='在 Label Studio 中编辑';meta.appendChild(edit);card.appendChild(meta);grid.appendChild(card)}
}
function draw(canvas,item,maxWidth){const img=new Image();img.decoding='async';img.src='/media?id='+encodeURIComponent(item.id);img.onload=()=>{const scale=Math.min(1,maxWidth/img.naturalWidth,620/img.naturalHeight);canvas.width=Math.max(1,Math.round(img.naturalWidth*scale));canvas.height=Math.max(1,Math.round(img.naturalHeight*scale));const ctx=canvas.getContext('2d');ctx.drawImage(img,0,0,canvas.width,canvas.height);const pts=item.keypoints.map(([x,y,v])=>[x*canvas.width,y*canvas.height,v]);if($('showEdges').checked){ctx.lineWidth=Math.max(2,canvas.width/350);ctx.strokeStyle='#30d5c8';for(const [a,b] of EDGES){if(pts[a][2]>0&&pts[b][2]>0){ctx.beginPath();ctx.moveTo(pts[a][0],pts[a][1]);ctx.lineTo(pts[b][0],pts[b][1]);ctx.stroke()}}}const radius=Math.max(4,canvas.width/150);ctx.font=`bold ${Math.max(11,canvas.width/55)}px sans-serif`;pts.forEach(([x,y,v],i)=>{if(v<=0)return;ctx.beginPath();ctx.arc(x,y,radius,0,Math.PI*2);ctx.fillStyle=v===2?'#42f57b':'#ffae42';ctx.fill();ctx.strokeStyle='#000';ctx.lineWidth=2;ctx.stroke();if($('showIds').checked){ctx.fillStyle='#fff';ctx.strokeStyle='#000';ctx.lineWidth=3;ctx.strokeText(String(i),x+radius+2,y-radius);ctx.fillText(String(i),x+radius+2,y-radius)}})}}
function openModal(item){$('modalTitle').textContent=item.filename;$('modalEdit').href=item.label_studio_url;draw($('modalCanvas'),item,1500);let html=`<p><b>${item.status==='accepted'?'主数据集':'待复核'}</b> · ${item.split} · ${item.source}<br>有效点 ${item.present_count}，区域 ${item.covered_regions}<br>Label Studio：项目 ${item.label_studio_project}，任务 ${item.label_studio_task}</p>`;if(item.reasons)html+=`<p style="color:#ffca79">${item.reasons}</p>`;html+='<table><thead><tr><th>ID</th><th>名称</th><th>x</th><th>y</th><th>v</th></tr></thead><tbody>';item.keypoints.forEach((p,i)=>{html+=`<tr style="opacity:${p[2]?1:.35}"><td>${i}</td><td>${NAMES[i]}</td><td>${p[0].toFixed(4)}</td><td>${p[1].toFixed(4)}</td><td>${p[2]}</td></tr>`});$('details').innerHTML=html+'</tbody></table>';$('modal').showModal()}
for(const id of ['status','split','source','minKpts'])$(id).onchange=()=>{state.page=1;load()};$('search').oninput=()=>{clearTimeout(window.searchTimer);window.searchTimer=setTimeout(()=>{state.page=1;load()},250)};
$('refresh').onclick=()=>load();$('prev').onclick=()=>{state.page--;load()};$('next').onclick=()=>{state.page++;load()};$('showEdges').onchange=$('showIds').onchange=()=>render();$('closeModal').onclick=()=>$('modal').close();$('modal').onclick=e=>{if(e.target===$('modal'))$('modal').close()};load();
</script></body></html>"""


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "CatPoseViewer/2.0"

    @property
    def viewer_server(self) -> "ViewerServer":
        return self.server  # type: ignore[return-value]

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/health":
            self.send_json({"ok": True, "items": len(self.viewer_server.items), "linked_tasks": len(self.viewer_server.items), "read_only": True})
            return
        if parsed.path == "/api/items":
            self.handle_items(parse_qs(parsed.query))
            return
        if parsed.path == "/media":
            self.handle_media(parse_qs(parsed.query))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        self.send_json({"error": "Viewer is read-only; edit in Label Studio"}, HTTPStatus.METHOD_NOT_ALLOWED)

    def handle_items(self, query: dict[str, list[str]]) -> None:
        get = lambda name, default: query.get(name, [default])[0]
        status, split, source = get("status", "all"), get("split", "all"), get("source", "all")
        search = get("search", "").strip().casefold()
        try:
            min_kpts = max(0, int(get("min_kpts", "0")))
            limit = min(60, max(1, int(get("limit", "24"))))
            page = max(1, int(get("page", "1")))
        except ValueError:
            self.send_json({"error": "Invalid numeric query"}, HTTPStatus.BAD_REQUEST)
            return
        items = [item for item in self.viewer_server.items if (status == "all" or item.status == status) and (split == "all" or item.split == split) and (source == "all" or item.source == source) and item.present_count >= min_kpts and (not search or search in item.filename.casefold())]
        total = len(items);pages = max(1, (total + limit - 1) // limit);page = min(page, pages);start = (page - 1) * limit
        self.send_json({"page": page, "limit": limit, "total": total, "pages": pages, "items": [item.public_dict() for item in items[start:start + limit]]})

    def handle_media(self, query: dict[str, list[str]]) -> None:
        item = self.viewer_server.by_id.get(query.get("id", [""])[0])
        if item is None or not item.image_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND);return
        content_type = mimetypes.guess_type(item.image_path.name)[0] or "application/octet-stream"
        size = item.image_path.stat().st_size;self.send_response(HTTPStatus.OK);self.send_header("Content-Type", content_type);self.send_header("Content-Length", str(size));self.send_header("Cache-Control", "private, max-age=300");self.end_headers()
        with item.image_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024): self.wfile.write(chunk)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


class ViewerServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], items: list[ViewerItem]):
        super().__init__(address, ViewerHandler);self.items = items;self.by_id = {item.item_id: item for item in items}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Unified dataset root")
    parser.add_argument("--source-dataset", type=Path, required=True, help="Original dataset for review items")
    parser.add_argument("--label-studio-db", type=Path, required=True)
    parser.add_argument("--label-studio-base-url", default="http://39.106.120.253")
    parser.add_argument("--host", default="127.0.0.1");parser.add_argument("--port", type=int, default=8092)
    return parser.parse_args()


def main() -> int:
    args = parse_args();dataset = args.dataset.resolve();source_dataset = args.source_dataset.resolve();task_map = load_label_studio_tasks(args.label_studio_db)
    items = load_items(dataset, source_dataset, task_map, args.label_studio_base_url);accepted = sum(item.status == "accepted" for item in items);review = len(items) - accepted
    server = ViewerServer((args.host, args.port), items);print(f"Cat pose viewer listening on http://{args.host}:{args.port} (accepted={accepted}, review={review}, linked={len(items)}, read_only=true)", flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
    return 0


if __name__ == "__main__": raise SystemExit(main())
