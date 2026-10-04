# task1_baseline/dossier.py —— 基准档案（dossier）数据模型与构建
#
# dossier 是任务 1 的产物、任务 2 的输入——两份任务之间的唯一文件契约。
# 因此从第一天起版本化：dossier_format_version 变更必须双方同步。
#
# v0.1 设计决定：
#   - 只存几何与元数据，不存像素；图像以路径引用
#   - 按视角 → 牙弓（upper/lower）→ 牙序（FDI 遍历序）组织
#   - 牙弓曲线 v0 = FDI 序质心折线（不做参数拟合，保持保真；拟合是后续分析的事）

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from data.iop_compass import IOPRecord, parse_annotation
from task1_baseline.geometry import arch_order_key, contour_orientation, polyline_length

DOSSIER_FORMAT_VERSION = "0.1"


def _image_size(image_path: Path) -> List[int] | None:
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            return [im.size[0], im.size[1]]
    except Exception:
        return None


def build_dossier(patient_key: str, records: List[IOPRecord],
                  created_by: str = "task1_baseline") -> Dict:
    """从一个患者的 5 条视角记录构建档案 dict（可 json.dumps）。"""
    views: Dict[str, Dict] = {}
    for rec in records:
        ann = parse_annotation(rec)
        arches: Dict[str, Dict] = {}
        for arch in ("upper", "lower"):
            teeth = [t for t in ann.teeth if t.arch == arch]
            if not teeth:
                continue
            teeth.sort(key=lambda t: arch_order_key(arch, t.fdi))
            entries = [{
                "fdi": t.fdi,
                "centroid": [round(t.centroid[0], 2), round(t.centroid[1], 2)],
                "area_px": round(t.pixel_area, 1),
                "bbox": [int(v) for v in t.bbox],
                "orientation_rad": round(contour_orientation(t.contour), 4),
                "contour_points": len(t.contour),
            } for t in teeth]
            polyline = [e["centroid"] for e in entries]
            arches[arch] = {
                "teeth": entries,
                "arch_polyline": polyline,
                "arch_polyline_len_px": round(polyline_length(polyline), 1),
            }
        views[rec.view] = {
            "image": rec.image_path.name,
            "image_size": _image_size(rec.image_path),
            "arches": arches,
        }
    return {
        "dossier_format_version": DOSSIER_FORMAT_VERSION,
        "source": {
            "dataset": "IOP-Compass",
            "patient_key": patient_key,
            "created_by": created_by,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "views": views,
    }


def save_dossier(dossier: Dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dossier['source']['patient_key']}.dossier.json"
    path.write_text(json.dumps(dossier, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def dossier_teeth_count(dossier: Dict) -> int:
    return sum(len(v["arches"][a]["teeth"])
               for v in dossier["views"].values() for a in v["arches"])
