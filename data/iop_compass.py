# data/iop_compass.py —— IOP-Compass 数据集读取接口（平台无关，仅标准库）
#
# 真实布局（2026-10-05 经 data/scan_layout.py 诊断确认）：
#   <root>/Patient_<n>/IOP_<View>_<n>.png  + 同名 .json
#   视角 token：Center / Upper / Down(=下牙弓) / Left / Right
#
# JSON schema（实测）：
#   { patient_id: str, case_name: str, annotation_count: int,
#     teeth: [ { appearance_idx: int, FDI_NUM: str(两位), class_name: str,
#                color: str, pixel_area: float,
#                x_min,y_min,x_max,y_max: int, centroid_x,centroid_y: float,
#                arch: str, contour_count: int,
#                contour: [[x,y]...],            # 主轮廓多边形
#                contours: [[[x,y]...]...] } ] }  # 全部轮廓
#
# 路径一律由调用方传入，本模块无任何硬编码路径（铁律 2）。
# 不做图像解码——调用方按需用 PIL/cv2 读图。

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# 真实视角 token → 标准视角名（注意：数据集用 Down 表示下牙弓）
_VIEW_MAP = {
    "CENTER": "center", "UPPER": "upper", "DOWN": "lower",
    "LEFT": "left", "RIGHT": "right",
}
STD_VIEWS = ("center", "upper", "lower", "left", "right")

_NAME_RE = re.compile(r"^IOP_([A-Za-z]+)_(\d+)\.png$", re.IGNORECASE)

# FDI 恒牙合法编号（11-18 上右 / 21-28 上左 / 31-38 下左 / 41-48 下右）
VALID_FDI = {f"{q}{i}" for q in (1, 2, 3, 4) for i in range(1, 9)}


@dataclass(frozen=True)
class IOPRecord:
    patient_key: str      # 目录名（Patient_<n>），数据集内患者键
    view: str             # 标准视角名（见 STD_VIEWS）
    raw_view: str         # 原始 token（如 Down），追溯命名差异用
    image_path: Path
    annotation_path: Path


@dataclass(frozen=True)
class ToothAnnotation:
    fdi: str                                        # 两位字符串，如 "24"
    arch: str                                       # upper / lower
    bbox: Tuple[int, int, int, int]                 # x_min, y_min, x_max, y_max
    centroid: Tuple[float, float]
    pixel_area: float
    appearance_idx: int
    contour: Tuple[Tuple[int, int], ...]            # 主轮廓多边形
    contours: Tuple[Tuple[Tuple[int, int], ...], ...]  # 全部轮廓


@dataclass(frozen=True)
class ViewAnnotation:
    patient_key: str                          # 目录名
    patient_id: str                           # JSON 内部患者 ID（另一套编号体系）
    case_name: str
    teeth: Tuple[ToothAnnotation, ...]


def iter_records(root: str | Path) -> Iterator[IOPRecord]:
    """遍历数据集，产出每条（患者, 视角）记录。"""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在: {root}")
    for patient_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for png in sorted(patient_dir.glob("*.png")):
            m = _NAME_RE.match(png.name)
            if not m:
                continue                        # 跳过非 IOP 命名的杂散文件
            view = _VIEW_MAP.get(m.group(1).upper())
            if view is None:
                continue                        # 未知视角 token，冒烟清单会暴露
            yield IOPRecord(
                patient_key=patient_dir.name,
                view=view,
                raw_view=m.group(1),
                image_path=png,
                annotation_path=png.with_suffix(".json"),
            )


def load_annotation(rec: IOPRecord) -> dict:
    """原始 JSON dict——逃生口；类型化读取请用 parse_annotation()。"""
    if not rec.annotation_path.exists():
        raise FileNotFoundError(f"标注缺失: {rec.annotation_path}")
    with rec.annotation_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_annotation(rec: IOPRecord) -> ViewAnnotation:
    """解析为类型化 ViewAnnotation。schema 不符时抛 ValueError。"""
    raw = load_annotation(rec)
    teeth: List[ToothAnnotation] = []
    try:
        for t in raw.get("teeth", []):
            teeth.append(ToothAnnotation(
                fdi=str(t.get("FDI_NUM", "")).strip(),
                arch=str(t.get("arch", "")).strip(),
                bbox=(int(t["x_min"]), int(t["y_min"]),
                      int(t["x_max"]), int(t["y_max"])),
                centroid=(float(t["centroid_x"]), float(t["centroid_y"])),
                pixel_area=float(t.get("pixel_area", 0.0)),
                appearance_idx=int(t.get("appearance_idx", -1)),
                contour=tuple((int(p[0]), int(p[1])) for p in t.get("contour", [])),
                contours=tuple(
                    tuple((int(p[0]), int(p[1])) for p in c)
                    for c in t.get("contours", [])
                ),
            ))
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"标注 schema 不符: {rec.annotation_path} ({e})") from e
    return ViewAnnotation(
        patient_key=rec.patient_key,
        patient_id=str(raw.get("patient_id", "")),
        case_name=str(raw.get("case_name", "")),
        teeth=tuple(teeth),
    )


def iter_patients(root: str | Path) -> Dict[str, List[IOPRecord]]:
    """按患者分组，供任务 1 的五视角对齐使用。"""
    patients: Dict[str, List[IOPRecord]] = {}
    for rec in iter_records(root):
        patients.setdefault(rec.patient_key, []).append(rec)
    return patients


if __name__ == "__main__":
    # 冒烟检查：python data/iop_compass.py <数据集根目录>
    # 期望值（2026-10-05 全量诊断基线）：
    #   患者数 1000 / 记录数 5000 / 五视角各 1000 / JSON 解析失败 0
    import sys
    if len(sys.argv) < 2:
        sys.exit("用法: python data/iop_compass.py <数据集根目录>")
    records = list(iter_records(sys.argv[1]))
    patients = iter_patients(sys.argv[1])
    view_hist = {v: 0 for v in STD_VIEWS}
    teeth_per_view = {v: [] for v in STD_VIEWS}
    bad_json = 0
    bad_fdi = 0
    count_mismatch = 0
    for rec in records:
        view_hist[rec.view] += 1
        try:
            ann = parse_annotation(rec)
        except Exception:
            bad_json += 1
            continue
        teeth_per_view[rec.view].append(len(ann.teeth))
        bad_fdi += sum(1 for t in ann.teeth if t.fdi not in VALID_FDI)
        # 顶层 annotation_count 与 teeth 长度一致性抽查
        raw = load_annotation(rec)
        if raw.get("annotation_count", len(ann.teeth)) != len(ann.teeth):
            count_mismatch += 1
    print(f"患者数: {len(patients)}")
    print(f"记录数: {len(records)}")
    print(f"视角分布: {view_hist}")
    print(f"JSON 解析失败: {bad_json}")
    print(f"FDI 非法条数: {bad_fdi}")
    print(f"annotation_count 不一致文件: {count_mismatch}")
    for v in STD_VIEWS:
        xs = teeth_per_view[v]
        if xs:
            print(f"  每图牙数 [{v}]: min={min(xs)} max={max(xs)} mean={sum(xs)/len(xs):.1f}")
