# data/iop_compass.py —— IOP-Compass 数据集读取接口（平台无关，仅标准库）
#
# 职责：把本地根目录（Colab 挂载的 Drive 或任意本地副本）读成
# 统一的"患者 × 5 视角"记录流。不做图像解码——调用方按需用 PIL/cv2 读图。
# 路径一律由调用方传入，本模块无任何硬编码路径（铁律 2）。
#
# 数据布局（每患者一个目录，详见 data/README.md）：
#   <root>/<患者目录>/IOP_Center_1.png  + IOP_Center_1.json   # boxes + FDI
#   <root>/<患者目录>/IOP_Upper_1.png   + ...
#   ...（五视角：Center / Upper / Lower / Left / Right）

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

# 视角名归一化：数据集中可能出现的写法 → 标准五视角
_VIEW_ALIASES = {
    "CENTER": "center", "FRONTAL": "center", "FRONT": "center",
    "UPPER": "upper", "MAXILLARY": "upper",
    "LOWER": "lower", "MANDIBULAR": "lower",
    "LEFT": "left", "RIGHT": "right",
}
STD_VIEWS = ("center", "upper", "lower", "left", "right")

_NAME_RE = re.compile(r"^IOP_([A-Za-z]+)_\d+\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class IOPRecord:
    patient_id: str          # = 患者目录名
    view: str                # 标准视角名（见 STD_VIEWS）
    image_path: Path
    annotation_path: Path    # 同名 .json；文件缺失在 load_annotation 时抛错


def _norm_view(token: str) -> Optional[str]:
    return _VIEW_ALIASES.get(token.upper())


def iter_records(root: str | Path) -> Iterator[IOPRecord]:
    """遍历数据集，产出每条（患者, 视角）记录。"""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在: {root}")
    for patient_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for png in sorted(patient_dir.glob("*.png")):
            m = _NAME_RE.match(png.name)
            if not m:
                continue                    # 跳过非 IOP 命名的杂散文件
            view = _norm_view(m.group(1))
            if view is None:
                continue                    # 未知视角名，留给冒烟清单核对
            yield IOPRecord(
                patient_id=patient_dir.name,
                view=view,
                image_path=png,
                annotation_path=png.with_suffix(".json"),
            )


def load_annotation(rec: IOPRecord) -> dict:
    """返回解析后的 JSON（boxes + FDI）。
    首次真实下载前，内部字段结构以站点说明为准；拿到数据后在
    此补确定性解析，并同步更新 data/README.md。"""
    if not rec.annotation_path.exists():
        raise FileNotFoundError(f"标注缺失: {rec.annotation_path}")
    with rec.annotation_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_patients(root: str | Path) -> Dict[str, List[IOPRecord]]:
    """按患者分组，供任务 1 的五视角对齐使用。"""
    patients: Dict[str, List[IOPRecord]] = {}
    for rec in iter_records(root):
        patients.setdefault(rec.patient_id, []).append(rec)
    return patients


if __name__ == "__main__":
    # 冒烟检查：python data/iop_compass.py <数据集根目录>
    import sys
    if len(sys.argv) < 2:
        sys.exit("用法: python data/iop_compass.py <数据集根目录>")
    records = list(iter_records(sys.argv[1]))
    patients = iter_patients(sys.argv[1])
    view_hist = {v: 0 for v in STD_VIEWS}
    bad_json = 0
    for rec in records:
        view_hist[rec.view] += 1
        try:
            load_annotation(rec)
        except Exception:
            bad_json += 1
    print(f"患者数: {len(patients)}")
    print(f"记录数: {len(records)}")
    print(f"视角分布: {view_hist}")
    print(f"JSON 解析失败: {bad_json}")
