# data/scan_layout.py —— 数据集布局诊断脚本（一次性，输出有界，防刷屏）
#
# 用途：真实数据集到手后，先跑本脚本摸清三件事——
#   1. 目录结构与命名规则（患者目录长什么样、图片/JSON 怎么命名）
#   2. JSON 标注的真实 schema（键、框格式、FDI 格式）
#   3. 图像基本属性（尺寸、通道数）
# 一切从实际数据发现，不预设任何命名假设。
#
# 用法（Colab）：
#   !python /content/AI-Dentist/data/scan_layout.py \
#       "/content/drive/MyDrive/Datasets/AI-Dentist_dataset/IOP-Compass"
# 把完整输出贴回给 Kimi Code，据此刻 redesign data/iop_compass.py。

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

MAX_EXAMPLES = 5      # 每类最多打印几个示例，防止刷屏


def main(root: Path) -> None:
    print(f"== 根目录: {root}")
    if not root.is_dir():
        sys.exit(f"!! 根目录不存在: {root}")

    # ---------- 1) 目录结构概览 ----------
    all_dirs = [p for p in root.rglob("*") if p.is_dir()]
    files = [p for p in root.rglob("*") if p.is_file()]
    exts = Counter(p.suffix.lower() for p in files)
    depths = Counter(len(p.relative_to(root).parts) for p in files)
    print(f"\n[1] 目录数: {len(all_dirs)}  文件数: {len(files)}")
    print(f"    扩展名分布: {dict(exts)}")
    print(f"    文件深度分布(深度→数量): {dict(sorted(depths.items()))}")

    top = sorted(root.iterdir())
    print(f"    顶层条目数: {len(top)}，前 {min(MAX_EXAMPLES, len(top))} 个:")
    for p in top[:MAX_EXAMPLES]:
        print(f"      {'[D]' if p.is_dir() else '[F]'} {p.name}")

    # ---------- 2) 患者目录内部采样 ----------
    patient_dirs = [p for p in top if p.is_dir()]
    print(f"\n[2] 患者目录采样（前 {min(MAX_EXAMPLES, len(patient_dirs))} 个的内部文件）:")
    for d in patient_dirs[:MAX_EXAMPLES]:
        names = sorted(x.name for x in d.iterdir())
        print(f"    {d.name}/  ({len(names)} 个文件)")
        for n in names:
            print(f"      {n}")

    # ---------- 3) 文件名模式聚类 ----------
    def pattern(name: str) -> str:
        return re.sub(r"\d+", "N", name)

    pats = Counter(pattern(p.name) for p in files)
    print(f"\n[3] 文件名模式 Top{MAX_EXAMPLES}（N = 数字段）:")
    for pat, c in pats.most_common(MAX_EXAMPLES):
        print(f"    {c:6d}  ×  {pat}")

    # ---------- 4) PNG ↔ JSON 同名配对率 ----------
    pngs = [p for p in files if p.suffix.lower() == ".png"]
    missing = [p for p in pngs if not p.with_suffix(".json").exists()]
    print(f"\n[4] PNG 总数: {len(pngs)}；缺同名 JSON: {len(missing)}")
    for p in missing[:MAX_EXAMPLES]:
        print(f"    缺: {p.relative_to(root)}")

    # ---------- 5) JSON schema 采样 ----------
    jsons = [p for p in files if p.suffix.lower() == ".json"]

    def skeleton(obj, indent=6) -> None:
        pad = " " * indent
        if isinstance(obj, dict):
            for k, v in obj.items():
                extra = f" (len={len(v)})" if hasattr(v, "__len__") else ""
                print(f"{pad}{k}: {type(v).__name__}{extra}")
                skeleton(v, indent + 4)
        elif isinstance(obj, list) and obj:
            print(f"{pad}[0..{len(obj)-1}] 元素 {type(obj[0]).__name__}，首元素:")
            skeleton(obj[0], indent + 4)

    print(f"\n[5] JSON schema 采样（前 {min(MAX_EXAMPLES, len(jsons))} 个的结构骨架）:")
    for jp in jsons[:MAX_EXAMPLES]:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            print(f"    --- {jp.relative_to(root)}")
            skeleton(data)
        except Exception as e:
            print(f"    !! 解析失败 {jp.name}: {e}")

    # ---------- 6) 关键词线索 ----------
    print(f"\n[6] 关键词线索（粗扫前 {min(50, len(jsons))} 个 JSON 的文本）:")
    key_counts = Counter()
    for jp in jsons[:50]:
        try:
            txt = jp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for key in ("fdi", "FDI", "tooth", "Tooth", "box", "Box", "bbox",
                    "polygon", "mask", "segmentation", "label", "quadrant"):
            if key in txt:
                key_counts[key] += 1
    print(f"    {dict(key_counts)}")

    # ---------- 7) 图像属性采样 ----------
    print(f"\n[7] 图像属性采样:")
    try:
        from PIL import Image
        for p in pngs[:MAX_EXAMPLES]:
            try:
                with Image.open(p) as im:
                    print(f"    {p.name}: size={im.size} mode={im.mode}")
            except Exception as e:
                print(f"    !! {p.name}: {e}")
    except ImportError:
        print("    （无 PIL，跳过；Colab 默认自带）")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法: python data/scan_layout.py <数据集根目录>")
    main(Path(sys.argv[1]))
