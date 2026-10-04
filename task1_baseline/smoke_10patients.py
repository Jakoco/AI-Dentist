# task1_baseline/smoke_10patients.py —— 端到端冒烟：抽样 N 患者 → 每患者一份 checkpoint
#
# 链路：iter_patients（数据接口）→ build_dossier（几何参数化）→ save_dossier（checkpoint JSON）
# 用途：任务 1 v0 全链路的首次贯通验证。期望值见打印尾部核对清单。
#
# 用法（Colab，仓库根目录下）：
#   python -m task1_baseline.smoke_10patients \
#       "/content/drive/MyDrive/Datasets/AI-Dentist_dataset/IOP-Compass" \
#       "/content/drive/MyDrive/Datasets/AI-Dentist_dataset/checkpoints/smoke10" \
#       --patients 10

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from data.iop_compass import iter_patients
from task1_baseline.dossier import build_dossier, dossier_teeth_count, save_dossier

EXPECTED_VIEWS = {"center", "upper", "lower", "left", "right"}


def main() -> int:
    ap = argparse.ArgumentParser(description="10 患者端到端 checkpoint 冒烟")
    ap.add_argument("root", help="IOP-Compass 数据集根目录")
    ap.add_argument("out_dir", help="checkpoint 输出目录（建议 Drive 路径）")
    ap.add_argument("--patients", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    patients = iter_patients(args.root)
    if len(patients) < args.patients:
        print(f"!! 患者总数 {len(patients)} < 抽样数 {args.patients}")
        return 1
    keys = sorted(random.Random(args.seed).sample(sorted(patients), args.patients))

    print(f"== 抽样 {len(keys)} 患者（seed={args.seed}）: {keys[0]} ... {keys[-1]}")
    problems = []
    total_teeth = 0
    for key in keys:
        dossier = build_dossier(key, patients[key],
                                created_by="task1_baseline.smoke_10patients")
        path = save_dossier(dossier, args.out_dir)
        views = set(dossier["views"])
        n_teeth = dossier_teeth_count(dossier)
        total_teeth += n_teeth
        flag = "OK " if views == EXPECTED_VIEWS else "!! "
        if views != EXPECTED_VIEWS:
            problems.append(f"{key}: 视角缺失 {EXPECTED_VIEWS - views}")
        print(f"  {flag}{key}: {len(views)} 视角, {n_teeth} 牙, -> {path.name}")

    # ---------- 核对清单 ----------
    out = Path(args.out_dir)
    files = sorted(out.glob("*.dossier.json"))
    print("\n== 核对清单 ==")
    print(f"  checkpoint 文件数: {len(files)}（期望 {args.patients}）")
    print(f"  总牙数: {total_teeth}（每患者期望 > 40，五视角合计）")
    print(f"  视角问题: {problems if problems else '无'}")
    print(f"  输出目录: {out}")
    ok = len(files) == args.patients and not problems and total_teeth > 40 * args.patients
    print("== 结果:", "PASS" if ok else "CHECK", "==")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
