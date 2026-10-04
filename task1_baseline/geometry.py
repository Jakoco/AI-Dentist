# task1_baseline/geometry.py —— 单牙几何特征与 FDI 牙弓排序
#
# 几何特征基于数据集自带的轮廓多边形（实例分割级标注），不碰像素模型。

from __future__ import annotations

from typing import Tuple

import numpy as np

# FDI 牙弓遍历序：右后→前→左后（恒牙）
# 上颌：18..11, 21..28；下颌：48..41, 31..38
_ARCH_SEQ = {
    "upper": [str(fdi) for fdi in list(range(18, 10, -1)) + list(range(21, 29))],
    "lower": [str(fdi) for fdi in list(range(48, 40, -1)) + list(range(31, 39))],
}
ARCH_ORDER = {
    (arch, fdi): i for arch, seq in _ARCH_SEQ.items() for i, fdi in enumerate(seq)
}


def arch_order_key(arch: str, fdi: str) -> int:
    """(arch, fdi) → 牙弓排序位。缺失牙不影响（按存在牙排序）。"""
    return ARCH_ORDER.get((arch, fdi), 999)


def contour_orientation(contour) -> float:
    """轮廓主轴方向角（弧度，[-π/2, π/2]）。PCA 第一主成分。

    用途：v0 用每颗牙的姿态角近似"牙齿旋转"，是回弹检测的候选信号之一。
    """
    pts = np.asarray(contour, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    pts = pts - pts.mean(axis=0)
    cov = pts.T @ pts / len(pts)
    try:
        _, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return 0.0
    v = vecs[:, -1]          # eigh 升序返回，最大特征值对应最后一列（主轴）
    ang = float(np.arctan2(v[1], v[0]))
    # 归一到 [-π/2, π/2)：主成分方向无头尾，翻转等价
    if ang >= np.pi / 2:
        ang -= np.pi
    elif ang < -np.pi / 2:
        ang += np.pi
    return ang


def polyline_length(points) -> float:
    """折线总长（像素单位）。"""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))))
