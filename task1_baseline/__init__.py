# task1_baseline/ —— 任务 1：个人牙齿基准档案（personal baseline dossier）
#
# v0 范围（2026-10-05）：从 IOP-Compass 的成品标注直接提取几何参数，
# 跳过自训分割器（那是阶段 1 的事）。本包三个模块：
#   geometry.py           单牙几何特征（姿态角）+ FDI 牙弓排序
#   dossier.py            档案数据模型 + 构建 + 序列化（任务 1/2 的格式契约，版本化）
#   smoke_10patients.py   端到端冒烟：N 患者 → 每患者一份 checkpoint JSON
