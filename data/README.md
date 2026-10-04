# data/ — 数据来源、授权与获取

本目录只放数据说明、获取流程与读取接口；**原始数据与模型权重永不进 git**（铁律 4）。

## IOP-Compass（主数据集）

- 来源：https://ditto.ing.unimore.it/iop-compass/ （AImageLab, University of Modena）
- 规模：1,000 患者 × 5 标准正畸视角 = 5,000 张口内照片，专家标注 FDI 牙实例
- 上游：图像源自 Bite2Text 数据集，沿用其标准化采集协议
- **本地副本**：`/content/drive/MyDrive/Datasets/AI-Dentist_dataset/IOP-Compass`（Google Drive，Colab 挂载访问）

### 授权与合规

- **使用依据（2026-10-04 邮件确认）**：作者（Federico Bolelli）回复——本项目为比赛项目、仅使用数据集，**按学术规范引用即可**
- **引用义务**（原文 "you must cite"，全量 BibTeX 见 `references.bib`）：
  1. Zelelew et al., *A Public Dataset for Tooth Segmentation in Multi-View Intraoral Photographs*, MICCAI 2026 ODIN —— 数据集本体
  2. Lumetti et al., *Do Multimodal LLMs Understand Intraoral Dental Data? Dataset, Platform, and Baselines*, ECCV 2026 —— 上游 Bite2Text
  3. Borghi et al., *Bits2Bites: Intra-oral Scans Occlusal Classification*, MICCAI 2025 ODIN —— 相关工作
- **禁止事项**：不二次分发原始数据；不上传至任何公开仓库；比赛/研究范围之外的用途需重新邮件确认

### 实际布局（2026-10-05 `scan_layout.py` 全量诊断确认）

```
IOP-Compass/
├── Patient_1/     IOP_Center_1.png + IOP_Center_1.json     （10 文件 = 5 视角 × png+json）
│                  IOP_Upper_1.png  + IOP_Upper_1.json
│                  IOP_Down_1.png   + IOP_Down_1.json      ← 下牙弓，注意叫 Down 不是 Lower
│                  IOP_Left_1.png   + IOP_Left_1.json
│                  IOP_Right_1.png  + IOP_Right_1.json
├── Patient_2/     ...
└── Patient_1000/  ...
```

- 患者目录：`Patient_<n>`（n = 1..1000）；目录名是数据集内患者键
- 文件名：`IOP_<View>_<n>.png`，数字后缀与目录编号一致；视角 token：**Center / Upper / Down / Left / Right**
- 全量核对：5,000 PNG + 5,000 JSON，配对零缺失
- 图像属性：约 3700×5000 像素 RGB 临床高清图（训练管线负责降采样，接口不解码图像）

### JSON 标注 schema（实测）

```json
{
  "patient_id": "str（内部患者 ID，另一套编号）",
  "case_name": "str",
  "annotation_count": 12,
  "teeth": [
    {
      "appearance_idx": 0,
      "FDI_NUM": "24",              // 两位字符串；恒牙合法集 11-18/21-28/31-38/41-48
      "class_name": "str",
      "color": "#RRGGBB",
      "pixel_area": 12345.6,
      "x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0,
      "centroid_x": 0.0, "centroid_y": 0.0,
      "arch": "upper",              // upper / lower
      "contour_count": 1,
      "contour": [[x, y], ...],     // 主轮廓多边形（实例分割级标注，优于纯 bbox）
      "contours": [[[x, y], ...]]   // 全部轮廓
    }
  ]
}
```

- 每图牙数随视角不同（下/上颌咬合面约 12，正面最多 24，侧位约 17-18），属正常
- 每牙信息：FDI 编号 + arch 归属 + bbox + 质心 + 像素面积 + 轮廓多边形 → 任务 1 牙弓参数化的直接原料

### 读取与冒烟

```python
import sys; sys.path.insert(0, '/content/AI-Dentist')
from data.iop_compass import iter_records, iter_patients, parse_annotation

recs = list(iter_records('/content/drive/MyDrive/Datasets/AI-Dentist_dataset/IOP-Compass'))
patients = iter_patients('/content/drive/MyDrive/Datasets/AI-Dentist_dataset/IOP-Compass')
ann = parse_annotation(recs[0])      # 类型化 ViewAnnotation
```

```bash
# 冒烟自检（期望值：患者 1000 / 记录 5000 / 五视角各 1000 / JSON 失败 0 / FDI 非法 0）
python data/iop_compass.py "/content/drive/MyDrive/Datasets/AI-Dentist_dataset/IOP-Compass"
```

接口说明：`data/iop_compass.py` 平台无关（仅标准库），三件套——
`iter_records`（记录流）/ `iter_patients`（按患者分组）/ `parse_annotation`（类型化解析；
`load_annotation` 保留为原始 dict 逃生口）。新数据版本落地时先跑 `scan_layout.py` 再对齐本文件。
