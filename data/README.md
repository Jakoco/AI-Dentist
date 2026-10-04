# data/ — 数据来源、授权与获取

本目录只放数据说明、获取流程与读取接口；**原始数据与模型权重永不进 git**（铁律 4）。

## IOP-Compass（主数据集）

- 来源：https://ditto.ing.unimore.it/iop-compass/ （AImageLab, University of Modena）
- 规模：1,000 患者 × 5 标准正畸视角 = 5,000 张口内照片，专家标注 FDI 牙实例
- 内容：每患者一个目录，`IOP_<View>_N.png` + 同名 `.json`（boxes + FDI）；五视角：Center / Upper / Lower / Left / Right
- 上游：图像源自 Bite2Text 数据集，沿用其标准化采集协议

### 授权与合规

- **使用依据（2026-10-04 邮件确认）**：作者（Federico Bolelli）回复——本项目为比赛项目、仅使用数据集，**按学术规范引用即可**
- **引用义务**（原文 "you must cite"，全量 BibTeX 见 `references.bib`）：
  1. Zelelew et al., *A Public Dataset for Tooth Segmentation in Multi-View Intraoral Photographs*, MICCAI 2026 ODIN —— 数据集本体
  2. Lumetti et al., *Do Multimodal LLMs Understand Intraoral Dental Data? Dataset, Platform, and Baselines*, ECCV 2026 —— 上游 Bite2Text
  3. Borghi et al., *Bits2Bites: Intra-oral Scans Occlusal Classification*, MICCAI 2025 ODIN —— 相关工作
- **禁止事项**：不二次分发原始数据；不上传至任何公开仓库；比赛/研究范围之外的用途需重新邮件确认

### 获取流程（站点需注册登录，Colab 无法自动下载，人工一次）

1. 注册登录 https://ditto.ing.unimore.it/iop-compass/ ，下载完整数据集压缩包
2. 解压，上传至 Google Drive：`MyDrive/Datasets/IOP-Compass/<每患者一个目录>/IOP_*.png + .json`
3. Colab 首格挂载 Drive 后读取：

```python
import sys; sys.path.insert(0, '/content/AI-Dentist')
from data.iop_compass import iter_records, iter_patients
recs = list(iter_records('/content/drive/MyDrive/Datasets/IOP-Compass'))
patients = iter_patients('/content/drive/MyDrive/Datasets/IOP-Compass')
print(len(patients), '患者,', len(recs), '条记录')
```

4. 跑模块自带冒烟检查，核对三项：患者数 ≈ 1000、每患者 5 视角、JSON 全部可解析：

```bash
python data/iop_compass.py /content/drive/MyDrive/Datasets/IOP-Compass
```

5. 冒烟结果记入 `reports/`

### 读取接口

`data/iop_compass.py`：平台无关（仅标准库），把数据集根目录读成"患者 × 5 视角"记录流。不做图像解码，调用方按需用 PIL/cv2 读图；JSON 内部结构以首次真实下载为准，届时在 `load_annotation()` 内补确定性解析。
