# AGENTS.md — AI-Dentist（v1.0，2026-10-03 定稿）

> 本文件是项目的唯一项目书与操作记忆。每阶段结束必须更新"当前进度"再 push。
> 讨论定稿记录：与 Kimi Code 会话完成 v0.1→v1.0（任务定义转向、接口、配置契约、数据策略）。

## 项目

离线辅助筛查软件：正畸矫正完成后为用户建立**个人牙齿基准档案**，之后用户定期自拍口腔视频，软件对比基准量化变化、提示复发风险。

## 最终成品流程（已定，勿偏离）

1. **矫正完成当天**：用户按 APP 引导自拍口腔视频（多角度、有质量门引导）
2. **一次性建模**：软件从视频生成个人基准档案，作为该用户全部后续检查的依据
3. **回弹检查**：用户再次自拍视频，与基准档案比对
4. **输出**：变化量 + 可视化叠加 + 置信度；低置信度给就医引导话术；**永不输出诊断断言**

## 铁律（每次会话开始前必读）

1. `core/` 平台无关：禁止 UI 框架、浏览器 API、系统调用；只接受 numpy/PIL/onnxruntime 类纯算法依赖
2. 代码中无硬编码路径，一切走配置文件或参数
3. 每阶段结束更新本文件（"当前进度"）再 git push
4. 模型文件（.onnx/.pt）不进 git，用 GitHub Release 或 Drive 存放；git 只放获取脚本与版本号
5. 医学定位：全项目措辞为"辅助筛查"；结果必须带置信度 + 低置信度引导就医话术
6. 拍摄质量是第一地基：基准档案质量不合格不收，宁拒勿烂

## 任务定义与技术核心

### 任务 1：基准档案（personal baseline dossier）——**不是三维扫描**

光谱结论：临床级 3D 重建（口扫仪/NeRF/3DGS）对手机视频是研究级难题且对任务 2 过剩。
回弹信号（牙弓变形、拥挤移位、个别牙旋转）在稀疏几何表示上全部可见。

- **v0 定义**：N 个规范视角的归一化图像 + 牙弓曲线参数 + 牙位关键点坐标（+ 可选单目深度图，不指望绝对精度）
- **三段式**（互相独立、各有验收标准）：
  1. 关键帧选择：视频 → 按清晰度/视角多样性挑 N 帧
  2. 牙齿分割：每帧分出牙弓/牙齿区域（公开数据集可 fine-tune 小分割器）
  3. FDI 实例分割 → 牙弓参数化（首选路径，基于 IOP-Compass 数据集自训小型分割器；备选：角点检测 → 拟合牙弓曲线）
- **v2 选项**：真三维（3DGS 等）仅当任务 2 证明稀疏表示不够时才启动
- **验收实验**：已知形变恢复——对图像施加已知形变，验证流水线能否恢复该变化（此实验同时生成任务 2 的训练数据，一份实验两头用）

### 任务 2：回弹检测可视化网络（开发者熟悉，先行的部分）

- **输入**：基准档案 vs 当前档案；**输出**：变化量、变化区域热力图、置信度
- **接口**：`screen_pair(baseline, current) -> ChangeResult`（结构化 dataclass，UI 只负责渲染与话术翻译）
- **训练数据**：合成形变为主（对基准施加已知位移/旋转，真值自带，可无限生成）；真实配对数据（诊所合作）只做**验证集**
- 与任务 1 完全解耦，唯一契约是**基准档案文件格式**

### 关键转向（为什么这样设计）

- **绝对判断 → 个人基准相对比对**：模型只需回答"与你自己矫正完成时相比变了吗"，不评判绝对整齐度——临床最站得住脚的辅助筛查定位
- **公开"正畸回弹"照片数据集不存在**（2026-10 调研结论，见"数据策略"）——此设计绕开标签难题
- **数据飞轮**：每次检查天然产生配对样本（脱敏 + 授权后），越用数据越多

## 技术路线

| 层 | 选型 | 理由 |
|---|---|---|
| 训练 | PyTorch，Colab GPU | 已有工作流 |
| 导出 | ONNX → INT8 量化 | 跨平台唯一通行格式 |
| .exe 推理 | onnxruntime（CPU 版）+ PySide6 | 诊所电脑无 GPU，CPU 版省 500MB 依赖 |
| 打包 | PyInstaller（文件夹模式） | 启动快，方便换模型 |
| 网页（阶段 4） | React + ONNX Runtime Web | 为 React Native 留复用路 |
| APP（远期） | React Native | 复用 React 组件逻辑 |

## 目标硬件与性能预算（硬指标）

- 诊所 Windows 10/11 办公机（8G 内存、无独显、i5 八代以上）
- **单次推理 < 1 秒（CPU）**；模型量化后 **≤ 10MB**；输入 224×224（可配置）
- 写进 config；任何模型结构变更后重新测量并记录到 reports/

## core/ 接口（已定）

```python
@dataclass(frozen=True)
class ChangeResult:          # 同理有 ScreeningResult（拍摄质量门用）
    scores: dict[str, float]   # 各类别/各区域得分
    top_label: str             # 中性枚举键，如 "needs_dentist_review"，无诊断措辞
    change_map: object         # 变化热力图（UI 叠加渲染用）
    change_magnitude: float    # 总变化量
    confidence: float
    low_confidence: bool
    model_version: str

class ScreenEngine:
    def __init__(self, model_dir: str | Path, config: EngineConfig): ...
    def screen(self, image) -> ScreeningResult: ...          # 质量门
    def screen_pair(self, baseline, current) -> ChangeResult: ...  # 回弹检查
```

- 输入只收 PIL Image（文件读取/摄像头采集是 app 层职责）
- class 名单不进 yaml：随模型产物走 `manifest.json`（版本、class_names、预处理哈希）
- 置信度现阶段用 softmax + 阈值；**校准（temperature scaling）列为阶段 1 报告项**

## 配置契约（DEPLOY-LOCK）

`configs/base.yaml` 分两区，`# DEPLOY-LOCK`（core/ 读，训练/部署必须一致，改动 = 新版本模型）与 `# TRAIN-ONLY`（每次实验可漂移）：

- 锁死：`input_hw: [224,224]`（注明 [H,W]）、`resize` 策略（stretch/letterbox/center_crop 三选一钉死）、`color_mode: RGB`、`layout: NCHW`、`scale/mean/std`（全精度数值）
- 纪律 1：训练 import core 的同一份 preprocess 代码（唯一实现，禁止拷贝两份）
- 纪律 2：INT8 会改变输出——smoke_test 对比 fp32/int8 logits 并记录最大漂移
- 纪律 3：版本三钉——onnx opset（17）、onnxruntime、torch 版本在 requirements 钉死

## 数据策略与候选数据集

**结论：公开"正畸回弹"照片数据集不存在**（2026-10 调研）。相邻公开数据集（许可以原文为准，复核入口：
[NIH 系统综述 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11633071/) 、
[ITU 牙科数据集清单](https://github.com/sergiouribe/dental_datasets_itu/blob/main/AI_Dental_Datasets_List.md)）：

| 候选 | 模态 | 许可 | 用途 |
|---|---|---|---|
| Zenodo 口内龋损图像集 | 口内照片 | CC 开放许可 | 阶段 0 首选（模态最接近） |
| 牙龈炎口内图集（1,096 张） | 口内照片 | 期刊数据，通常 CC-BY，下载前确认 | 阶段 0 备选 |
| Dental Radiography Database（Kaggle） | 全景片 | CC BY-SA 4.0 | 阶段 0 备选 / 阶段 1 扩充 |
| DENTEX | 全景片，23,999 牙标注 | 挑战赛条款，需核实 | 阶段 1 考虑 |
| Aariz 头影测量集（figshare） | 侧位头颅片 | CC | 仅当走"测量角度"路线 |

### ★ IOP-Compass（2026-10-04 发现，公开数据首选）

意大利摩德纳大学 AImageLab（MICCAI 2026 ODIN）：**1,000 患者 × 5 标准正畸视角 = 5,000 张口内照片，专家标注 FDI 编号牙实例分割**，患者级隔离冻结划分 + 可复现基准（仓库 https://github.com/AImageLab-zip/IOP-Compass ，数据 https://ditto.ing.unimore.it/iop-compass/ ）。

- **任务 1 直接受益**：五规范视角 = 基准档案视角集；ResNet18 视角分类器用于关键帧后的视角归组；FDI 实例分割替代原"角点检测"路径（掩码 → 每牙轮廓/质心/姿态 → 拟合牙弓曲线，信息量大于角点）
- **任务 2 连锁升级**：合成形变改为真实牙实例级位移/旋转（移动某 FDI 实例已知 δ，标签自带），变化热力图按牙渲染
- **额外样板**：其 Flask+React 临床修正界面是阶段 3 标注工具参考
- **许可红线**：**已获作者邮件确认（2026-10-04）：比赛项目、仅使用数据集，按学术规范引用即可**（引用条目 `data/references.bib`，获取流程 `data/README.md`）；其余红线不变——仓库无 LICENSE（用其代码仍需邮件）；SegmentAnyTooth 权重非商用（产品禁带）；Mask R-CNN 为 vanilla torchvision 可借鉴思路；数据集不二次分发
- **产品路径**：用其数据集自训小型实例分割器（nano 级，契合 <1s/≤10MB），权重自有，导出 INT8 ONNX

- 训练数据主来源：**合成形变**（无限标注）；真实配对只做验证集（牙医金标：Little 指数等）
- 数据来源与授权写进 `data/README.md`（合规红线）

## 目录结构（初始化时按此创建）

```
AI-Dentist/
├── AGENTS.md            # 本文件
├── configs/             # YAML，一份配置对应一次实验
│   └── base.yaml
├── core/                # ★ 平台无关：模型 + 预处理 + 后处理
│   ├── __init__.py
│   ├── model.py         # 加载 ONNX、推理、返回 ChangeResult/ScreeningResult
│   ├── preprocess.py    # 与训练完全同一份实现
│   └── postprocess.py   # 置信度、低置信度标记
├── task1_baseline/      # 任务 1：基准档案（关键帧/分割/关键点 三段）
├── training/            # 任务 2 训练（Colab）：train.py / dataset.py / export_onnx.py
├── app_desktop/         # 阶段 2：PySide6 + PyInstaller packaging/
├── app_web/             # 阶段 4 预留
├── data/                # 数据说明与获取脚本（原始数据不进 git）
│   └── README.md
├── reports/             # 每轮实验报告
└── scripts/
    ├── setup.sh         # Colab 一键环境
    └── smoke_test.py    # 量化后模型最小验证（fp32 vs int8 对比）
```

## 阶段规划

**阶段 0：可行性（1 周）**
- 公开小数据集（口内照片类）训 MobileNetV3 级网络 → 导出 INT8 ONNX → CPU 测速（硬指标：<1s、≤10MB）
- **决定性实验**：已知形变恢复（配准 + 变化量化能否恢复施加的已知变化）
- 产出 `reports/00_feasibility.md` 含真实测速数字；不通过则不进入下一阶段

**阶段 1：模型（3 周）**
- 数据来源与授权落定（data/README.md）
- 任务 1 三段式 v0 + 任务 2 合成数据训练；精度 + 延迟达标
- 产出：Release v0.1 模型 + `reports/01_model.md`

**阶段 2：.exe MVP（3 周）**
- PySide6 最小界面：导入照片/视频 → 推理 → 结果 + 置信度 + 导出；在无 Python 环境机器测试通过
- 产出：可分发文件夹 + `reports/02_mvp.md`；到点即停，宁粗糙也要进反馈

**阶段 3：反馈迭代（2 周）** — 真实用户（牙医/同学）试用，反馈入 reports/，决定下步

**阶段 4：网页版（视结论，4 周）** — React + ONNX Runtime Web，GitHub Pages

**阶段 5（远期）：APP** — React Native，仅当 3/4 证明需求

## 当前进度

- [x] 仓库创建 + 本地初始化
- [x] 项目书 v1.0 定稿（任务定义、接口、配置契约、数据策略与 Kimi Code 讨论完成）
- [x] 同类项目调研：IOP-Compass（公开数据首选，许可红线与产品路径已记录，见"数据策略"）
- [x] 数据合规与接口：作者确认引用即可；`data/README.md`（授权+下载流程）、`data/iop_compass.py`（读取接口，含冒烟自检）、`data/references.bib`（三篇引用）已就位
- [ ] 目录结构初始化（按上节创建骨架 + requirements 钉版本）
- [ ] 基准档案文件格式契约定型（任务 1/2 唯一接口）
- [ ] 阶段 0 启动：任务 1 关键帧选择原型 + 口内分割公开数据调研

## 下一步（第一个小任务）

1. 下载 IOP-Compass → 上传 Drive（流程见 `data/README.md`）→ Colab 跑 `data/iop_compass.py` 冒烟（患者≈1000、每患者 5 视角、JSON 全解析），结果记入 `reports/`
2. 按目录结构初始化骨架，requirements 钉死 torch / onnxruntime / opencv-python-headless
3. 任务 1 第一段：视频关键帧选择原型（任意公开口腔视频可验）

## 关联项目

- **Bristol**（github.com/Jakoco/Bristol）：高尔顿板群智网络研究课题。代码零复用；工作流（Colab 首格模板、实验记录入 reports/ + 打 tag、AGENTS.md 随项目演进重写）整体复用
