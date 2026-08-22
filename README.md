# World-Mix Skeleton (`worldmix-skeleton`)

本包只包含 **骨骼动画** 相关内容：标准 53 骨、GLB 蒙皮、语言/JSON 运动、ViT 对齐与对齐消融。

**不包含**：地形生成、Hunyuan 全链路训练、高斯椭球主交付路径。

> 当前仅 **Diane** 资产完整跑通；API 设计为通用（换 `--glb` + `--command` 即可）。  
> 可独立 `pip install`，不再依赖 World-Mix 主仓库。

---

## 1. 这条路线解决什么问题

外部生成的静态 GLB（例如 Hunyuan shape）**外观各各不同**，但游戏/仿真需要 **固定骨架拓扑** 才能复用同一套动作。

常见失败路径：

| 做法 | 问题 |
|------|------|
| 每个 GLB 单独 fit 一套骨骼拓扑 | 动作库无法复用；骨骼 name/parent 不一致 |
| 把 mesh 当高斯椭球壳来 animate | 稠密 GLB 极慢；纹理/拓扑信息丢失 |
| 跳过对齐直接套模板骨 | 比例严重错位，LBS 扭曲 |

World-Mix 的做法：

1. **不动拓扑**：始终使用标准 53 骨模板（name / parent 不变）
2. **只调比例**：用正面 ViT + 几何 landmark 估计关节，缩放骨长以匹配 GLB
3. **同一套运动**：`text_command_to_motion` 或 motion JSON → `pose_skeleton_frames`（与 `pose_avatar` 相同 FK）
4. **直接驱动 mesh**：顶点 LBS（`skin_vertices`），不是必须走高斯椭球

一句话：**外观来自任意 GLB，运动来自固定标准骨 + 语言指令**。

---

## 2. 管线一览

```
outputs/model_diane/
  shape.glb            # 静态 mesh（~95k 顶点 / ~27 万面）

        ↓  align_glb_to_standard_skeleton (vit_align.py)
        ↓  compute_vertex_skin_weights (glb_skeleton_fit.py)

outputs/shell_diane/                    ← ★ 已验证可行输出
  fitted_textured_shell_walk.gif        # LBS + "walk forward"
  retargeted_skeleton.json              # 重定向后 53 骨
  vit_alignment_report.json             # 关节误差 / 比例
  aligned_rest_shell_front.png          # T-pose 预览

outputs/reference/                      ← 运动模板参考（无 GLB）
  standard_skeleton_gaussian_walk.gif   # 纯标准骨 + 语言 walk（高斯可视化）
  language_motion_wave.json             # wave 动作 JSON 示例
```

**预览渲染说明**：Hunyuan mesh 极稠，全量光栅化很慢；可行预览用 **6000 面 + 正面单视角**（见下方复现命令）。

---

## 3. 快速开始（pip install）

```bash
cd World-Mix-Skeleton-Pack
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # 或 pip install -e .
export PYTORCH_ENABLE_MPS_FALLBACK=1   # Apple Silicon 建议
```

**一条命令出动画**（包内 demo GLB）：

```bash
worldmix-animate outputs/model_diane/shape.glb \
  --output outputs/shell_diane_repro \
  --command "walk forward" \
  --frames 12 --fps 8 \
  --max-faces 6000 --front-only
```

等价 Python API：

```python
from worldmix.character.glb_standard_overlay import build_glb_shell_on_standard_skeleton, save_front_shell_animation
from worldmix.character.motion_adapter import text_command_to_motion

shell = build_glb_shell_on_standard_skeleton("path/to/mesh.glb")
motion = text_command_to_motion("walk forward", frames=12)
save_front_shell_animation(shell, "out/walk.gif", motion_sequence=motion, max_faces=6000)
```

---

## 4. CLI 命令

| 命令 | 作用 |
|------|------|
| `worldmix-animate GLB` | ★ 主入口：GLB → LBS GIF + JSON sidecars |
| `worldmix-language-avatar "walk forward"` | baseline：标准骨 + 语言（无 GLB，高斯可视化） |
| `worldmix-align-study --glb ...` | 对齐消融（需 `[study]` 或 `[dev]` 装 PyYAML） |

`scripts/` 目录保留同名脚本，逻辑与 CLI 一致，例如 `python scripts/animate_glb_textured_shell.py ...`。

---

## 5. 如何复现（逐步）

### 5.1 ★ 主路线：GLB 直接动画

```bash
worldmix-animate outputs/model_diane/shape.glb \
  --output outputs/shell_diane_repro \
  --command "walk forward" \
  --frames 12 --fps 8 \
  --max-faces 6000 \
  --front-only \
  --panel-width 320 --panel-height 480
```

**预期产物**（约 10–20 秒，M1/MPS）：

- `fitted_textured_shell_walk.gif`
- `retargeted_skeleton.json`
- `vit_alignment_report.json`

**可复现性**：2026-08-22 已用包内 `shape.glb` 跑通（4 帧 smoke + 12 帧完整路径均成功；`motion_check.same_pipeline: True`）。

### 5.2 Baseline：无 GLB，仅标准骨 + 语言

```bash
worldmix-language-avatar "walk forward" --output outputs/language_avatar_walk
worldmix-language-avatar "wave hello"  --output outputs/language_avatar_wave
```

### 5.3 对齐参数消融（仅骨骼）

```bash
pip install -e ".[study]"
worldmix-align-study \
  --config experiments/configs/character_alignment_study.yaml \
  --glb outputs/model_diane/shape.glb \
  --output outputs/alignment_study
```

结果表：`outputs/alignment_study/character/character_study_table.csv`

### 5.4 单测

```bash
pytest tests/ -q
```

---

## 6. 文件夹结构

```
World-Mix-Skeleton-Pack/
├── pyproject.toml              # pip install 入口
├── worldmix/
│   ├── character/              # 核心实现
│   ├── eval/                   # 对齐定量（无 terrain）
│   └── cli/                    # worldmix-animate 等
├── scripts/                    # 与 CLI 等价的脚本入口
├── experiments/configs/
│   └── character_alignment_study.yaml
├── outputs/
│   ├── model_diane/            # 输入资产
│   ├── shell_diane/            # ★ 主路线产物（已生成）
│   ├── alignment_study/        # 对齐消融 CSV
│   └── reference/              # 运动 baseline 参考
└── tests/
```

---

## 7. 已打包的关键输出（可直接查看）

| 文件 | 含义 |
|------|------|
| `outputs/shell_diane/fitted_textured_shell_walk.gif` | **主结果**：GLB 被标准骨 + 语言走路驱动 |
| `outputs/shell_diane/retargeted_skeleton.json` | 53 骨 rest pose（已按 mesh 缩放） |
| `outputs/shell_diane/vit_alignment_report.json` | 对齐定量（mean joint err ≈ 0.26） |
| `outputs/alignment_study/character/character_study_table.csv` | ViT vs geometric 等 7 组消融 |
| `outputs/reference/*.gif` | 无 GLB 时的标准骨运动参考 |

---

## 8. 核心代码速查

| 做什么 | 文件 |
|--------|------|
| 标准 53 骨定义 + `pose_avatar` FK | `worldmix/character/standard_skeleton_gaussians.py` |
| 语言 / BVH → 每帧 bone 旋转 | `worldmix/character/motion_adapter.py` |
| GLB → ViT 关节 → 比例 retarget | `worldmix/character/vit_align.py` |
| GLB shell + LBS + 出 GIF | `worldmix/character/glb_standard_overlay.py` |
| 顶点蒙皮 | `worldmix/character/mesh_shell_skinning.py` |
| 对齐指标 + 消融 | `worldmix/eval/metrics.py`, `character_study.py` |

---

## 9. 已知限制（读结果前请看）

| 项 | 说明 |
|----|------|
| 单资产验证 | 目前只有 Diane 白模完整跑通；其他 GLB 需自行试 |
| 无纹理 GLB | 默认粉色；上色需 Hunyuan paint（不在本包） |
| 对齐阈值 | Diane 白模 `alignment_passed: false`（err≈0.26 > 0.18），但拓扑与运动管线正确 |
| ViT vs geometric | 本资产上 **geometric 关节误差更低**（见 `alignment_study` CSV）；可用 `--no-vit` |
| 稠密 mesh | 勿去掉 `--max-faces`；全 mesh 光栅化会极慢 |
| ViT 权重 | 首次需 HF 缓存 `google/vit-base-patch16-224` |
| 废弃路径 | per-GLB 骨骼 morph、高斯壳作为主交付 — **不要用于本任务** |
| 与主仓库 | 包名 `worldmix-skeleton`，import 仍为 `worldmix.*`；勿与完整 `worldmix` 同环境安装 |

---

## 10. 本包 deliberately 不含

- `terrain_study` / 地形 CSV / `run_parameter_study.py`（terrain 部分）
- Hunyuan 权重下载与 shape 推理脚本（输入 GLB 已提供）
- 高斯椭球动画脚本（仅 `outputs/reference/` 留一份运动参考 GIF）

---

## 与主仓库关系

本包从 `/Users/rayana/Desktop/World-Mix` 的 `worldmix/character` + `worldmix/eval` 快照而来，**可独立 pip install**，不再依赖主仓库 `pip install -e .`。

打包日期：2026-08-22
