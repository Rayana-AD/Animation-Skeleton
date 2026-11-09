# 稠密 mesh 分 batch 预览（实验模块）

> **状态**：代码已接入，主页 **尚未** 采用其输出。48k 面 batched 光栅在 Diane 上曾出现碎面、镂空（见 issue：预览子集与外层壳采样仍待标定）。当前 README 展示仍用 `--max-faces 6000` 稳定路径。

## 动机

- **LBS**：始终在 **全部顶点** 上计算，与预览面数无关。  
- **瓶颈在光栅**：27 万面 GLB 若构建全 mesh 邻接或单帧绘制过多三角面，CPU 预览不可行。  
- **思路**：与大规模点云 attention 的分块计算类似——先把 **绘制** 拆成 batch，**不改动** `pose_skeleton_frames` / `skin_vertices` 关节路径。

## 入口

| 位置 | 作用 |
|------|------|
| `worldmix/character/mesh_rasterize.py` → `rasterize_ortho_batched` | 深度排序三角面，按 `triangle_batch` 写入同一 z-buffer；**不**调用 `build_adjacency()` |
| `worldmix/character/glb_standard_overlay.py` → `_preview_triangles(..., batched_shell=True)` | 正面外层壳 + 可选 `max_faces` 上限 |
| `worldmix/cli/animate.py` | `--batched-preview`、`--raster-batch`、`--outer-keep` |

## 调用示例（实验）

```bash
worldmix-animate mesh.glb --batched-preview --max-faces 48000 \
  --raster-batch 8192 --outer-keep 0.75 --front-only
```

```python
save_front_shell_animation(
    shell, "out.gif",
    motion_sequence=motion,
    max_faces=48000,
    batched_raster=True,
    triangle_batch=8192,
)
```

## 待验证

- [ ] 外层壳 `keep_fraction` 与 `select_render_triangles` 组合下的视觉完整性（避免 T-pose / 动画镂空）  
- [ ] 与 ViT 对齐 rest pose 预览的一致性  
- [ ] 面数 / 分辨率 / 帧率与时间曲线（M1/MPS）

验证通过后再替换 README 主展示 GIF。
