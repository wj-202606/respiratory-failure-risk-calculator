# -*- coding: utf-8 -*-
"""
Fig 1: 样本筛选流程图（CHEST style v3）
- 左右栏统一浅灰蓝配色，加 A/B 面板标签
- ★ v3：右栏 Box 2 从 "(see Methods)" 改为列出 4 条 inclusion criteria
- 输出到 论文图/Figure1_flowchart.pdf
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE, "figures")
FIG_DIR_CHEST = os.path.join(BASE, "论文图")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(FIG_DIR_CHEST, exist_ok=True)

# ---- CHEST style ----
sys.path.insert(0, BASE)
try:
    from chest_style import apply_style
    apply_style()
    print("[style] chest_style applied")
except Exception as _e:
    print(f"[style] 未找到 chest_style，使用默认: {_e}")

COLOR_BOX       = "#E8F0F8"
COLOR_BOX_EMPH  = "#D6E4F0"
COLOR_BOX_TITLE = "#FFFFFF"
COLOR_BORDER    = "#333333"
COLOR_ARROW     = "#4A6FA5"
COLOR_TEXT      = "#222222"

def draw_box(ax, x, y, w, h, text, face=COLOR_BOX,
             fontsize=11, bold=False, edge=COLOR_BORDER, lw=1.0):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=lw, edgecolor=edge, facecolor=face)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, color=COLOR_TEXT,
            fontweight="bold" if bold else "normal")

def draw_arrow(ax, x1, y1, x2, y2, color=COLOR_ARROW):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.1, color=color)
    ax.add_patch(arrow)

# ============================================================
# 画布
# ============================================================
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis("off")

# A / B 面板标签
ax.text(0.5, 10.75, "A", ha="left", va="top",
        fontsize=14, fontweight="bold", color="black")
ax.text(7.5, 10.75, "B", ha="left", va="top",
        fontsize=14, fontweight="bold", color="black")

# ============================================================
# 左栏 A: Development cohort
# ============================================================
BW = 5.0
BH = 0.8
LX = 1.0
CX = LX + BW / 2
ys = [9.2, 7.9, 6.6, 5.3, 3.5]

draw_box(ax, LX, ys[0], BW, BH,
         "Adults with J96 diagnosis (ICD-10)\nn = 15,749", face=COLOR_BOX)

draw_box(ax, LX, ys[1], BW, BH,
         "+ Pulmonary disease as primary\nreason for admission (J-chapter)\nn = 2,559",
         face=COLOR_BOX)

draw_box(ax, LX, ys[2], BW, BH,
         "+ Age ≥18, hospital stay ≥24 h,\nsurvived first 24 h\nn = 2,456",
         face=COLOR_BOX)

draw_box(ax, LX, ys[3], BW, BH,
         "+ Arterial blood gas analysis\nwithin 24 h of admission\nn = 2,028",
         face=COLOR_BOX_EMPH, bold=True)

draw_box(ax, LX, ys[4], BW, 0.6,
         "Random 7:3 split (seed = 123)",
         face=COLOR_BOX_TITLE, fontsize=11)

sub_w = BW / 2 - 0.1
draw_box(ax, LX, ys[4] - 1.4, sub_w, BH,
         "Training subset\nn = 1,421\n159 deaths", face=COLOR_BOX_EMPH)
draw_box(ax, LX + BW / 2 + 0.1, ys[4] - 1.4, sub_w, BH,
         "Internal validation\nn = 607\n67 deaths", face=COLOR_BOX_EMPH)

for i in range(4):
    draw_arrow(ax, CX, ys[i], CX, ys[i + 1] + BH)
draw_arrow(ax, CX, ys[4], LX + sub_w / 2, ys[4] - 1.4 + BH)
draw_arrow(ax, CX, ys[4], LX + BW / 2 + 0.1 + sub_w / 2, ys[4] - 1.4 + BH)

# ============================================================
# 右栏 B: External validation (MIMIC-IV)
# ★ v3：Box 2 加 4 条 inclusion criteria，框高 1.0，字号 10.5
# ============================================================
RX = 8.0
RCX = RX + BW / 2

# 重新定义右栏 ys（因为 Box 2 变高，需微调）
ys_B = [9.2, 7.9, 6.5, 5.1, 3.3]     # ★ 调整 B 栏 y 坐标
BH_B = 0.9                            # ★ B 栏统一框高（原 0.8）

draw_box(ax, RX, ys_B[0], BW, BH_B,
         "Adults with J96 diagnosis (ICD-10)\nin MIMIC-IV v3.1\nn = 20,780",
         face=COLOR_BOX)

# ★ Box 2：4 条 criteria（本框加高到 1.1）
draw_box(ax, RX, ys_B[1] - 0.1, BW, 1.1,
         "+ Inclusion criteria applied:\n"
         "Age ≥18 y; hospital stay ≥24 h;\n"
         "survived first 24 h; first admission\n"
         "n = 2,456",
         face=COLOR_BOX, fontsize=10.5)

draw_box(ax, RX, ys_B[2], BW, BH_B,
         "Final MIMIC-IV cohort\nn = 2,456",
         face=COLOR_BOX_EMPH, bold=True)

draw_box(ax, RX, ys_B[3], BW, BH_B,
         "Complete cases for ALB, BUN, BE\nwithin 24 h of admission\nn = 434 (88 deaths)",
         face=COLOR_BOX_EMPH, bold=True)

draw_box(ax, RX, ys_B[4], BW, 0.6,
         "Sensitivity analysis",
         face=COLOR_BOX_TITLE, fontsize=11)

draw_box(ax, RX, ys_B[4] - 1.4, BW, BH_B,
         "Complete cases for ALB, BUN, HCO$_3^-$\nn = 684 (109 deaths)",
         face=COLOR_BOX)

# 箭头（B 栏，注意 Box 2 变高）
draw_arrow(ax, RCX, ys_B[0], RCX, ys_B[1] - 0.1 + 1.1)              # Box1 → Box2(加高)
draw_arrow(ax, RCX, ys_B[1] - 0.1, RCX, ys_B[2] + BH_B)             # Box2(顶) → Box3
draw_arrow(ax, RCX, ys_B[2], RCX, ys_B[3] + BH_B)
draw_arrow(ax, RCX, ys_B[3], RCX, ys_B[4] + 0.6)
draw_arrow(ax, RCX, ys_B[4], RCX, ys_B[4] - 1.4 + BH_B)

# ============================================================
# 缩写注释 + 分隔线
# ============================================================
ax.text(0.5, 1.0,
        "ABG = arterial blood gas; ALB = albumin; BUN = blood urea nitrogen;\n"
        "BE = base excess; HCO$_3^-$ = bicarbonate.",
        ha="left", va="center", fontsize=9, style="italic", color="#555555")

ax.plot([7.0, 7.0], [1.5, 10.5], color="#CCCCCC", linewidth=0.8, linestyle=":")

plt.tight_layout()

# ============================================================
# 保存
# ============================================================
out_chest = os.path.join(FIG_DIR_CHEST, "Figure1_flowchart.pdf")
plt.savefig(out_chest, dpi=1200, bbox_inches="tight")

out_old = os.path.join(FIGURES_DIR, "Fig1_flowchart.pdf")
plt.savefig(out_old, dpi=300, bbox_inches="tight")

plt.close()

print(f"\n✅ 已保存:")
print(f"   {out_chest}      (CHEST style)")
print(f"   {out_old}        (旧版兼容)")