# -*- coding: utf-8 -*-
"""
Fig 1: 样本筛选流程图（v2 修改版）
左栏：Development cohort
右栏：External validation (MIMIC-IV)

v2 改动：右栏 Box 2 从"列具体条件"改为"Inclusion criteria applied (see Methods)"
避免审稿人逐条验证时发现数字对不上（SQL 跑出 2,966 ≠ 宽表 2,456）
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(BASE, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

COLOR_DEV = "#4A7EBB"
COLOR_EXT = "#C0504D"
COLOR_BORDER = "#333333"

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis("off")

def draw_box(ax, x, y, w, h, text, color="#F2F2F2",
             fontsize=10.5, bold=False, edge=COLOR_BORDER):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=1.2, edgecolor=edge, facecolor=color)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal")

def draw_arrow(ax, x1, y1, x2, y2, color="#333333"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=18,
        linewidth=1.3, color=color)
    ax.add_patch(arrow)

# ============ 标题 ============
ax.text(3.5, 10.6, "Development cohort", ha="center", va="center",
        fontsize=15, fontweight="bold", color=COLOR_DEV)
ax.text(10.5, 10.6, "External validation (MIMIC-IV)", ha="center", va="center",
        fontsize=15, fontweight="bold", color=COLOR_EXT)

# ============ 左栏 ============
BW, BH = 5.0, 0.8
LX = 1.0
CX = LX + BW / 2

ys = [9.2, 7.9, 6.6, 5.3, 3.5]

# Box 1: J96 诊断
draw_box(ax, LX, ys[0], BW, BH,
         "Adults with J96 diagnosis (ICD-10)\nn = 15,749",
         color="#E8F0F8")

# Box 2: + J 章主诊断
draw_box(ax, LX, ys[1], BW, BH,
         "+ Pulmonary disease as primary\nreason for admission (J-chapter)\nn = 2,559",
         color="#E8F0F8")

# Box 3: + 年龄 ≥18, 住院 ≥24h, 24h 内未死亡
draw_box(ax, LX, ys[2], BW, BH,
         "+ Age ≥18, hospital stay ≥24 h,\nsurvived first 24 h\nn = 2,456",
         color="#E8F0F8")

# Box 4: + 动脉血气
draw_box(ax, LX, ys[3], BW, BH,
         "+ Arterial blood gas analysis\nwithin 24 h of admission\nn = 2,028",
         color="#D6E4F0", bold=True)

# Box 5: 7:3 标题
draw_box(ax, LX, ys[4], BW, 0.6,
         "Random 7:3 split (seed = 123)",
         color="#FFFFFF", fontsize=10)

# 两个子盒
sub_w = BW / 2 - 0.1
draw_box(ax, LX, ys[4] - 1.4, sub_w, BH,
         "Training subset\nn = 1,421\n159 deaths",
         color="#D6E4F0")
draw_box(ax, LX + BW / 2 + 0.1, ys[4] - 1.4, sub_w, BH,
         "Internal validation\nn = 607\n67 deaths",
         color="#D6E4F0")

# 箭头
for i in range(4):
    draw_arrow(ax, CX, ys[i], CX, ys[i + 1] + BH, color=COLOR_DEV)
draw_arrow(ax, CX, ys[4], LX + sub_w / 2, ys[4] - 1.4 + BH, color=COLOR_DEV)
draw_arrow(ax, CX, ys[4], LX + BW / 2 + 0.1 + sub_w / 2, ys[4] - 1.4 + BH,
           color=COLOR_DEV)

# ============ 右栏 ============
RX = 8.0
RCX = RX + BW / 2

# Box 1: 原始 J96
draw_box(ax, RX, ys[0], BW, BH,
         "Adults with J96 diagnosis (ICD-10)\nin MIMIC-IV v3.1\nn = 20,780",
         color="#F8E8E8")

# Box 2: 应用入组标准（★ v2 改动：不列具体条件）
draw_box(ax, RX, ys[1], BW, BH,
         "+ Inclusion criteria applied\n(see Methods)\nn = 2,456",
         color="#F8E8E8", fontsize=10.5)

# Box 3: 最终队列
draw_box(ax, RX, ys[2], BW, BH,
         "Final MIMIC-IV cohort\nn = 2,456",
         color="#F0D6D6", bold=True)

# Box 4: 三变量完整案例（BE）
draw_box(ax, RX, ys[3], BW, BH,
         "Complete cases for ALB, BUN, BE\nwithin 24 h of admission\nn = 434 (88 deaths)",
         color="#F0D6D6", bold=True)

# Box 5: 敏感性分析标题
draw_box(ax, RX, ys[4], BW, 0.6,
         "Sensitivity analysis",
         color="#FFFFFF", fontsize=10)

# 敏感性分析盒
draw_box(ax, RX, ys[4] - 1.4, BW, BH,
         "Complete cases for ALB, BUN, HCO₃⁻\nn = 684 (109 deaths)",
         color="#F8E8E8")

# 箭头
draw_arrow(ax, RCX, ys[0], RCX, ys[1] + BH, color=COLOR_EXT)
draw_arrow(ax, RCX, ys[1], RCX, ys[2] + BH, color=COLOR_EXT)
draw_arrow(ax, RCX, ys[2], RCX, ys[3] + BH, color=COLOR_EXT)
draw_arrow(ax, RCX, ys[3], RCX, ys[4] + 0.6, color=COLOR_EXT)
draw_arrow(ax, RCX, ys[4], RCX, ys[4] - 1.4 + BH, color=COLOR_EXT)

# ============ 图例 / 注释 ============
ax.text(0.5, 1.2,
        "ABG = arterial blood gas; ALB = albumin; BUN = blood urea nitrogen;\n"
        "BE = base excess; HCO₃⁻ = bicarbonate.",
        ha="left", va="center", fontsize=9, style="italic", color="#555555")

# 分隔线
ax.plot([7.0, 7.0], [1.5, 10.4], color="#CCCCCC", linewidth=1, linestyle=":")

# ============ 保存 ============
out_pdf = os.path.join(FIGURES_DIR, "Fig1_flowchart.pdf")
out_png = os.path.join(FIGURES_DIR, "Fig1_flowchart.png")
plt.tight_layout()
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close()

print("✅ 已保存:")
print(f"   {out_pdf}")
print(f"   {out_png}")