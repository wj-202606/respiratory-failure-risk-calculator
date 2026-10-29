# -*- coding: utf-8 -*-
"""
Figure 1: 样本筛选流程图 (v21, 方案 B)
- v21: 开发队列排除框中，区分 "Discharged alive within 24 h" 和 "Death within 24 h of admission"
- v20: eICU-CRD 分支改为 ICU 入科时间口径
       All ICU stays 200,859 -> Met inclusion criteria 4,696 -> Final cohort 4,696 (977 deaths)
       BE model 1,458 (363 deaths); HCO3 model 2,287 (521 deaths)
       Excluded: 196,163 (screening) + 3 (BE out-of-range) / 5 (HCO3 out-of-range)
       标注 "ICU-admission time origin"
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

sys.path.insert(0, BASE)
try:
    from chest_style import apply_style
    apply_style()
except Exception:
    pass

COLOR_BOX       = "#FFFFFF"
COLOR_BOX_EXCL  = "#F5F5F5"
COLOR_BOX_EMPH  = "#E8F0F8"
COLOR_BORDER    = "#333333"
COLOR_ARROW     = "#555555"
COLOR_TEXT      = "#111111"

def draw_box(ax, x, y, w, h, text, face=COLOR_BOX, align="center",
             fontsize=11.5, bold=False, edge=COLOR_BORDER, lw=1.0):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=lw, edgecolor=edge, facecolor=face)
    ax.add_patch(box)
    ha = "center" if align == "center" else "left"
    tx = x + w / 2 if align == "center" else x + 0.15
    ty = y + h / 2
    ax.text(tx, ty, text, ha=ha, va="center",
            fontsize=fontsize, color=COLOR_TEXT,
            fontweight="bold" if bold else "normal",
            linespacing=1.4)

def draw_arrow(ax, x1, y1, x2, y2, color=COLOR_ARROW, rad=0.0, lw=1.2):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(arrow)

# ============================================================
# 画布
# ============================================================
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis("off")

# ============================================================
# 栏 A：开发队列
# ============================================================
ax.text(0.2, 11.5, "A  Development cohort", fontsize=15, fontweight="bold")

CX_A = 2.5
BW_A = 3.2
ys_A = [10.8, 7.2, 4.6]

draw_box(ax, CX_A - BW_A/2, ys_A[0], BW_A, 0.65, "Patients with J96 diagnosis\nn = 4,628", fontsize=11)
draw_box(ax, CX_A - BW_A/2, ys_A[1], BW_A, 0.65, "Final development cohort\nn = 2,028", face=COLOR_BOX_EMPH, bold=True, fontsize=11)
draw_box(ax, CX_A - BW_A/2, ys_A[2], BW_A, 0.65, "Random 7:3 split\n(seed = 123)", fontsize=11)

draw_arrow(ax, CX_A, ys_A[0], CX_A, ys_A[1] + 0.65, lw=1.5)
draw_arrow(ax, CX_A, ys_A[1], CX_A, ys_A[2] + 0.65, lw=1.5)

# 底部 7:3 分叉
y_split = 2.8
sub_w = 1.4
draw_box(ax, CX_A - 1.6, y_split, sub_w, 0.75, "Training\nn = 1,420\n154 deaths", face=COLOR_BOX_EMPH, fontsize=11.5)
draw_box(ax, CX_A + 0.2, y_split, sub_w, 0.75, "Validation\nn = 608\n72 deaths", face=COLOR_BOX_EMPH, fontsize=11.5)
draw_arrow(ax, CX_A, ys_A[2], CX_A - 0.9, y_split + 0.75, lw=1.2)
draw_arrow(ax, CX_A, ys_A[2], CX_A + 0.9, y_split + 0.75, lw=1.2)

# 右侧排除框 (此处为更新后的文字)
ex_A_x = 4.4
ex_A_w = 3.3
ex_text_A = (
    "Excluded (n = 2,600):\n"
    "1. Repeat admissions (n = 1,299)\n"
    "2. No respiratory failure (n = 411)\n"
    "3. Non-pulmonary primary\n   discharge diagnosis (n = 562)\n"
    "4. Discharged alive within 24 h (n = 51)\n"
    "5. Death within 24 h of admission (n = 38)\n"
    "6. Missing >30% labs (n = 197)\n"
    "7. Unknown death status (n = 42)"
)
draw_box(ax, ex_A_x, 7.6, ex_A_w, 2.8, ex_text_A, face=COLOR_BOX_EXCL, align="left", fontsize=11)
draw_arrow(ax, CX_A + 0.05, 9.0, ex_A_x - 0.05, 9.0, rad=0.0, lw=1.2)

# ============================================================
# 栏 B1：MIMIC-IV
# ============================================================
ax.text(9.0, 11.5, "B1  MIMIC-IV v3.1 (single-center)", fontsize=15, fontweight="bold", color="#2F5597")

CX_B1 = 10.5
BW_B1 = 3.4
ys_B1 = [10.8, 8.8]

draw_box(ax, CX_B1 - BW_B1/2, ys_B1[0], BW_B1, 0.65, "Patients with J96 diagnosis (ICD-10)\nn = 20,780", fontsize=11)
draw_box(ax, CX_B1 - BW_B1/2, ys_B1[1], BW_B1, 0.65, "Final MIMIC-IV cohort\nn = 2,456", face=COLOR_BOX_EMPH, bold=True, fontsize=11)

draw_arrow(ax, CX_B1, ys_B1[0], CX_B1, ys_B1[1] + 0.65, lw=1.5)

# 右侧排除框
ex_B1_x = CX_B1 + BW_B1/2 + 0.4
ex_B1_w = 2.6
ex_text_B1 = (
    "Excluded (n = 18,324):\n"
    "- Age < 18 or hospital stay < 24 h\n"
    "- Repeat admissions\n"
    "- Death < 24 h\n"
    "- Other screening rules"
)
draw_box(ax, ex_B1_x, 9.4, ex_B1_w, 1.3, ex_text_B1, face=COLOR_BOX_EXCL, align="left", fontsize=11)
draw_arrow(ax, CX_B1 + 0.05, 10.0, ex_B1_x - 0.05, 10.0, rad=0.0, lw=1.2)

# 底部 BE / HCO3 分叉
y_branch_B1 = 6.8
sub_w_B1 = 1.6
draw_box(ax, CX_B1 - 1.7, y_branch_B1, sub_w_B1, 0.8, "BE model\nn = 434 (88 deaths)", face=COLOR_BOX_EMPH, fontsize=11.5)
draw_box(ax, CX_B1 + 0.1, y_branch_B1, sub_w_B1, 0.8, "HCO$_3^-$ model\nn = 684 (109 deaths)", face=COLOR_BOX_EMPH, fontsize=11.5)

draw_arrow(ax, CX_B1, ys_B1[1], CX_B1 - 0.9, y_branch_B1 + 0.8, lw=1.2)
draw_arrow(ax, CX_B1, ys_B1[1], CX_B1 + 0.9, y_branch_B1 + 0.8, lw=1.2)

ax.text(CX_B1 - 1.8, 7.9, "Excl. missing\npredictors\n(n = 2,022)", ha="right", va="center", fontsize=10.0, color="#555555")
ax.text(CX_B1 + 1.8, 7.9, "Excl. missing\npredictors\n(n = 1,772)", ha="left", va="center", fontsize=10.0, color="#555555")

# ============================================================
# 栏 B2：eICU-CRD（方案 B：ICU 入科时间口径）
# ============================================================
ax.text(9.0, 5.6, "B2  eICU-CRD v2.0 (multicenter; ICU-admission time origin)",
        fontsize=15, fontweight="bold", color="#2F5597")

CX_B2 = 10.5
BW_B2 = 3.6
ys_B2 = [4.8, 3.9, 2.6]

# 起始框：全部 ICU stays
draw_box(ax, CX_B2 - BW_B2/2, ys_B2[0], BW_B2, 0.65,
         "All ICU stays in eICU-CRD v2.0\nn = 200,859", fontsize=11.5)

# 中间框：符合入组标准 4,696
draw_box(ax, CX_B2 - BW_B2/2, ys_B2[1], BW_B2, 0.6,
         "Met inclusion criteria\nn = 4,696 (977 deaths)", fontsize=11.0)

# 最终框
draw_box(ax, CX_B2 - BW_B2/2, ys_B2[2], BW_B2, 0.65,
         "After range filtering\nBE: n = 4,693\nHCO$_3^-$: n = 4,691",
         face=COLOR_BOX_EMPH, bold=True, fontsize=11.5)

# 两段主箭头
draw_arrow(ax, CX_B2, ys_B2[0], CX_B2, ys_B2[1] + 0.6, lw=1.5)
draw_arrow(ax, CX_B2, ys_B2[1], CX_B2, ys_B2[2] + 0.65, lw=1.5)

# 右侧排除框：调窄、下移，避免出框
ex_B2_x = CX_B2 + BW_B2/2 + 0.35   # 更贴近主框
ex_B2_w = 2.8                      # 变窄
ex_B2_y = 3.05                     # 下移
ex_B2_h = 2.05
ex_text_B2 = (
    "Excluded (n = 196,163):\n"
    "- No ICD-9 518.81 primary dx\n"
    "- Age < 18\n"
    "- ICU stay $\\leq$ 24 h or\n   other exclusion criteria\n"
    "- Repeat ICU admissions\n"
    "- Other screening rules\n"
    "\n"
    "Excluded after lab filter:\n"
    "- BE model: n = 3 (out-of-range)\n"
    "- HCO$_3^-$ model: n = 5 (out-of-range)"
)
ex_B2_y = 3.25
ex_B2_h = 2.15
draw_box(ax, ex_B2_x, ex_B2_y, ex_B2_w, ex_B2_h, ex_text_B2,
         face=COLOR_BOX_EXCL, align="left", fontsize=11.0)
# 箭头完全落在起始框与排除框之间的空隙
_x_start = CX_B2 + BW_B2/2 + 0.03
_x_end   = ex_B2_x - 0.08
draw_arrow(ax, _x_start, ys_B2[0] + 0.32,
           _x_end,   ys_B2[0] + 0.32, rad=0.0, lw=1.2)

# 底部 BE / HCO3 分叉
y_branch_B2 = 0.7
sub_w_B2 = 1.9
draw_box(ax, CX_B2 - 2.0, y_branch_B2, sub_w_B2, 0.8, "BE model\nn = 1,458 (363 deaths)", face=COLOR_BOX_EMPH, fontsize=11.5)
draw_box(ax, CX_B2 + 0.1, y_branch_B2, sub_w_B2, 0.8, "HCO$_3^-$ model\nn = 2,287 (521 deaths)", face=COLOR_BOX_EMPH, fontsize=11.5)

draw_arrow(ax, CX_B2, ys_B2[2], CX_B2 - 1.05, y_branch_B2 + 0.8, lw=1.2)
draw_arrow(ax, CX_B2, ys_B2[2], CX_B2 + 1.05, y_branch_B2 + 0.8, lw=1.2)

ax.text(CX_B2 - 2.1, 1.75, "Excl. missing\npredictors\n(n = 3,235)", ha="right", va="center", fontsize=10.0, color="#555555")
ax.text(CX_B2 + 2.1, 1.75, "Excl. missing\npredictors\n(n = 2,404)", ha="left", va="center", fontsize=10.0, color="#555555")
# ============================================================
# 底部说明 & 保存
# ============================================================
ax.text(0.2, 0.1, "ALB = albumin; BUN = blood urea nitrogen; BE = base excess; HCO$_3^-$ = bicarbonate. "
                  "The eICU-CRD cohort uses ICU admission as the time origin (prediction time 24 h after ICU admission).",
        ha="left", va="center", fontsize=10, style="italic", color="#666666")

plt.tight_layout()
out_chest = os.path.join(FIG_DIR_CHEST, "Figure1_flowchart.pdf")
plt.savefig(out_chest, dpi=1200, bbox_inches="tight")
out_old = os.path.join(FIGURES_DIR, "Fig1_flowchart.pdf")
plt.savefig(out_old, dpi=300, bbox_inches="tight")
plt.close()

print(f"已保存: {out_chest}")