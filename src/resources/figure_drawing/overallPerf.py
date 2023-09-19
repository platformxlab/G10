# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from statsFiguresUtil import *
from figureUtils import *

inc_base = True

# hatch_color = "#78756e"
hatch_color = "white"

if inc_base:
  color_arr = colors_roller_6
  hatch_arr = ["", "-", "/", "\\", "x", ""]
else:
  color_arr = colors_roller_4
  hatch_arr = ["", "-", "x", "", "/", "\\"]

# plot font size options
plt.rc("font", size=11)
plt.rc("xtick", labelsize=18)
plt.rc("ytick", labelsize=15)
plt.rc("legend", fontsize=18)
plt.rc("hatch", color="white")
mpl.rcParams["axes.labelsize"] = 22
mpl.rcParams["hatch.linewidth"] = 1.8
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams.update({'font.size': 16})
mpl.rcParams.update({'font.family': 'serif'})

fig, ax0 = plt.subplots(1, 1, figsize=(14, 2.5))
plt.subplots_adjust(top = 1.02, bottom=0.01, hspace=0.6, wspace=0.20)

if inc_base:
  data_file = f"overall_performance/all.base.txt"
else:
  data_file = f"overall_performance/all.txt"

if inc_base:
  bar_width = 0.105
  horiz_margin = 0.6
  horiz_major_tick = 0.75
else:
  bar_width = 0.135
  horiz_margin = 0.6
  horiz_major_tick = 0.7
try:
  with open(data_file, "r") as f:
    lines = f.read()
except Exception as e:
  exit(1)

settings = []
workloads = []
sections = lines.strip().split("\n\n")
settings = [setting_translation[setting.strip()] for setting in sections[0].split("|")]
data_array = np.zeros((len(sections) - 1, len(settings)))
settings = settings[:-1]
x_tick_array = np.zeros((len(sections) - 1, len(settings)))
for section_idx, section in enumerate(sections[1:]):
  lines = section.strip().split("\n")
  workload = lines[0].strip()
  workloads.append(workload)
  data_array[section_idx, :] = np.array([1 / float(data) for data in lines[1].split()]) / 1000
  x_tick_array[section_idx, :] = section_idx * horiz_major_tick + (np.arange(len(settings)) - (len(settings) - 1) / 2) * bar_width
for j, setting in enumerate(settings):
  if inc_base:
    base = -1
  else:
    base = 0
  ax0.bar(x_tick_array[:, j], data_array[:, j] / data_array[:, base], color=color_arr[j], width=bar_width, edgecolor=hatch_color, hatch=hatch_arr[j], label=setting, zorder=3)
# for j, setting in enumerate(settings):
  ax0.bar(x_tick_array[:, j], data_array[:, j] / data_array[:, base], color="none", width=bar_width, edgecolor="white", linewidth=0.8, zorder=3)
# for x_tick, data in zip(x_tick_array[:, 0], data_array[:, 0]):
#   ax.text(x_tick, 1.05, f"{data:5.2f} kops/s", ha="center", va="bottom", rotation=90, fontsize=10)

ax0.set_xticks(np.arange(len(workloads)) * horiz_major_tick)
ax0.set_xticklabels([workload.replace("|", "\n") for workload in workloads])
ax0.set_xlim([-horiz_margin * horiz_major_tick, (len(workloads) - 1 + horiz_margin) * horiz_major_tick])
if inc_base:
  ax0.set_yticks(np.arange(0, 1.2, 0.2))
  ax0.set_ylim([0, 1.4])
else:
  ax0.set_yticks(np.arange(0, 1.8, 0.25))
  ax0.set_ylim([0, 2.1])
ax0.set_ylabel("Normalized\nPerformance", fontsize=20)
# ax0.yaxis.set_label_coords(-0.06, 0.4)
# ax0.hlines(y=1, xmin=ax0.get_xlim()[0], xmax=ax0.get_xlim()[1], colors="grey", linestyles="--")
ax0.yaxis.grid(zorder=0)
ax0.hlines(0, xmin=ax0.get_xlim()[0], xmax=ax0.get_xlim()[1], zorder=9, color='black', linewidth=1)

handles, labels = ax0.get_legend_handles_labels()
if inc_base:
  legend = ax0.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, columnspacing=2.5)
else:
  legend = ax0.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, columnspacing=1.5)

# shift = max([t.get_window_extent().width for t in legend.get_texts()])
# for t in legend.get_texts():
#     t.set_position(((shift - t.get_window_extent().width) / 2 - 6, 0))

plt.show()

extent = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).expanded(0.85, 1.45)
x_len, y_len = extent.x1 - extent.x0, extent.y1 - extent.y0
if inc_base:
  extent.y0 -= y_len * 0.08
else:
  extent.y0 -= y_len * 0.02
extent.y1 -= y_len * 0.11
extent.x0 -= x_len * 0.04
extent.x1 -= x_len * 0.02
# figname = list(net_name_translation.values())[model]
# fig.savefig(f"OverallPerf{figname}.png", bbox_inches=extent)
if inc_base:
  fig.savefig(f"output/OverallPerfNew.png", bbox_inches=extent)
  fig.savefig(f"output/OverallPerfNew.pdf", bbox_inches=extent)
else:
  fig.savefig(f"output/OverallPerfRealImpl.png", bbox_inches=extent)
  fig.savefig(f"output/OverallPerfRealImpl.pdf", bbox_inches=extent)

# %%
