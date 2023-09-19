# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from statsFiguresUtil import *
from figureUtils import *

hatch_arr = ["/", "\\", "", "", "x", "-"]
# hatch_color = "#78756e"
hatch_color = "white"

color_arr = colors_dark4
color_arr = colors_dark7_2
color_arr = colors_dark6
# color_arr[1], color_arr[3] = color_arr[3], color_arr[1]
color_arr = colors_roller_2
# plot font size options
plt.rc("font", size=15)
plt.rc("axes", titlesize=30)
plt.rc("xtick", labelsize=18)
plt.rc("ytick", labelsize=18)
plt.rc("legend", fontsize=18)
plt.rc("hatch", color="white")
mpl.rcParams["hatch.linewidth"] = 1.8
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams.update({'font.size': 16})
mpl.rcParams.update({'font.family': 'serif'})

fig, ax0 = plt.subplots(1, 1, figsize=(15, 3))
ax1 = ax0.twinx()
plt.subplots_adjust(top = 1.02, bottom=0.01, hspace=0.6, wspace=0.20)


bar_width = 0.105
horiz_margin = 0.6
horiz_major_tick = 0.7

data_file = f"overall_traffic/traffic.txt"
traffic_names = [
    f"From SSD",
    f"To SSD",
    f"From CPU",
    f"To CPU",
]
figure_traffic_names = [
    f"GPU - SSD",
    f"GPU - Host Mem"
]

all_data_arr = np.zeros(0)
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
x_tick_array = np.zeros((len(sections) - 1, len(settings)))
for traffic_idx in range(len(traffic_names)):
  for section_idx, section in enumerate(sections[1:]):
    lines = section.strip().split("\n")
    workload = lines[0].strip()
    if traffic_idx == 0:
      workloads.append(workload)
    data_array[section_idx, :] = np.array([float(data) * 4096 / 1024 ** 3 for data in lines[1 + traffic_idx].split()])
    x_tick_array[section_idx, :] = section_idx * horiz_major_tick + (np.arange(len(settings)) - (len(settings) - 1) / 2) * bar_width
  if len(all_data_arr.shape) == 1:
    all_data_arr = np.zeros((*data_array.shape, 4))
  all_data_arr[:, :, traffic_idx] = data_array

sum_slice = np.sum(all_data_arr, axis=2)
all_data_arr[:, :, 0] = all_data_arr[:, :, 0] + all_data_arr[:, :, 1]
all_data_arr[:, :, 1] = all_data_arr[:, :, 2] + all_data_arr[:, :, 3]
for slice_idx, figure_traffic_name in enumerate(figure_traffic_names):
  if slice_idx == 0:
    bottom_slice = np.zeros(all_data_arr[:, :, 0].shape)
  else:
    bottom_slice = np.sum(all_data_arr[:, :, :slice_idx], axis=2)
  for j, workload in enumerate(workloads):
    if j < 2:
      ax0.bar(x_tick_array[j, :], all_data_arr[j, :, slice_idx], color=color_arr[slice_idx], bottom=bottom_slice[j, :], width=bar_width, edgecolor=hatch_color, hatch=hatch_arr[slice_idx], label=figure_traffic_name)
      ax0.bar(x_tick_array[j, :], all_data_arr[j, :, slice_idx], color="none", bottom=bottom_slice[j, :], width=bar_width, edgecolor="white", linewidth=0.8, label=figure_traffic_name)
    else:
      ax1.bar(x_tick_array[j, :], all_data_arr[j, :, slice_idx], color=color_arr[slice_idx], bottom=bottom_slice[j, :], width=bar_width, edgecolor=hatch_color, hatch=hatch_arr[slice_idx], label=figure_traffic_name)
      ax0.bar(x_tick_array[j, :], all_data_arr[j, :, slice_idx], color="none", bottom=bottom_slice[j, :], width=bar_width, edgecolor="white", linewidth=0.8)
  # for x_tick, data in zip(x_tick_array[:, 0], data_array[:, 0]):
  #   ax.text(x_tick, 1.05, f"{data:5.2f} kops/s", ha="center", va="bottom", rotation=90, fontsize=10)
ax0.set_xticks(np.arange(len(workloads)) * horiz_major_tick)
ax0.set_xticklabels([workload.replace("|", "\n") for workload in workloads])
ax0.set_xlim([-horiz_margin * horiz_major_tick, (len(workloads) - 1 + horiz_margin) * horiz_major_tick])
yticks = np.arange(0, 700, 200)
ax0.set_ylim([0, 700])
ax0.set_yticks(yticks)
ax0.set_yticklabels([str(y) for y in yticks])
ax0.set_ylabel("Traffic (GB)", fontsize=20)
hspace_multiplier = 0.025
ax0.vlines(x=horiz_major_tick * (1.5 - hspace_multiplier), ymin=ax0.get_ylim()[0], ymax=ax0.get_ylim()[1], colors="grey", linestyles="-", linewidth=3.5)
ax0.vlines(x=horiz_major_tick * (1.5 + hspace_multiplier), ymin=ax0.get_ylim()[0], ymax=ax0.get_ylim()[1], colors="grey", linestyles="-", linewidth=3.5)

rect = mpl.patches.Rectangle((401.7, 10), 317 * hspace_multiplier, 400, linewidth=1, edgecolor='white', facecolor='white', zorder=10)
fig.patches.extend([rect])
ax1.set_ylim([0, 2800])
ax1.set_yticks(yticks * 4)
ax1.set_yticklabels([str(y) for y in yticks * 4])
ax1.set_ylabel("Traffic (GB)", fontsize=20)
ax0.yaxis.grid()
# ax0.yaxis.set_label_coords(-0.05, 0.47)
  
handles, labels = ax0.get_legend_handles_labels()
handles, labels = handles[0:len(handles):len(settings)], labels[0:len(labels):len(settings)]
legend = ax0.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.74, 1.03), ncol=2)

bottom_slice = np.sum(all_data_arr[:, :, :2], axis=2)
arrow_width, arrow_head_width, arrow_fontsize = 2, 8, 18
y_base = 600
# ============
ax0.annotate(list(setting_translation.values())[0],
  xy=(bar_width * 1.5, 1), 
  xytext=(horiz_major_tick * 0 - bar_width * 1.5 - 0.1, y_base),
  arrowprops=dict(facecolor='black', shrink=0., width=arrow_width, headwidth=arrow_head_width, headlength=4, visible=False),
  fontsize=arrow_fontsize, linespacing=0,
  bbox=dict(boxstyle="square,pad=0", fc="w", ec="none", alpha=0.),
)
ax0.annotate("",
  xy=(horiz_major_tick * 0 - bar_width * 1.5, bottom_slice[0, 0]), 
  xytext=(horiz_major_tick * 0 - bar_width * 1.5, y_base),
  arrowprops=dict(facecolor='black', shrink=0., width=arrow_width, headwidth=arrow_head_width, headlength=4),
  fontsize=arrow_fontsize, linespacing=0,
  bbox=dict(boxstyle="square,pad=0", fc="w", ec="none", alpha=0.),
)
# ============
ax0.annotate(list(setting_translation.values())[1],
  xy=(horiz_major_tick * 0 - bar_width * 0.5, bottom_slice[0, 1]), 
  xytext=(horiz_major_tick * 0 - bar_width * 1.5 + 0.1, y_base - 95),
  arrowprops=dict(facecolor='black', shrink=0., width=arrow_width, headwidth=arrow_head_width, headlength=4, visible=False),
  fontsize=arrow_fontsize, linespacing=0,
  bbox=dict(boxstyle="square,pad=0", fc="w", ec="none", alpha=0.),
)
ax0.annotate("",
  xy=(horiz_major_tick * 0 - bar_width * 0.5, bottom_slice[0, 1]), 
  xytext=(horiz_major_tick * 0 - bar_width * 1.5 + 0.11215, y_base - 115),
  arrowprops=dict(facecolor='black', shrink=0., width=arrow_width, headwidth=arrow_head_width, headlength=4),
  fontsize=arrow_fontsize, linespacing=0,
  bbox=dict(boxstyle="square,pad=0", fc="w", ec="none", alpha=0.),
)
# ============
ax0.annotate(list(setting_translation.values())[2],
  xy=(horiz_major_tick * 0 + bar_width * 0.5, bottom_slice[0, 2]), 
  xytext=(horiz_major_tick * 0 - bar_width * 1.5 + 0.15, y_base - 200),
  arrowprops=dict(facecolor='black', shrink=0., width=arrow_width, headwidth=arrow_head_width, headlength=4, visible=False),
  fontsize=arrow_fontsize, linespacing=0,
  bbox=dict(boxstyle="square,pad=0", fc="w", ec="none", alpha=0.),
)
ax0.annotate("",
  xy=(horiz_major_tick * 0 + bar_width * 0.5, bottom_slice[0, 2]), 
  xytext=(horiz_major_tick * 0 - bar_width * 1.5 + 0.25, y_base - 220),
  arrowprops=dict(facecolor='black', shrink=0., width=arrow_width, headwidth=arrow_head_width, headlength=4),
  fontsize=arrow_fontsize, linespacing=0,
  bbox=dict(boxstyle="square,pad=0", fc="w", ec="none", alpha=0.),
)
# ============
ax0.annotate(list(setting_translation.values())[3],
  xy=(horiz_major_tick * 0 + bar_width * 1.5, bottom_slice[0, 3]), 
  xytext=(horiz_major_tick * 0 - bar_width * 1.5 + 0.33, y_base - 300),
  arrowprops=dict(facecolor='black', shrink=0., width=arrow_width, headwidth=arrow_head_width, headlength=4, visible=False),
  fontsize=arrow_fontsize, linespacing=0,
  bbox=dict(boxstyle="square,pad=0", fc="w", ec="none", alpha=0.),
)
ax0.annotate("",
  xy=(horiz_major_tick * 0 + bar_width * 1.5, bottom_slice[0, 3]), 
  xytext=(horiz_major_tick * 0 - bar_width * 1.5 + 0.36, y_base - 320),
  arrowprops=dict(facecolor='black', shrink=0., width=arrow_width, headwidth=arrow_head_width, headlength=4),
  fontsize=arrow_fontsize, linespacing=0,
  bbox=dict(boxstyle="square,pad=0", fc="w", ec="none", alpha=0.),
)

# shift = max([t.get_window_extent().width for t in legend.get_texts()])
# for t in legend.get_texts():
#     t.set_position(((shift - t.get_window_extent().width) / 2 - 6, 0))

plt.show()

extent = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).expanded(0.85, 1.45)
x_len, y_len = extent.x1 - extent.x0, extent.y1 - extent.y0
extent.y0 -= y_len * 0.04
extent.y1 -= y_len * 0.1
extent.x0 -= x_len * 0.015
extent.x1 += x_len * 0.055
# figname = list(net_name_translation.values())[model]
# fig.savefig(f"OverallPerf{figname}.png", bbox_inches=extent)
fig.savefig(f"output/OverallTraffic.png", bbox_inches=extent)
fig.savefig(f"output/OverallTraffic.pdf", bbox_inches=extent)
