# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from statsFiguresUtil import *
from figureUtils import *

marker_arr = ["^", "o", "x", "s", "v"]
line_styles = ["-.", "--", ":", "-", "--"]
# hatch_color = "#78756e"
hatch_color = "white"

color_arr = colors_roller_3
# plot font size options
plt.rc("font", size=11)
plt.rc("xtick", labelsize=17)
plt.rc("ytick", labelsize=15)
plt.rc("legend", fontsize=18)
plt.rc("hatch", color="white")
mpl.rcParams["axes.labelsize"] = 18
mpl.rcParams["hatch.linewidth"] = 1.8
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams.update({'font.size': 16})
mpl.rcParams.update({'font.family': 'serif'})

fig, (((ax0, ax1, ax2, ax3, ax4))) = plt.subplots(1, 5, figsize=(27, 10))
plt.subplots_adjust(top=0.3, bottom=0.01, hspace=0.53, wspace=0.25)
axs = (ax0, ax1, ax2, ax3, ax4)

models = [BERT, VIT, INCEPTION, RESNET, SENET]
stat_prefix = "overall_batchsize"
for model_idx in range(len(models)):
  ax = axs[model_idx]
  model = models[model_idx]
  if model == INCEPTION:
    data_file = f"{stat_prefix}/inception.txt"
  elif model == RESNET:
    data_file = f"{stat_prefix}/resnet.txt"
  elif model == SENET:
    data_file = f"{stat_prefix}/senet.txt"
  elif model == BERT:
    data_file = f"{stat_prefix}/bert.txt"
  elif model == VIT:
    data_file = f"{stat_prefix}/vit.txt"
  elif model == RESNEXT:
    data_file = f"{stat_prefix}/resnext.txt"

  bar_width = 0.105
  horiz_margin = 0.6
  horiz_major_tick = 0.7
  try:
    with open(data_file, "r") as f:
      lines = f.read()
  except Exception as e:
    continue

  first_dim_arr = []
  second_dim_arr = []

  sections = lines.strip().split("\n\n")
  first_dim_arr = [setting_translation[setting.strip()] for setting in sections[0].split("|")]
  data_array = np.zeros((len(sections) - 1, len(first_dim_arr)))
  for section_idx, section in enumerate(sections[1:]):
    lines = section.strip().split("\n")
    second_dim = lines[0].strip()
    second_dim_arr.append(float(second_dim))
    data_array[section_idx, :] = np.array([second_dim_arr[-1] / (float(data) / 1200e6) for data in lines[1].split()])
  for j, setting in enumerate(first_dim_arr):
    plot_idxs = data_array[:, j] != 0
    x_arr = [second_dim_arr[i] for i in range(len(plot_idxs)) if plot_idxs[i]]
    y_arr = [data_array[i, j] for i in range(len(plot_idxs)) if plot_idxs[i]]
    ax.plot(x_arr, y_arr, linestyle=line_styles[j], color=color_arr[j], marker=marker_arr[j], markerfacecolor="none", label=first_dim_arr[j], linewidth=3, markersize=9)

  ax.set_xticks(second_dim_arr)
  ax.set_xticklabels([str(int(d)) for d in second_dim_arr])
  ax.set_xlabel(f"Batch Size")
  plt.text(0.5, -0.29, f"({chr(ord('a') + model_idx)}) {list(net_name_translation.values())[model]}", ha="center", transform=ax.transAxes)
  ax.set_ylim([0, (data_array[:, j]).max() * 1.2])
  if model == BERT:
    ax.set_yticks(np.arange(0, ax.get_ylim()[1], 15))
  elif model == VIT:
    ax.set_yticks(np.arange(0, ax.get_ylim()[1], 100))
  elif model == INCEPTION:
    ax.set_yticks(np.arange(0, ax.get_ylim()[1], 7))
  elif model == RESNET:
    ax.set_yticks(np.arange(0, ax.get_ylim()[1], 4))
  elif model == SENET:
    ax.set_yticks(np.arange(0, ax.get_ylim()[1], 2))
  # ax.ticklabel_format(axis='y', scilimits=(0, 2))
  # ax.yaxis.set_label_coords(-0.13, 0.47)
  ax.grid()
  
ax0.set_ylabel("Sequence / sec")
ax1.set_ylabel("Image / sec")
ax2.set_ylabel("Image / sec")
ax3.set_ylabel("Image / sec")
ax4.set_ylabel("Image / sec")
# ax0.yaxis.set_label_coords(-0.1, 0.47)

handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.367), ncol=5)
# shift = max([t.get_window_extent().width for t in legend.get_texts()])
# for t in legend.get_texts():
#     t.set_position(((shift - t.get_window_extent().width) / 2 - 6, 0))

plt.show()

extent = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).expanded(0.84, 0.85)
x_len, y_len = extent.x1 - extent.x0, extent.y1 - extent.y0
extent.y0 -= y_len * 0.188
extent.y1 -= y_len * 0.68
extent.x0 -= x_len * 0.00
extent.x1 -= x_len * 0.01
# figname = list(net_name_translation.values())[model]
# fig.savefig(f"OverallPerf{figname}.png", bbox_inches=extent)
fig.savefig(f"output/OverallPerfBatchSize.png", bbox_inches=extent)
fig.savefig(f"output/OverallPerfBatchSize.pdf", bbox_inches=extent)
