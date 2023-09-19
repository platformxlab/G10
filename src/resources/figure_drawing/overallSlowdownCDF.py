# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import re

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from statsFiguresUtil import *
from figureUtils import *

colors = colors_roller_2
colors = colors_custom1
colors = colors_roller_2
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

model_names = ["BERT_Base", "VIT", "Inceptionv3", "ResNet152", "SENet154"]
model_display_names = ["BERT", "ViT", "Inceptionv3", "ResNet152", "SENet154"]
settings = ["lru", "FlashNeuron", "deepUM", "prefetch_lru"]

def get_kernel_time(model_name):
    filename = f"overall_slowdown_cdf/{model_name}-lru.txt"
    kernel_times = []
    with open(filename, "r") as f:
        for line in f:
            numbers = re.findall('\d+', line)
            if len(numbers) != 12 or int(numbers[0]) != 1:
                continue
            ideal_time = float(numbers[4])
            kernel_times.append(ideal_time)
    return np.array(kernel_times)


def parse(model_name, setting):
    filename = f"overall_slowdown_cdf/{model_name}-{setting}.txt"
    slowdown_list = []
    try:
        with open(filename, "r") as f:
            for line in f:
                if setting == "FlashNeuron":
                    slowdown_list.append(float(line.strip()) + 1)
                else:
                    numbers = re.findall('\d+', line)
                    if len(numbers) != 12 or int(numbers[0]) != 1:
                        continue
                    kernel_time = float(numbers[3]) - float(numbers[2])
                    ideal_time = float(numbers[4])
                    slowdown_list.append(float(kernel_time) / ideal_time)
    except Exception:
        pass
    if len(slowdown_list) == 0:
        return np.array([])
    return np.array(slowdown_list)


def plot_search_trace(lists, bname, ax, color_list,
                      linestyles = ["-.", "--", ":", "-"], 
                      labels = [], ylim = None, title = "",
                      bs_list = None, ytick_base = 1, ylabel = ""):
    '''
    color_list[0] for T10, color_list[1:] for baseline_points;
    baseline_points: [poplib (mem, time), roller (mem, time)]
    '''

    ymax = 0
    for idx, lst in enumerate(lists):
        speedup = np.sort(lst)
        if len(lst) != 0:
            ymax = max(np.max(lst[:-1]), ymax)
        ax.plot(1 - np.arange(len(speedup)) / (len(speedup)-1), speedup, color=color_list[idx], zorder=3, linestyle=linestyles[idx],  linewidth=5, label=setting_translation[bname[idx]])
    ax.invert_xaxis()
    # ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid(which="major", axis="both", linestyle="-", linewidth=0.5, color="grey", zorder=1)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(title)
    
    # x_labels = np.arange(0, 1.1, 0.25)
    # x_labels = [0.5, 0.9, 0.99, 0.999]
    x_labels = [0.6, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    x_labels_actual = [1 - x for x in x_labels]
    ax.set_xticks(x_labels_actual)
    ax.set_xticklabels([f"{x * 100:.3g}%" for x in x_labels])
    ax.set_xlim(1, 0.0005)
    ax.set_xlim(0.25, 0)
    print(ymax)
    if title.find("ResNet152") != -1:
        ax.set_ylim(0.9, ymax / 100)
    elif title.find("BERT") != -1:
        ax.set_ylim(0.9, ymax / 5)
    elif title.find("SENet") != -1:
        ax.set_ylim(0.9, ymax / 90)
    else:
        ax.set_ylim(0.9, ymax * 1.5)
    # ax.legend()

fig, (((ax0, ax1, ax2, ax3, ax4))) = plt.subplots(1, 5, figsize=(27, 10))
plt.subplots_adjust(top=0.3, bottom=0.01, hspace=0.53, wspace=0.25)
axs = (ax0, ax1, ax2, ax3, ax4)

for model_idx, model_name in enumerate(model_names):
    ax = axs[model_idx]
    lsts = []
    kernel_times = get_kernel_time(model_name)
    kernel_total_time = np.sum(kernel_times)
    for setting in settings:
        exe_times = np.array([])
        slowdowns = parse(model_name, setting)
        if len(slowdowns) != 0:
            exe_times = slowdowns * kernel_times / 1200
            exe_times = slowdowns
            # exe_times = np.array([exe_time for exe_time in exe_times if exe_time < kernel_total_time * 0.02])
        total_exe_time = np.sum(exe_times) * 1200
        lsts.append(exe_times)
    plot_search_trace(lsts, settings, ax, color_list=colors, labels=model_display_names, title=f"({chr(ord('a') + model_idx)}) {model_display_names[model_idx]}", ylabel="Slowdown" if model_idx == 0 else "")

handles, labels = ax0.get_legend_handles_labels()
fig.legend(labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.36), ncol=4, handlelength=3.5, handletextpad=0.55, columnspacing=1.8)

plt.show()

extent = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).expanded(0.85, 1.45)
x_len, y_len = extent.x1 - extent.x0, extent.y1 - extent.y0
extent.y0 += y_len * 0.11
extent.y1 -= y_len * 0.6
extent.x0 += x_len * 0.018
extent.x1 -= x_len * 0.02
# figname = list(net_name_translation.values())[model]
# fig.savefig(f"OverallPerf{figname}.png", bbox_inches=extent)
fig.savefig(f"output/KernelTimeCDF.png", bbox_inches=extent)
fig.savefig(f"output/KernelTimeCDF.pdf", bbox_inches=extent)
