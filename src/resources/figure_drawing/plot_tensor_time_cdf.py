import os
from typing import Tuple

from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter, PercentFormatter
from fig_common import *


title0 = "output/tensor_size_cdf"
title1 = "output/tensor_time_cdf"

fig_size = (10,6)

#PDF0 = PdfPages(title0 + ".pdf")
PDF1 = PdfPages(title1 + ".pdf")



def plot_search_trace(list, bname: str, ax: plt.Axes, color_list: List[str] = color_platte_darkgreen[4:],
                      linestyles: List[str] = ["-", "--", ".", "-."], 
                      labels: List[str] = [], ylim: Tuple[float, float] = None, title: str = "",
                      bs_list: List[int] = None, ytick_base: int = 1, ylabel: str = ""):
    '''
    color_list[0] for T10, color_list[1:] for baseline_points;
    baseline_points: [poplib (mem, time), roller (mem, time)]
    '''

    speedup = np.sort(list)
    ax.plot(np.arange(len(speedup)) / (len(speedup)-1), speedup, color=color_list[0], zorder=3, linestyle=linestyles[0],  linewidth=4)
        # ax.scatter(np.indices(speedup.shape) / speedup.shape[0], speedup, color=color_list[i], marker="o", s=5, label=f"{bs}", zorder=3)
        # ax.scatter(np.indices(t10_times.shape) / t10_times.shape[0], np.sort(t10_times), color=color_list[i], marker="o", s=10, label=f"{bs}", zorder=3)
        # ax.scatter(np.indices(roller_times.shape) / roller_times.shape[0], np.sort(roller_times), color=color_list[i+1], marker="x", s=10, label=f"{bs}", zorder=3)
    ax.set_yscale("log")
    # plot line y=1
    #ax.plot(ax.get_xlim(), [1, 1], color="black", linestyle="--", linewidth=2, zorder=2, label="Roller")

    ax.grid(which="major", axis="both", linestyle="-", linewidth=0.5, color="grey", zorder=1)
    #ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    #ax.yaxis.set_major_formatter(ScalarFormatter())
    #ax.yaxis.set_major_locator(MultipleLocator(base=ytick_base))

    ax.set_ylabel(ylabel)
    ax.set_xlabel(title)
    
    ax.set_xlim(0, 1)
    ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]#np.arange(0, 1.1, 0.1)

    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x:0.0%}" if x in [0, 0.2, 0.4, 0.6, 0.8, 1] else "" for x in ticks], fontsize=16)
    upper = 10**7 if ax.get_ylim()[1]<10**7*2 else 10**8*2
    ut = 8 if ax.get_ylim()[1]<10**7*2 else 9
    ax.set_ylim(ax.get_ylim()[0], upper)
    ax.set_yticks([10**i for i in range(1, ut)])

    # ax.legend(prop={'size':12}, loc="upper left", ncol=1, labelspacing=0.4, columnspacing=1.2)



# Figure = plt.figure(figsize=fig_size)

# ax = Figure.add_subplot(221)
# plot_search_trace(bert_sd_size, "bert", ax, color_list = [colors[1], colors[3]], linestyles=["-", "--"], labels=["BS1"], title="(a) BERT-128", ylabel="Size (byte)")

# ax = Figure.add_subplot(222)
# plot_search_trace(vit_sd_size, "vit", ax, color_list = [colors[1], colors[3]], linestyles=["-", "--"], labels=["BS1"], title="(b) ViT-256")

# ax = Figure.add_subplot(223)
# plot_search_trace(resnet_sd_size, "resnet", ax, color_list = [colors[1], colors[3]], linestyles=["-", "--"], labels=["BS1"], title="(c) ResNet152-256", ylabel="Size (byte)")

# ax = Figure.add_subplot(224)
# plot_search_trace(in_sd_size, "inception", ax, color_list = [colors[1], colors[3]], linestyles=["-", "--"], labels=["BS1"], title="(d) Inceptionv3-512")

# Figure.text(0.5, 0, '% of Tensors', ha='center', va='center', fontsize=16)
# Figure.tight_layout(pad=1.05)


# PDF0.savefig(Figure, bbox_inches='tight')
# PDF0.close()

Figure = plt.figure(figsize=fig_size)

exec(open('../../../results/BERT_Base/128-prefetch_lru_TensorPeriodLog.py').read())
ax = Figure.add_subplot(221)
plot_search_trace(sd_time, "bert", ax, color_list = [colors[1], colors[3]], linestyles=["-", "--"], labels=["BS1"], title="(a) BERT-128", ylabel="Inactive Time ($\mu $s)")

exec(open('../../../results/VIT/512-prefetch_lru_TensorPeriodLog.py').read())
ax = Figure.add_subplot(222)
plot_search_trace(sd_time, "vit", ax, color_list = [colors[1], colors[3]], linestyles=["-", "--"], labels=["BS1"], title="(b) ViT-512")

exec(open('../../../results/ResNet152/512-prefetch_lru_TensorPeriodLog.py').read())
ax = Figure.add_subplot(223)
plot_search_trace(sd_time, "resnet", ax, color_list = [colors[1], colors[3]], linestyles=["-", "--"], labels=["BS1"], title="(c) ResNet152-512", ylabel="Inactive Time ($\mu $s)")

exec(open('../../../results/Inceptionv3/512-prefetch_lru_TensorPeriodLog.py').read())
ax = Figure.add_subplot(224)
plot_search_trace(sd_time, "inception", ax, color_list = [colors[1], colors[3]], linestyles=["-", "--"], labels=["BS1"], title="(d) Inceptionv3-512")

Figure.text(0.5, 0, '% of Tensor Inactive Periods', ha='center', va='center', fontsize=16)
Figure.tight_layout(pad=1.05)

PDF1.savefig(Figure, bbox_inches='tight')
PDF1.close()
