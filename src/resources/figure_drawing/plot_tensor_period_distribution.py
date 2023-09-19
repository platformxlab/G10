import os
from typing import Tuple, Union

from matplotlib.ticker import LogLocator
from fig_common import *


Figure = plt.figure( figsize=(8, 8) )
PDF = PdfPages( "output/tensor_periods_distribution.pdf" )



def plot_cost_model(times, sizes, ax: plt.Axes, color_list: List[str], ylabel: bool = True, log_x: bool = True, log_y: bool = True, y_lim: Tuple[float, float] = None, plot_line_slope: float = 1717.986918):
    '''
    color_list[0] for T10, color_list[1:] for baseline_points;
    baseline_points: [poplib (mem, time), roller (mem, time)]
    '''

    
    data = np.array([times, sizes]).T

    traces = np.array(data)
    # ax.scatter(list(range(traces.shape[0])), traces[:, 0], color="lightgreen", marker="o", s=10, label="Measured")
    # ax.plot(list(range(traces.shape[0])), traces[:, 1], color="navy", label="Predicted")
    
    # plot y=x
    #ax.plot([0, 1e8], np.array([0, 1e8]) * plot_line_slope, color="grey", linestyle="--", linewidth=1, zorder=2)
    
    ax.scatter(traces[:, 0], traces[:, 1], color="navy", marker="o", s=10, zorder=3)

    ax.set_xlabel("Inactive Time ($\mu$s)")
    if ylabel:
        ax.set_ylabel("Size (byte)")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    # set xtick labels the same as ytick labels
    ax.xaxis.set_major_locator(LogLocator(10, subs=(1.0,), numticks=8))
    ax.yaxis.set_major_locator(LogLocator(10, subs=(1.0,), numticks=8))
    ax.xaxis.set_minor_locator(LogLocator(10, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=8))
    ax.yaxis.set_minor_locator(LogLocator(10, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=8))
    ax.grid(which="major", axis="both", linestyle="-", linewidth=0.5, color="grey", zorder=1)

    if y_lim:
        ax.set_ylim(y_lim[0] or ax.get_ylim()[0], y_lim[1] or ax.get_ylim()[1])
    
    ax.set_xlim(ax.get_xlim())



    
exec(open('../../../results/BERT_Base/128-prefetch_lru_TensorPeriodLog.py').read())
ax = Figure.add_subplot(221)
plot_cost_model(np.array(sd_time), sd_size, ax, ["forestgreen", "peru", "royalblue"], y_lim=(None, 4e8))
ax.text(0.45, -0.34, "(a) BERT-128", \
  horizontalalignment='center', verticalalignment='center', \
  transform=ax.transAxes)
ax.set_xlim(1, 1e7)

exec(open('../../../results/VIT/512-prefetch_lru_TensorPeriodLog.py').read())
ax = Figure.add_subplot(222)
plot_cost_model(np.array(sd_time), sd_size, ax, ["forestgreen", "peru", "royalblue"], ylabel=False, y_lim=(None, 4e8))
ax.text(0.45, -0.34, "(b) ViT-512", \
  horizontalalignment='center', verticalalignment='center', \
  transform=ax.transAxes)
ax.set_xlim(1, 4e6)

exec(open('../../../results/ResNet152/512-prefetch_lru_TensorPeriodLog.py').read())
ax = Figure.add_subplot(223)
plot_cost_model(np.array(sd_time), sd_size, ax, ["forestgreen", "peru", "royalblue"], y_lim=(None, 4e9))
ax.text(0.45, -0.34, "(c) ResNet152-512", \
  horizontalalignment='center', verticalalignment='center', \
  transform=ax.transAxes)
ax.set_xlim(1, 2e8)

exec(open('../../../results/Inceptionv3/512-prefetch_lru_TensorPeriodLog.py').read())
ax = Figure.add_subplot(224)
plot_cost_model(np.array(sd_time), sd_size, ax, ["forestgreen", "peru", "royalblue"], ylabel=False, y_lim=(None, 4e9))
ax.text(0.45, -0.34, "(d) Inceptionv3-512", \
  horizontalalignment='center', verticalalignment='center', \
  transform=ax.transAxes)
ax.set_xlim(1, 1e8)

Figure.tight_layout(pad=0.8)

PDF.savefig(Figure, bbox_inches='tight')
PDF.close()
