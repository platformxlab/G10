import enum
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
import pickle
import sys
from glob import glob
import networkx as nx
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LinearLocator
import matplotlib.backends.backend_pdf
import matplotlib
from networkx.drawing.nx_agraph import to_agraph 
import os
import pandas as pd
from PyPDF2 import PdfMerger
import glob
import math

TEXT_ONLY = False

plt.rcParams.update({'figure.max_open_warning': 0})
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'font.family': 'serif'})
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'
matplotlib.rcParams['hatch.linewidth'] = 0.5


line_styles = ['--', '-', '-.', ':']
hatches = ["", "\\", "//", "||"]
colors = ['#ff796c', 'plum', '#95d0fc', 'gray']
line_colors = ['brown', 'forestgreen', '#23a8eb', 'gray', 'black']
line_colors = ['brown', 'forestgreen', 'gray', '#23a8eb', 'black']
markers = ['.', '.', '*', 'v', '^']
    

def plot_timeline(ax: plt.Axes, results, filename, xlabel="Hours", ylabel="Migrate Overhead (hrs)", cumulative=False, step=False, aggregate=False, scaling=1.0, markevery=1, legend=False, yscale_log=True):
    if TEXT_ONLY:
        return
    
    # ax.figure(figsize=(8, 3), dpi=300)
    max_y_value = - float('inf')
    # values = np.arange()
    # plt.yticks(values * value_increment, ['%d' % val for val in values])
    if step:
        plot_func = ax.step
    else:
        plot_func = ax.plot
    for i, (plot_policy, series) in enumerate(results.items()):
        if cumulative:
            series = np.cumsum(series)
        seriess = np.array(series) * scaling
        max_y_value = max(max_y_value, max(seriess))
    if aggregate:
        agg_seriess = np.sum(series for _, series in results.items())
        agg_seriess = np.array(agg_seriess) * scaling
        max_y_value = max(max_y_value, max(agg_seriess))

    import pandas as pd

    for label in ["active", "input", "weight", "intermediate"]:
        if label in results.keys():
            results[label] = pd.Series(results[label]).rolling(6).max().dropna().tolist()
            print(label, max_y_value, max(results[label]))
    if "all" in results:
        results["all"] = pd.Series(results["all"]).rolling(6).max().dropna().tolist()

    # line_styles = ["-", "-", "-", "-"]
    line_styles = ["-", "--", "-.", ":"]
    for i, (plot_policy, series) in enumerate(results.items()):
        if cumulative:
            series = np.cumsum(series)
        series = np.array(series) * scaling / max_y_value
        plot_func(np.arange(len(series)), series, label=plot_policy, color=line_colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], linewidth=3, markevery=markevery)
    if aggregate:
        agg_series = np.sum(series for _, series in results.items())
        agg_series = np.array(agg_series) * scaling / max_y_value
        plot_func(np.arange(len(agg_series)), agg_series, label="total", color="purple", linewidth=2)
        
    if legend:
        ax.legend(ncol=4, fontsize=16, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.35))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if yscale_log:
        ax.set_yscale('log')
    else:
        ax.set_yticks([0, 0.5, 1])

    ax.set_xlim(0, max(len(s) for s in results.values()))

    ax.set_yticklabels([f'{i:0.0%}' for i in ax.get_yticks()])

    # plt.xlim([0, TIMESTEPS])
    if max_y_value != 0 and max_y_value != - float('inf'):
        ax.set_ylim(ax.get_ylim()[0], 1.15*max_y_value / max_y_value )
        #plt.locator_params(axis='y', nbins=5)
        # num_yticks = 5
        # # nearest_unit = 10**math.floor(math.log10(max_y_value // num_yticks))
        # ytick_gap = max_y_value*1.2 / num_yticks
        # plt.yticks(np.arange(num_yticks) * ytick_gap)
    # ax.set_ylim(0.002, 1.2)
    ax.grid(which='major', axis='y', color='#000000', linestyle='--')
    # ax.tight_layout()
    # print(ax.get_yticks(), [f'{i:0.0%}' for i in ax.get_yticks()])
    # ax.savefig(f"{filename}")
    # ax.clf()

def plot_multi_timeline(multi_results, filename, xlabel="Hours", ylabel="Migrate Overhead (hrs)", cumulative=False, step=False, aggregate=False):
    if TEXT_ONLY:
        return
    
    num_subplots = len(multi_results)
    if num_subplots == 0:
        return
    fig = plt.figure(figsize=(10, 4*num_subplots), dpi=300)
    axes = fig.subplots(nrows=num_subplots, ncols=1)
    max_y_value = - float('inf')
    for i, (top_label, results) in enumerate(multi_results.items()):
        if num_subplots == 1:
            ax = axes
        else:
            ax = axes[i]
        if step:
            plot_func = ax.step
        else:
            plot_func = ax.plot
        for i, (second_label, series) in enumerate(results.items()):
            if cumulative:
                series = np.cumsum(series)
            plot_func(np.arange(len(series)), series, label=second_label, color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], marker=markers[i % len(markers)])
            max_y_value = max(max_y_value, max(series))
        if aggregate:
            agg_series = np.sum(series for _, series in results.items())
            plot_func(np.arange(len(agg_series)), agg_series, label="total", color="purple", linewidth=2)
            max_y_value = max(max_y_value, max(agg_series))
        ax.legend(ncol=4, fontsize=12, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.2))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(top_label)
        ax.set_xlim([0, TIMESTEPS])
        if max_y_value != 0 and max_y_value != - float('inf'):
            ax.set_ylim([0, 1.2*max_y_value])
        ax.grid(b=True, which='major', axis='y', color='#000000', linestyle='--')
    
    plt.tight_layout()  
    plt.savefig(f"{filename}")
    plt.clf()

    

from fig_common import *

title = "dnn_mem_consumption_breakdown_live"
Figure = plt.figure(figsize=(8, 10))
PDF = PdfPages("output/" + title + ".pdf")

ACTIVE_NO_TOTAL = 0
ACTIVE_TOTAL = 1
ACTIVE_INPUT_TOTAL = 2
GLOBAL = 3

selection = GLOBAL

exec(open('../../../results/BERT_Base/128-prefetch_lru_NNMemConsumptionLog.py').read())
live = active
live_breakdown = active_breakdown
live_input = [item[0] for item in live_breakdown]
live_weight = [item[1] + 1 for item in live_breakdown]
live_intermediate = [item[2] for item in live_breakdown]
real = total
global_input = [input_size for _ in real]
global_weight = [global_weight for _ in real]
global_intermediate = [s - global_input[0] - global_weight[0] for s in real]
if selection == 0:
    motiv1 = {"weight" : live_weight, "intermediate" : live_intermediate}
elif selection == 1:
    motiv1 = {"all" : real, "weight" : live_weight, "intermediate" : live_intermediate}
elif selection == 2:
    motiv1 = {"all" : real, "input" : live_input, "weight" : live_weight, "intermediate" : live_intermediate}
elif selection == 3:
    motiv1 = {"all" : real, "input" : global_input, "weight" : global_weight, "intermediate" : global_intermediate}
ax = Figure.add_subplot(411)
plot_timeline(ax, motiv1, "mem_consumption_bert", "CUDA Kernel Index\n(a) BERT-128", " ", markevery=1, legend=True, yscale_log=False)



exec(open('../../../results/VIT/512-prefetch_lru_NNMemConsumptionLog.py').read())
live = active
live_breakdown = active_breakdown
live_input = [item[0] for item in live_breakdown]
live_weight = [item[1] + 1 for item in live_breakdown]
live_intermediate = [item[2] for item in live_breakdown]
real = total
global_input = [input_size for _ in real]
global_weight = [global_weight for _ in real]
global_intermediate = [s - global_input[0] - global_weight[0] for s in real]
if selection == 0:
    motiv1 = {"weight" : live_weight, "intermediate" : live_intermediate}
elif selection == 1:
    motiv1 = {"all" : real, "weight" : live_weight, "intermediate" : live_intermediate}
elif selection == 2:
    motiv1 = {"all" : real, "input" : live_input, "weight" : live_weight, "intermediate" : live_intermediate}
elif selection == 3:
    motiv1 = {"all" : real, "input" : global_input, "weight" : global_weight, "intermediate" : global_intermediate}
ax = Figure.add_subplot(412)
plot_timeline(ax, motiv1, "mem_consumption_vit", "CUDA Kernel Index\n(b) ViT-512", " ", markevery=1, yscale_log=False)



exec(open('../../../results/ResNet152/512-prefetch_lru_NNMemConsumptionLog.py').read())
live = active
live_breakdown = active_breakdown
live_input = [item[0] for item in live_breakdown]
live_weight = [item[1] + 1 for item in live_breakdown]
live_intermediate = [item[2] for item in live_breakdown]
real = total
global_input = [input_size for _ in real]
global_weight = [global_weight for _ in real]
global_intermediate = [s - global_input[0] - global_weight[0] for s in real]
if selection == 0:
    motiv1 = {"weight" : live_weight, "intermediate" : live_intermediate}
elif selection == 1:
    motiv1 = {"all" : real, "weight" : live_weight, "intermediate" : live_intermediate}
elif selection == 2:
    motiv1 = {"all" : real, "input" : live_input, "weight" : live_weight, "intermediate" : live_intermediate}
elif selection == 3:
    motiv1 = {"all" : real, "input" : global_input, "weight" : global_weight, "intermediate" : global_intermediate}
ax = Figure.add_subplot(413)
plot_timeline(ax, motiv1, "mem_consumption_resnet", "CUDA Kernel Index\n(c) ResNet152-512", " ", markevery=1, yscale_log=False)



exec(open('../../../results/Inceptionv3/512-prefetch_lru_NNMemConsumptionLog.py').read())
live = active
live_breakdown = active_breakdown
live_input = [item[0] for item in live_breakdown]
live_weight = [item[1] + 1 for item in live_breakdown]
live_intermediate = [item[2] for item in live_breakdown]
real = total
global_input = [input_size for _ in real]
global_weight = [global_weight for _ in real]
global_intermediate = [s - global_input[0] - global_weight[0] for s in real]
if selection == 0:
    motiv1 = {"weight" : live_weight, "intermediate" : live_intermediate}
elif selection == 1:
    motiv1 = {"all" : real, "weight" : live_weight, "intermediate" : live_intermediate}
elif selection == 2:
    motiv1 = {"all" : real, "input" : live_input, "weight" : live_weight, "intermediate" : live_intermediate}
elif selection == 3:
    motiv1 = {"all" : real, "input" : global_input, "weight" : global_weight, "intermediate" : global_intermediate}
ax = Figure.add_subplot(414)
plot_timeline(ax, motiv1, "mem_consumption_incept", "CUDA Kernel Index\n(d) Inceptionv3-512", " ", markevery=1, yscale_log=False)



Figure.text(-0.15, 3.2, "Memory Consumption (Live)", rotation=90, \
    horizontalalignment='center', verticalalignment='center', \
    transform=ax.transAxes)

Figure.tight_layout(pad=1.)

PDF.savefig(Figure, bbox_inches='tight')
PDF.close()


