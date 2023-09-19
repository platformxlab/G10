from typing import Dict, List

from matplotlib import pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter
import statistics


PROJECT_NAME = "G10"



# colors = ['#ff796c', '#fac205', '#95d0fc', '#96f97b']
# colors = ['lightgrey', '#96f97b', 'cyan', '#fc9096']
colors: List[str] = ["brown", "royalblue", "peru", "forestgreen"] # ['#ff796c', '#a4d46c', '#fca45c', '#95dbd0']
colors5: List[str] = colors + ["purple"]
color_platte_blue: List[str] = ["#bef7ff", "#b2ecff", "#a6e2ff", "#9ad7ff", "#8eccff", "#82c2ff", "#75b7ff", "#69acff", "#5da1ff", "#5197ff", "#458cff"]
color_platte_darkgreen: List[str] = ["#a7e1ab", "#8cd892", "#72cf78", "#57c65f", "#3fba48", "#36a03d", "#2d8533", "#246a29", "#1b501e", "#123514", "#0a1a0a"]

# "lightblue", "lightgreen", "darkblue", "green"

bar_hatches: List[str] = ['', '\\\\', '///', '-', '+', 'x', 'o', 'O', '.', '*']

legend_names: List[str] = ["PopART", "Ansor", "Roller", PROJECT_NAME]
legend_names_sc : List[str] = legend_names[:3] + [PROJECT_NAME + "-DataTransfer", PROJECT_NAME + "-Compute"]

# motivation_figsize = (9, 1.7)

BarWidth = 0.5
BarGap = 1.2

# matplotlib common settings
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'font.family': 'serif'})
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'
matplotlib.rcParams['hatch.linewidth'] = 0.7
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42