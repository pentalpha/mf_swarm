

import json
import sys
from os import path
from glob import glob
from multiprocessing import Pool
import matplotlib.pyplot as plt
from decimal import Decimal
import pandas as pd
from matplotlib.patches import ConnectionPatch, Rectangle
#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe
from mf_swarm_lib.core.node_factory import save_classifier_architecture
from mf_swarm_lib.utils.parsing import load_final_metrics, load_final_solutions
from mf_swarm_lib.utils.plotting import model_colors, plot_taxon_metrics
from mf_swarm_lib.utils.util_base import calc_n_params, plm_sizes
from tqdm import tqdm

#Equivalent to summarize_pairs_benchmark.py, but for taxa benchmark + the final table from pairs benchmark
