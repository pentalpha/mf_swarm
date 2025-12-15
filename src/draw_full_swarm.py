import sys
from mf_swarm_lib.utils.plotting import draw_swarm_panel

if __name__ == '__main__':
    full_swarm_exp_dir = sys.argv[1]
    plots_dir = full_swarm_exp_dir+'/plots'
    draw_swarm_panel(full_swarm_exp_dir, plots_dir)