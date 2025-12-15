import sys
from mf_swarm_lib.utils.plotting import draw_cv_relevance

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python custom_statistics.py <full_swarm_exp_dir> <output_dir>")
        sys.exit(1)

    full_swarm_exp_dir = sys.argv[1]
    output_dir = sys.argv[2]

    draw_cv_relevance(full_swarm_exp_dir, output_dir)
    print(f"Statistics saved to {output_dir}/cv_relevance_boxplot.png")