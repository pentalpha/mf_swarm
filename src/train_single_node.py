import sys
import os

# Add the directory containing mf_swarm_lib to the python path (src/)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from mf_swarm_lib.training.train_single_node import training_process

if __name__ == "__main__":
    print(sys.argv)

    params_json_path = sys.argv[1]
    results_json_path = sys.argv[2]
    training_process(params_json_path, results_json_path)
