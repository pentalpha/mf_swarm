
mamba env crete -f mf_swarm_base.yml
conda run --live-stream -n mf_swarm_base pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python src/base_benchmark.py ~/data/dimension_db 1 ~/data/mf_datasets 30 0.1