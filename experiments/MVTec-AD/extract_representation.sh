export PYTHONPATH=../../:$PYTHONPATH
# CUDA_VISIBLE_DEVICES=$2
CUDA_VISIBLE_DEVICES=0
# # python -m torch.distributed.launch --nproc_per_node=$1 ../../tools/extract_representation.py
torchrun --nproc_per_node=1 ../../tools/extract_representation.py
# python ../../tools/extract_representation.py
