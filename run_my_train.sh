#python imagenet_my.py -b 64
python imagenet_my.py -b 64 --with_NTXentLoss

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 imagenet_my.py -e 2 --with_NTXentLoss
