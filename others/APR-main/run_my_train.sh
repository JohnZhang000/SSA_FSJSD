export CUDA_VISIBLE_DEVICES=0
python main.py --model allconv --max-epoch 200
python main.py --model wide_resnet --max-epoch 200
python main.py --model resnext --max-epoch 200
python main.py --model densenet --max-epoch 200
python main.py --model allconv --max-epoch 200 --dataset cifar100
python main.py --model wide_resnet --max-epoch 200 --dataset cifar100
python main.py --model resnext --max-epoch 200 --dataset cifar100
python main.py --model densenet --max-epoch 200 --dataset cifar100