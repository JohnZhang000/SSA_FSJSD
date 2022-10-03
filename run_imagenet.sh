epoch=90
num_classes=100

cd others/AugMax-main/
python tiny_augmax_training_ddp.py --dataset IN --num_classes $num_classes --epochs $epoch --batch_size 128 --decay_epochs 30 60
cd ../../

cd others/APR-main/
python imagenet_my.py --data '/media/ubuntu204/F/Dataset/ILSVRC2012-'$num_classes -a resnet50 -j 16 --epochs $epoch -b 128 --aug none
python imagenet_my.py --data '/media/ubuntu204/F/Dataset/ILSVRC2012-'$num_classes -a resnet50 -j 16 --epochs $epoch -b 128 --aug aprs
cd ../../

cd others/ME-ADA-main/
python main_imagenet.py --num_classes $num_classes --epochs $epoch --num_workers 16  --algorithm MEADA
python main_imagenet.py --num_classes $num_classes --epochs $epoch --num_workers 16  --algorithm ADA
cd ../../

python imagenet_std.py -e $epoch --num_classes $num_classes -b 128
python imagenet_my.py -e $epoch --num_classes $num_classes -b 128