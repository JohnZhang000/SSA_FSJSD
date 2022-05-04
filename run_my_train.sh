export CUDA_VISIBLE_DEVICES=2
# python cifar_my.py -m allconv
# python cifar_my.py -m allconv
# python cifar_my.py -m allconv --no_fsim
# python cifar_my.py -m allconv --no_fsim
# python cifar_my.py -m wrn
# python cifar_my.py -m wrn --no_fsim
# python cifar_my.py -m resnext -e 200
# python cifar_my.py -m densenet -e 200 -wd 0.0001
# python cifar_my.py -m allconv --dataset cifar100
python cifar_my.py -m wrn --dataset cifar100
# python cifar_my.py -m resnext -e 200 --dataset cifar100
python cifar_my.py -m densenet -e 200 -wd 0.0001 --dataset cifar100
# python cifar_my.py -m allconv --contrast_scale 0.5
# python cifar_my.py -m allconv --contrast_scale 1.0
# python cifar_my.py -m allconv --contrast_scale 1.5
# python cifar_my.py -m allconv --contrast_scale 2.0
# python cifar_my.py -m allconv --imp_thresh 0.25
# python cifar_my.py -m allconv --imp_thresh 0.5
# python cifar_my.py -m allconv --imp_thresh 0.75
# python cifar_my.py -m allconv --imp_thresh 1.0
# python cifar_my.py -m allconv --noise_scale 0.25
# python cifar_my.py -m allconv --noise_scale 0.5
# python cifar_my.py -m allconv --noise_scale 0.75
# python cifar_my.py -m allconv --noise_scale 1.0
# python cifar_my.py -m allconv --alpha 1.0
# python cifar_my.py -m allconv --alpha 3.0
# python cifar_my.py -m allconv --alpha 5.0
# python cifar_my.py -m allconv --alpha 7.0
# python cifar_my.py -m allconv --alpha 9.0