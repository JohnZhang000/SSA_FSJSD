# python cifar_my.py -m allconv -nfs --mixture-width 3 --mixture-depth 4 --contrast_scale 0.5
# python cifar_my.py -m allconv -nfs --mixture-width 3 --mixture-depth 4 --contrast_scale 1.0
# python cifar_my.py -m allconv -nfs --mixture-width 3 --mixture-depth 4 --contrast_scale 1.5
# python cifar_my.py -m allconv -nfs --mixture-width 3 --mixture-depth 4 --contrast_scale 2.0

# python cifar_my_SSA.py -m allconv --mixture-width 3 --mixture-depth 4


# export CUDA_VISIBLE_DEVICES=2
# python cifar_my.py -m allconv -nfs --mixture-width 5 --aug-std
# python cifar_my.py -m allconv -nfs --mixture-width 7 --aug-std
# python cifar_my.py -m allconv -nfs --mixture-width 9 --aug-std
# python cifar_my.py -m allconv -nfs --mixture-width 11 --aug-std
# python cifar_my.py -m allconv -nfs --mixture-depth 6 --aug-std
# python cifar_my.py -m allconv -nfs --mixture-depth 8 --aug-std
# python cifar_my.py -m allconv -nfs --mixture-depth 10 --aug-std
# python cifar_my.py -m allconv -nfs --mixture-depth 12 --aug-std
# python cifar_my.py -m allconv -nfs --mixture-width 5
# python cifar_my.py -m allconv -nfs --mixture-width 7
# python cifar_my.py -m allconv -nfs --mixture-width 9
# python cifar_my.py -m allconv -nfs --mixture-width 11
# python cifar_my.py -m allconv -nfs --mixture-depth 6
# python cifar_my.py -m allconv -nfs --mixture-depth 8
# python cifar_my.py -m allconv -nfs --mixture-depth 10
# python cifar_my.py -m allconv -nfs --mixture-depth 12
# python cifar_my.py -m allconv --mixture-depth 6
# python cifar_my.py -m allconv --mixture-depth 8
# python cifar_my.py -m allconv --mixture-depth 10
# python cifar_my.py -m allconv --mixture-depth 12
# python cifar_my.py -m allconv -nfs --mixture-width 3
# python cifar_my.py -m allconv -nfs --mixture-width 1
# python cifar_my.py -m allconv --mixture-width 1

# python cifar_my.py -m allconv --mixture-width 1 --mixture-depth 12 --seed 271828
# python cifar_my_loss1.py -m allconv --mixture-width 1 --mixture-depth 10 --seed 271828 -nfs
# python cifar_my.py -m allconv --mixture-width 3 --mixture-depth 4 --aug-std
# python cifar_my_SSA.py -m allconv --alpha 1.0
# python cifar_my_SSA.py -m allconv --alpha 3.0
# python cifar_my_SSA.py -m allconv --alpha 5.0
# python cifar_my_SSA.py -m allconv --alpha 7.0
# python cifar_my_SSA.py -m allconv --alpha 9.0


# python cifar_my.py -m allconv --seed 42
# python cifar_my.py -m allconv --seed 269733094
# python cifar_my.py -m allconv --seed 271828
# python cifar_my.py -m allconv --no_fsim
# python cifar_my.py -m allconv --no_fsim
#python cifar_my_SSA.py -m wrn
#python cifar_my_SSA.py -m resnext -e 200
#python cifar_my_SSA.py -m densenet -e 200 -wd 0.0001
#python cifar_my_SSA.py -m allconv --dataset cifar100
#python cifar_my_SSA.py -m wrn --dataset cifar100
#python cifar_my_SSA.py -m resnext -e 200 --dataset cifar100
#python cifar_my_SSA.py -m densenet -e 200 -wd 0.0001 --dataset cifar100

python cifar_my_SSA.py -m allconv --dataset cifar100 -r ./results/2022-08-28-05_39_08_allconv_cifar100/checkpoint.pth.tar --evaluate
python cifar_my_SSA.py -m wrn --dataset cifar100 -r ./results/2022-08-28-06_46_19_wrn_cifar100/checkpoint.pth.tar --evaluate
python cifar_my_SSA.py -m resnext -e 200 --dataset cifar100 -r ./results/2022-08-28-08_58_30_resnext_cifar100/checkpoint.pth.tar --evaluate
#python cifar_my_SSA.py -m densenet -e 200 -wd 0.0001 --dataset cifar100

# python cifar_my_SSA.py -m allconv --mixture-width 1 --mixture-depth 10

#python cifar_my_SSA.py -m allconv --mixture-width 3 --mixture-depth 4
#python cifar_my_SSA.py -m allconv --mixture-width 3 --mixture-depth 6
#python cifar_my_SSA.py -m allconv --mixture-width 3 --mixture-depth 8
#python cifar_my_SSA.py -m allconv --mixture-width 3 --mixture-depth 10
#python cifar_my_SSA.py -m allconv --mixture-width 3 --mixture-depth 12

#python cifar_my_SSA.py -m allconv --mixture-width 1 --mixture-depth 8
#python cifar_my_SSA.py -m allconv --mixture-width 5 --mixture-depth 8
#python cifar_my_SSA.py -m allconv --mixture-width 7 --mixture-depth 8


#python cifar_my_SSA.py -m allconv --mixture-width 1 --mixture-depth 8 --contrast_scale 0.5
#python cifar_my_SSA.py -m allconv --mixture-width 1 --mixture-depth 8 --contrast_scale 1.0
#python cifar_my_SSA.py -m allconv --mixture-width 1 --mixture-depth 8 --contrast_scale 1.5
#python cifar_my_SSA.py -m allconv --mixture-width 1 --mixture-depth 8 --contrast_scale 2.0
#python cifar_my_SSA.py -m allconv --mixture-width 1 --mixture-depth 8 --contrast_scale 2.5
#python cifar_my_SSA.py -m allconv --mixture-width 1 --mixture-depth 8 --contrast_scale 2.0 --imp_thresh 0.25
#python cifar_my_SSA.py -m allconv --mixture-width 1 --mixture-depth 8 --contrast_scale 2.0 --imp_thresh 0.5
#python cifar_my_SSA.py -m allconv --mixture-width 1 --mixture-depth 8 --contrast_scale 2.0 --imp_thresh 0.75
#python cifar_my_SSA.py -m allconv --mixture-width 1 --mixture-depth 8 --contrast_scale 2.0 --noise_scale 0.25
#python cifar_my_SSA.py -m allconv --mixture-width 1 --mixture-depth 8 --contrast_scale 2.0 --noise_scale 0.5
#python cifar_my_SSA.py -m allconv --mixture-width 1 --mixture-depth 8 --contrast_scale 2.0 --noise_scale 0.75
