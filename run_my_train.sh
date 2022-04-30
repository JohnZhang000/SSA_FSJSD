python cifar_my.py -m allconv
python cifar_my.py -m wrn
python cifar_my.py -m resnext -e 200
python cifar_my.py -m densenet -e 200 -wd 0.0001
python cifar_my.py -m allconv --no_fsim
python cifar_my.py -m allconv --no_topk
python cifar_my.py -m allconv --no_timei