# python tiny_imagenet_my.py -e 200 -nfs
# python tiny_imagenet_my.py -e 200

# cd others/AugMax-main/
# python tiny_augmax_training_ddp.py -e 90
# cd ../../

# cd others/APR-main/
# python tiny_imagenet.py --aug none
# python tiny_imagenet.py --aug aprs
# cd ../../

# cd others/ME-ADA-main/
# python main_tiny_imagenet.py --algorithm MEADA
# python main_tiny_imagenet.py --algorithm ADA
# cd ../../

# cd others/AugMax-main/
# python tiny_augmax_training_ddp.py -e 200
# cd ../../

# python tiny_imagenet_std.py -e 200

# python tiny_imagenet_my.py -e 2
# # python tiny_imagenet_my.py -e 1
# python tiny_imagenet_std.py -e 2
# cd others/APR-main/
# python tiny_imagenet.py --epochs 2 --aug none
# python tiny_imagenet.py --epochs 2 --aug aprs
# cd ../../
# cd others/ME-ADA-main/
# python main_tiny_imagenet.py --algorithm MEADA --epochs 2
# python main_tiny_imagenet.py --algorithm ADA --epochs 2
# cd ../../
# cd others/AugMax-main/
# python tiny_augmax_training_ddp.py --e 2

# python tiny_imagenet_my.py -e 200
# python tiny_imagenet_my.py -e 200
# python tiny_imagenet_my.py -e 200
# python tiny_imagenet_std.py -e 200
# python tiny_imagenet_std.py -e 200
python tiny_imagenet_my_SSA.py -e 200