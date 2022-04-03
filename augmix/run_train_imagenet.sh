clear
saved_dir_all='./snapshots/'
saved_dir=$saved_dir_all$(date +%Y%m%d_%H%M%S)$a
if [ ! -d "$saved_dir" ]; then
        mkdir $saved_dir
fi
#saved_dir='../saved_tests/20210128_231201_s'
log_name=$saved_dir'/terminal.log'

dataset_imagenet='/media/ubuntu204/F/Dataset/ILSVRC2012-100'
dataset_imagenet_C='/media/ubuntu204/F/Dataset/ImageNet-C-100'

train_batch=(256)
model_used=(resnet18)
lrs=(0.1)
start_time=$(date +%Y%m%d_%H%M%S)$a

echo  'SUMMARY:ADVERSARIAL_TRAIN'          |tee $log_name
echo  'start_time:       '${start_time}    |tee -a $log_name
echo  'saved_dir:        '${saved_dir}     |tee -a $log_name
echo  'dataset_train:    '${dataset_imagenet} |tee -a $log_name
echo  'dataset_val:      '${dataset_imagenet_C}   |tee -a $log_name
echo  'log_name:         '${log_name}      |tee -a $log_name

echo  ''                                                                                                 |tee -a $log_name
echo  '*************************************** train my model *************************************'     |tee -a $log_name

for ((i=0;i<${#model_used[*]};i++))
do 
	model=${model_used[i]}
	model_dir_my=$saved_dir'/'$model
	if [ ! -d "$model_dir_my" ]; then
	    mkdir $model_dir_my
	fi
	batch_size_now=${train_batch[i]}
    
    echo  ''                                |tee -a $log_name
    echo  'model:            '${model}      |tee -a $log_name
    echo  'lr_now:           '${lrs[i]}      |tee -a $log_name
    echo  'batch_size:       '$batch_size_now      |tee -a $log_name
    python imagenet.py $dataset_imagenet $dataset_imagenet_C --epochs 180 -m $model -s $model_dir_my -lr ${lrs[i]} -b $batch_size_now |tee -a $log_name

    end_time=$(date +%Y%m%d_%H%M%S)$a
    echo  'end_time:       '${end_time}    |tee -a $log_name
done


