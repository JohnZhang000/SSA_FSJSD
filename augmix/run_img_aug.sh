clear
saved_dir_all='./snapshots/'
saved_dir=$saved_dir_all$(date +%Y%m%d_%H%M%S)$a
if [ ! -d "$saved_dir" ]; then
        mkdir $saved_dir
fi
#saved_dir='../saved_tests/20210128_231201_s'
log_name=$saved_dir'/terminal.log'

dataset_src='/media/ubuntu204/F/Dataset/ILSVRC2012/train'
#dataset_src='/media/ubuntu204/F/Dataset/cifar-10/train'

gpu_num=4
gpu_now=0
rpl0=0

fft_level=28
start_level=4
img_size=224

#fft_level=8
#start_level=4
#img_size=32
start_time=$(date +%Y%m%d_%H%M%S)$a

echo  'SUMMARY:ADVERSARIAL_TRAIN'             |tee $log_name
echo  'start_time:       '${start_time}       |tee -a $log_name
echo  'saved_dir:        '${saved_dir}        |tee -a $log_name
echo  'dataset_src:      '${dataset_src}      |tee -a $log_name
echo  'fft_level:        '${fft_level}        |tee -a $log_name
echo  'rpl0:             '${rpl0}             |tee -a $log_name
echo  'start_level:      '${start_level}      |tee -a $log_name
echo  'gpu_num:          '${gpu_num}          |tee -a $log_name
echo  'gpu_now:          '${gpu_now}          |tee -a $log_name
echo  'img_size:         '${img_size}         |tee -a $log_name
echo  'log_name:         '${log_name}         |tee -a $log_name

echo  ''                                                                                                 |tee -a $log_name
echo  '*************************************** train my model *************************************'     |tee -a $log_name


python ../train_code/batch_img_transform_cp_mgpu.py $fft_level $rpl0 $start_level $dataset_src $gpu_num $gpu_now $img_size |tee -a $log_name
        
end_time=$(date +%Y%m%d_%H%M%S)$a
echo  'end_time:       '${end_time}    |tee -a $log_name


