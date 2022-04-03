clear
saved_dir='snapshots'
log_name=$saved_dir'/terminal_tsf1.log'

rpl_level=(5)
rpl0=(0)
model_type=(wrn allconv densenet resnext)

echo  ''                                     |tee $log_name
    
for ((i=0;i<${#model_type[*]};i++))
do 
    for ((j=0;j<${#rpl_level[*]};j++))
    do
        checkpts[j]='snapshots/'${model_type[i]}'/L'${rpl0[j]}'H'${rpl_level[j]}'.pth.tar'
    done
          
    #checkpts[0+${#rpl_level[*]}]='snapshots/'${model_type[i]}'/augmix.pth.tar'
    #checkpts[1+${#rpl_level[*]}]='snapshots/'${model_type[i]}'/deepaugment.pth.tar'
    #checkpts[2+${#rpl_level[*]}]='snapshots/'${model_type[i]}'/deepaugment_augmix.pth.tar'
    #checkpts[3+${#rpl_level[*]}]='snapshots/'${model_type[i]}'/ANT.pth.tar'
    #checkpts[4+${#rpl_level[*]}]='snapshots/'${model_type[i]}'/ANT_SIN.pth.tar'
    #checkpts[3+${#rpl_level[*]}]='snapshots/'${model_type[i]}'/vanilla.pth.tar'

    
    echo  ''                                     |tee -a $log_name
    echo  'model:            '${model_type[i]}   |tee -a $log_name
    echo  'checkpt:          '${#checkpts[*]}    |tee -a $log_name

    
    for ((j=0;j<${#checkpts[*]};j++))
    do 
        echo  ''                                    |tee -a $log_name
        echo  'checkpt:          '${checkpts[j]}    |tee -a $log_name
        python cifar_vanilla.py -m ${model_type[i]} --resume ${checkpts[j]} --evaluate      |tee -a $log_name
    done
done