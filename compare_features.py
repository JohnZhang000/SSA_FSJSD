from asyncio.log import logger
import os
import time
import logging
import general as g
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data import DataLoader
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity
from augment_and_mix import augment_and_mix

def get_vec_sim(arrays1,arrays2):
    assert(2==len(arrays1.shape))
    assert(arrays1.shape==arrays2.shape)

    sims=np.zeros(arrays1.shape[0],dtype=np.float)
    for i in range(arrays1.shape[0]):
        sims_tmp=cosine_similarity(np.expand_dims(arrays1[i,:],axis=0),np.expand_dims(arrays2[i,:],axis=0))
        sims[i]=sims_tmp[0][0]
    return sims

def get_hog(imgs):
    assert(4==len(imgs.shape))
    assert(imgs.shape[2]==imgs.shape[3])
    imgs_grey=imgs[:,0,:,:]*0.3+imgs[:,1,:,:]*0.59+imgs[:,2,:,:]*0.11

    features=[]
    for i in range(imgs_grey.shape[0]):
        feature_tmp = hog(imgs_grey[i,...],feature_vector=True) # return HOG map
        features.append(feature_tmp)
    features=np.vstack(features)
    return features

def get_spectrum(imgs):
    images_ycbcr=g.rgb_to_ycbcr(imgs)
    images_dct=g.img2dct(images_ycbcr)
    # images_dct=g.img2dct_4part(images_ycbcr)
    return images_dct

def compare_spectrum(dataset_setting_clean,dataset_setting_crupt,file_names,saved_dir):
    # 计算干净图像的频谱
    image_shape=dataset_setting_clean.input_shape
    image_num=len(file_names)
    images_clean=np.zeros([image_num,image_shape[0],image_shape[1],image_shape[2]])
    for i,file_name in enumerate(file_names):
        image_tmp=Image.open(os.path.join(dataset_setting_clean.dataset_dir,'val',file_name)).resize([image_shape[1],image_shape[2]])
        image_tmp=image_tmp.convert('RGB')
        image_tmp=np.array(image_tmp).transpose(2,0,1)
        images_clean[i,...]=image_tmp/255.0
    images_clean=get_spectrum(images_clean)

    # 计算损坏图像的频谱差
    for i,dataset_dir in enumerate(dataset_setting_crupt.dataset_dir):
        images_crupt=np.zeros_like(images_clean)
        saved_dir_tmp='_'.join(dataset_dir.split('/')[-3:])
        saved_dir_tmp=os.path.join(saved_dir,saved_dir_tmp)
        for i,file_name in enumerate(file_names):
            image_tmp=Image.open(os.path.join(dataset_dir,file_name)).resize([image_shape[1],image_shape[2]])
            image_tmp=image_tmp.convert('RGB')
            image_tmp=np.array(image_tmp).transpose(2,0,1)
            images_crupt[i,...]=image_tmp/255.0
        images_crupt=get_spectrum(images_crupt)
        images_diff=images_crupt-images_clean
        images_diff=np.mean(images_diff,axis=0)
        g.save_images_channel(saved_dir_tmp,images_diff)

def compare_features(compare_type,dataset_setting_clean,dataset_setting_crupt,file_names,saved_dir):
    # 计算干净图像的频谱
    image_shape=dataset_setting_clean.input_shape
    image_num=len(file_names)
    images_clean=np.zeros([image_num,image_shape[0],image_shape[1],image_shape[2]])
    for i,file_name in enumerate(file_names):
        image_tmp=Image.open(os.path.join(dataset_setting_clean.dataset_dir,'val',file_name)).resize([image_shape[1],image_shape[2]])
        image_tmp=image_tmp.convert('RGB')
        image_tmp=np.array(image_tmp).transpose(2,0,1)
        images_clean[i,...]=image_tmp/255.0
    if 'spectrum'==compare_type:
        features_clean=get_spectrum(images_clean)
    elif 'hog'==compare_type:
        features_clean=get_hog(images_clean)
    logger.info('Finish {} with {}'.format('cleans',compare_type))

    # 计算损坏图像的频谱差
    for i,dataset_dir in enumerate(dataset_setting_crupt.dataset_dir):
        images_crupt=np.zeros_like(images_clean)
        crupt_name='_'.join(dataset_dir.split('/')[-3:])
        saved_dir_tmp=os.path.join(saved_dir,crupt_name)
        if not os.path.exists(saved_dir_tmp):
            os.makedirs(saved_dir_tmp)
        for i,file_name in enumerate(file_names):
            image_tmp=Image.open(os.path.join(dataset_dir,file_name)).resize([image_shape[1],image_shape[2]])
            image_tmp=image_tmp.convert('RGB')
            image_tmp=np.array(image_tmp).transpose(2,0,1)
            images_crupt[i,...]=image_tmp/255.0
        if 'spectrum'==compare_type:
            features_crupt=get_spectrum(images_crupt)
            # 保存对比结果
            features_diff=features_crupt-features_clean
            features_diff=np.mean(features_diff,axis=0)
            g.save_images_channel(saved_dir_tmp,features_diff)

        elif 'hog'==compare_type:
            features_crupt=get_hog(images_crupt)
            sims=get_vec_sim(features_crupt,features_clean)
            np.savetxt(os.path.join(saved_dir_tmp,'hog.txt'),sims)
        logger.info('Finish {} with {}'.format(crupt_name,compare_type))

def get_features_cnn(model,images,batch_size):
    image_num=images.shape[0]
    batch_num=int(np.ceil(image_num/batch_size))
    features = []
    def hook(module, input, output): 
        features.append(input[0].clone().detach().cpu().numpy())

    handle=model.module.fc.register_forward_hook(hook)

    with torch.no_grad():
        for batch_idx in range(batch_num):
            start_idx=batch_idx*batch_size
            end_idx=min((batch_idx+1)*batch_size,image_num)
            y = model(torch.from_numpy(images[start_idx:end_idx,...]).cuda())
    features_clean=np.vstack(features)
    handle.remove() ## hook删除 
    return features_clean

def compare_features_cnn(model,dataset_setting_clean,dataset_setting_crupt,file_names,saved_dir):
    # 设置
    image_shape=data_setting_clean.input_shape
    image_num=len(file_names)

    # 计算干净图像的特征
    images_clean=np.zeros([image_num,image_shape[0],image_shape[1],image_shape[2]],dtype=np.float32)
    for i,file_name in enumerate(file_names):
        image_tmp=Image.open(os.path.join(dataset_setting_clean.dataset_dir,'val',file_name)).resize([image_shape[1],image_shape[2]])
        image_tmp=image_tmp.convert('RGB')
        image_tmp=np.array(image_tmp,dtype=np.float32)
        image_tmp=image_tmp/255.0
        image_tmp=(image_tmp-dataset_setting_clean.mean)/dataset_setting_clean.std
        images_clean[i,...]=image_tmp.transpose(2,0,1)
    features_clean=get_features_cnn(model,images_clean,data_setting_clean.batch_size)
    logger.info('Finish {} with {}'.format('cleans','cnn'))

    # 计算损坏图像的频谱差
    for i,dataset_dir in enumerate(dataset_setting_crupt.dataset_dir):
        images_crupt=np.zeros_like(images_clean)
        crupt_name='_'.join(dataset_dir.split('/')[-3:])
        # saved_dir_tmp=os.path.join(saved_dir,crupt_name)
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
        for i,file_name in enumerate(file_names):
            image_tmp=Image.open(os.path.join(dataset_dir,file_name)).resize([image_shape[1],image_shape[2]])
            image_tmp=image_tmp.convert('RGB')
            image_tmp=np.array(image_tmp,dtype=np.float32)
            image_tmp=image_tmp/255.0
            image_tmp=(image_tmp-dataset_setting_crupt.mean)/dataset_setting_crupt.std
            images_crupt[i,...]=image_tmp.transpose(2,0,1)
        features_crupt=get_features_cnn(model,images_crupt,dataset_setting_crupt.batch_size)
        sims=get_vec_sim(features_crupt,features_clean)
        np.savetxt(os.path.join(saved_dir,crupt_name+'_cnn_resnet.txt'),sims)
        logger.info('Finish {} with {}'.format(crupt_name,'cnn'))

def compare_features_augmix(dataset_setting_clean,severity,width,depth,alpha,file_names,saved_dir):
    # 计算干净图像的频谱
    image_shape=dataset_setting_clean.input_shape
    image_num=len(file_names)
    images_clean=np.zeros([image_num,image_shape[0],image_shape[1],image_shape[2]])
    for i,file_name in enumerate(tqdm(file_names)):
        image_tmp=Image.open(os.path.join(dataset_setting_clean.dataset_dir,'val',file_name)).resize([image_shape[1],image_shape[2]])
        image_tmp=image_tmp.convert('RGB')
        image_tmp=np.array(image_tmp).transpose(2,0,1)
        images_clean[i,...]=image_tmp/255.0
    features_clean=get_hog(images_clean)
    logger.info('Finish {}'.format('cleans'))

    images_crupt=np.zeros_like(images_clean)
    for i in tqdm(range(images_clean.shape[0])):
        image_tmp=images_clean[i,...].transpose(1,2,0)
        image_tmp=augment_and_mix(image_tmp,severity,width,depth,alpha)
        images_crupt[i,...]=image_tmp.transpose(2,0,1)
    features_crupt=get_hog(images_crupt)
    setting_name='s'+str(severity)+'_w'+str(width)+'_d'+str(depth)+'_a'+str(alpha)
    logger.info('Finish {}'.format(setting_name))

    sims=get_vec_sim(features_crupt,features_clean)
    saved_dir_tmp=os.path.join(saved_dir,'augmix')
    if not os.path.exists(saved_dir_tmp):
        os.makedirs(saved_dir_tmp)
    np.savetxt(os.path.join(saved_dir_tmp,setting_name+'.txt'),sims)





def output_names(dir_src):
    filenames=[]
    for root,dirs,files in os.walk(dir_src):
        if not files:
            continue
        sysn=root.split('/')[-1]
        for filename in files:
            if filename is None:
                print('filename is None')
            filenames.append(sysn+'/'+filename)
    pro_idx = np.random.permutation(len(filenames))
    file=open(dir_src+'.txt','w')
    for i in range(len(filenames)):
        file.write(filenames[pro_idx[i]]+'\n')
    file.close()




'''
设置
'''
model_name='resnet50_imagenet_augmix'
num_images=1000
os.environ['CUDA_VISIBLE_DEVICES']='2'

now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
saved_dir=os.path.join('./results',model_name+'_'+str(num_images),now)
if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)

'''
初始化日志系统
'''
set_level=logging.INFO
logger=logging.getLogger(name='r')
logger.setLevel(set_level)
formatter=logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s -%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

fh=logging.FileHandler(os.path.join(saved_dir,'log_mcts.log'))
fh.setLevel(set_level)
fh.setFormatter(formatter)

ch=logging.StreamHandler()
ch.setLevel(set_level)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)
g.setup_seed(0)

'''
初始化模型和数据集
'''
if 'imagenet' in model_name:
    dataset_name='imagenet'
elif 'mnist' in model_name:
    dataset_name='mnist'
else:
    dataset_name='cifar-10'

ckpt  = './models/'+model_name+'.pth.tar'
model=g.select_model(model_name, ckpt)
data_setting_clean=g.dataset_setting(dataset_name)
data_setting_crupt=g.dataset_setting(dataset_name+'-c')

# output_names(data_setting_clean.dataset_dir+'val') #随机排序并输出文件名
images_all=open(os.path.join(data_setting_clean.dataset_dir, 'val.txt')).read().split('\n')
images_selected=images_all[:num_images]

'''
输出频谱
'''
# compare_spectrum(data_setting_clean,data_setting_crupt,images_selected,saved_dir)
# compare_features('hog',data_setting_clean,data_setting_crupt,images_selected,saved_dir)
# compare_features_cnn(model,data_setting_clean,data_setting_crupt,images_selected,saved_dir)
severitys=[3]
widths=[3]
depths=[-1]
for severity in severitys:
    for width in widths:
        for depth in depths:
            compare_features_augmix(data_setting_clean,severity,width,depth,1,images_selected,saved_dir)
