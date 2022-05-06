from asyncio.log import logger
import os
import time
import logging
import general as g
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image#,ImageColor
# from torch.utils.data import DataLoader
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity
# from augment_and_mix import augment_and_mix
from scipy.io import savemat,loadmat
from skimage.color import rgb2lab,rgb2ycbcr,rgb2luv,rgb2hsv
# from sklearn.decomposition import PCA #大数据的包实现主成分分析
# import joblib
import json
import shutil
from scipy.stats import pearsonr
import sys

def get_vec_sim(arrays1,arrays2):
    assert(2==len(arrays1.shape))
    assert(arrays1.shape==arrays2.shape)

    sims=np.zeros(arrays1.shape[0],dtype=np.float32)
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

def cvt_color(img):
    img=img.convert('RGB')
    img=np.array(img)
    # img=img/255.0
    # img=rgb2lab(img,'A')

    img=rgb2ycbcr(img)
    # img=rgb2hsv(img)
    
    img=img.transpose(2,0,1)
    return img/255.0

def compare_spectrum(dataset_setting_clean,dataset_setting_crupt,file_names,saved_dir):
    # 计算干净图像的频谱
    c,h,w=dataset_setting_clean.input_shape
    image_num=len(file_names)
    images_clean=np.zeros([image_num,c,h,w])
    for i,file_name in enumerate(file_names):
        image_tmp=Image.open(os.path.join(dataset_setting_clean.dataset_dir,'val',file_name)).resize([h,w])
        image_tmp=cvt_color(image_tmp)
        images_clean[i,...]=image_tmp#/255.0
    images_clean=g.img2dct(images_clean)

    # 计算损坏图像的频谱差
    levels=5
    corruptions=int(len(dataset_setting_crupt.dataset_dir)/levels)
    for i in range(corruptions):
        my_mat={}
        for level in range(levels):
            dataset_dir=dataset_setting_crupt.dataset_dir[i*levels+level]
            crupt_name='_'.join(dataset_dir.split('/')[-3:])
            images_crupt=np.zeros_like(images_clean)
            for j,file_name in enumerate(file_names):
                image_tmp=Image.open(os.path.join(dataset_dir,file_name)).resize([h,w])
                image_tmp=cvt_color(image_tmp)
                images_crupt[j,...]=image_tmp#/255.0
            images_crupt=g.img2dct(images_crupt)
            images_diff=np.mean(images_crupt-images_clean,axis=0)/(images_clean.mean(axis=0)+g.epsilon)

            for k in range(images_diff.shape[0]):
                my_mat[str(level+1)+'_c'+str(k)]=images_diff[k,...]
        savemat(os.path.join(saved_dir,crupt_name[:-2]+'.mat'),my_mat)
        logger.info('Finish {} with {}'.format(crupt_name[:-2],'spectrums'))

def get_features(model,images,batch_size):
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

def get_acc(model,images,labels,batch_size):
    image_num=images.shape[0]
    batch_num=int(np.ceil(image_num/batch_size))
    correct_num=0
    labels=torch.from_numpy(labels)

    with torch.no_grad():
        for batch_idx in range(batch_num):
            start_idx=batch_idx*batch_size
            end_idx=min((batch_idx+1)*batch_size,image_num)
            y = model(torch.from_numpy(images[start_idx:end_idx,...]).cuda())
            correct_num+=torch.sum(torch.argmax(y,dim=1).detach().cpu()==labels[start_idx:end_idx].reshape(-1)).item()
    acc=correct_num/image_num
    return acc

def compare_features(model,dataset_setting_clean,dataset_setting_crupt,file_names,saved_dir):
    # 设置
    c,h,w=data_setting_clean.input_shape
    image_num=len(file_names)

    # 计算干净图像的特征
    images_clean=np.zeros([image_num,c,h,w],dtype=np.float32)
    for i,file_name in enumerate(file_names):
        image_tmp=Image.open(os.path.join(dataset_setting_clean.dataset_dir,'val',file_name)).resize([h,w])
        image_tmp=image_tmp.convert('RGB')
        image_tmp=np.array(image_tmp,dtype=np.float32)
        image_tmp=image_tmp/255.0
        image_tmp=(image_tmp-dataset_setting_clean.mean)/dataset_setting_clean.std
        images_clean[i,...]=image_tmp.transpose(2,0,1)
    features_clean=get_features(model,images_clean,data_setting_clean.batch_size)
    logger.info('Finish {} with {}'.format('cleans','features'))

    # 计算损坏图像的频谱差
    levels=5
    corruptions=int(len(dataset_setting_crupt.dataset_dir)/levels)
    for i in range(corruptions):
        my_mat={}
        features_all=[]
        for level in range(levels):
            dataset_dir=dataset_setting_crupt.dataset_dir[i*levels+level]
            crupt_name='_'.join(dataset_dir.split('/')[-3:])
            if not os.path.exists(saved_dir):
                os.makedirs(saved_dir)
            images_crupt=np.zeros_like(images_clean)
            for j,file_name in enumerate(file_names):
                image_tmp=Image.open(os.path.join(dataset_dir,file_name)).resize([h,w])
                image_tmp=image_tmp.convert('RGB')
                image_tmp=np.array(image_tmp,dtype=np.float32)
                image_tmp=image_tmp/255.0
                image_tmp=(image_tmp-dataset_setting_crupt.mean)/dataset_setting_crupt.std
                images_crupt[j,...]=image_tmp.transpose(2,0,1)
            features_crupt=get_features(model,images_crupt,dataset_setting_crupt.batch_size)
            sims=get_vec_sim(features_crupt,features_clean)
            features_all.append(sims.reshape(-1,1))
        my_mat['feature']=np.hstack(features_all)
        savemat(os.path.join(saved_dir,crupt_name[:-2]+'.mat'),my_mat)
        logger.info('Finish {} with {}'.format(crupt_name[:-2],'features'))

def compare_acc(model,dataset_setting_clean,dataset_setting_crupt,file_names,labels,saved_dir):
    # 设置
    c,h,w=data_setting_clean.input_shape
    image_num=len(file_names)

    # 计算干净图像的特征
    images_clean=np.zeros([image_num,c,h,w],dtype=np.float32)
    for i,file_name in enumerate(file_names):
        image_tmp=Image.open(os.path.join(dataset_setting_clean.dataset_dir,'val',file_name)).resize([h,w])
        image_tmp=image_tmp.convert('RGB')
        image_tmp=np.array(image_tmp,dtype=np.float32)
        image_tmp=image_tmp/255.0
        image_tmp=(image_tmp-dataset_setting_clean.mean)/dataset_setting_clean.std
        images_clean[i,...]=image_tmp.transpose(2,0,1)
    acc_clean=get_acc(model,images_clean,labels,data_setting_clean.batch_size)
    logger.info('Finish {} with {}'.format('cleans','acc'))

    # 计算损坏图像的频谱差
    levels=5
    corruptions=int(len(dataset_setting_crupt.dataset_dir)/levels)
    for i in range(corruptions):
        my_mat={}
        accs=[]
        # accs.append(acc_clean)
        for level in range(levels):
            dataset_dir=dataset_setting_crupt.dataset_dir[i*levels+level]
            crupt_name='_'.join(dataset_dir.split('/')[-3:])
            if not os.path.exists(saved_dir):
                os.makedirs(saved_dir)
            images_crupt=np.zeros_like(images_clean)
            for j,file_name in enumerate(file_names):
                image_tmp=Image.open(os.path.join(dataset_dir,file_name)).resize([h,w])
                image_tmp=image_tmp.convert('RGB')
                image_tmp=np.array(image_tmp,dtype=np.float32)
                image_tmp=image_tmp/255.0
                image_tmp=(image_tmp-dataset_setting_crupt.mean)/dataset_setting_crupt.std
                images_crupt[j,...]=image_tmp.transpose(2,0,1)
            acc_crupt=get_acc(model,images_crupt,labels,dataset_setting_crupt.batch_size)
            accs.append(acc_crupt)
        my_mat['acc']=np.vstack(accs).reshape(1,-1)
        savemat(os.path.join(saved_dir,crupt_name[:-2]+'.mat'),my_mat)
        logger.info('Finish {} with {}'.format(crupt_name[:-2],'acc'))

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

def get_labels(file_names):
    labels=[]
    classes=json.load(open('imagenet_class_to_idx.json'))
    for filename in file_names:
        class_name=filename.split('/')[0]
        label=classes[class_name]
        labels.append(label)
    return np.vstack(labels)

def compare_correlation(saved_dir,corruption_types):
    saved_dir_tmp='/'.join(saved_dir.split('/')[:-1])
    accs=os.listdir(os.path.join(saved_dir_tmp,'acc'))
    features=os.listdir(os.path.join(saved_dir_tmp,'feature'))
    assert(len(accs)==len(features))
    assert(len(accs)==len(corruption_types)+1)
    cors=np.zeros(20)
    my_mat={}
    for i,corruption in enumerate(corruption_types):
        acc_mat=loadmat(os.path.join(saved_dir_tmp,'acc',corruption.replace('/','_'),'acc.mat'))['acc']
        feature_mat=loadmat(os.path.join(saved_dir_tmp,'feature',corruption.replace('/','_'),'feature.mat'))['feature']
        cor=pearsonr(acc_mat.reshape(-1),feature_mat.mean(axis=0))
        cors[i]=cor[0]
    my_mat['correlation']=cors.reshape(4,5)
    savemat(os.path.join(saved_dir,'correlation.mat'),my_mat)
    np.savetxt(os.path.join(saved_dir,'correlation.txt'),cors.reshape(4,5))


'''
设置
'''
job='correlation' # 'spectrum' or 'feature' or 'acc' or 'correlation'
model_name='resnet50_imagenet'
num_images=1000
os.environ['CUDA_VISIBLE_DEVICES']='0'

now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
saved_dir=os.path.join('./results',model_name+'_'+str(num_images),job)
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

data_setting_clean=g.dataset_setting(dataset_name)
data_setting_crupt=g.dataset_setting(dataset_name+'-c')

# output_names(data_setting_clean.dataset_dir+'val') #随机排序并输出文件名
images_all=open(os.path.join(data_setting_clean.dataset_dir, 'val.txt')).read().split('\n')
images_selected=np.random.choice(images_all,num_images)
labels_selected=get_labels(images_selected)

ckpt  = './models/'+model_name+'.pth.tar'
# ckpt  = './results/2022-04-04-23_20_35/checkpoint.pth.tar'
model=g.select_model(model_name, ckpt)

'''
输出频谱
'''
if 'spectrum' == job:
    compare_spectrum(data_setting_clean,data_setting_crupt,images_selected,saved_dir)
elif 'feature' == job:
    compare_features(model,data_setting_clean,data_setting_crupt,images_selected,saved_dir)
elif 'acc' == job:
    compare_acc(model,data_setting_clean,data_setting_crupt,images_selected,labels_selected,saved_dir)
elif 'correlation' == job:
    compare_correlation(saved_dir,data_setting_crupt.corruption_types)
else:
    raise Exception('job must be spectrum or feature or acc')
logger.info('Finish Calculating')

'''
重新整理以便于绘图
'''
if 'correlation' == job:
    sys.exit(0)

for file in os.listdir(saved_dir):
    if not '.mat' in file:
        continue
    folder_name=file[:-4]
    if os.path.exists(os.path.join(saved_dir,folder_name)):
        shutil.rmtree(os.path.join(saved_dir,folder_name))
    os.makedirs(os.path.join(saved_dir,folder_name))
    os.rename(os.path.join(saved_dir,file),os.path.join(saved_dir,folder_name,job+'.mat'))
