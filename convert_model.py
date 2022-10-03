import torch
from torchvision import models

name='./others/AugMax-main/results/2022-05-09-00_47_23_tin_e200/latest.pth'
model_type='resnet50'

net = models.__dict__[model_type](num_classes=200)
net=net.cuda()
net=torch.nn.DataParallel(net)
checkpoint = torch.load(name)
# start_epoch = checkpoint['epoch'] + 1
# best_acc1 = checkpoint['best_acc1']
# net.load_state_dict(checkpoint['state_dict'])
net.load_state_dict(checkpoint['model'])
torch.save(net.state_dict(), name.replace('.tar', '_converted.pth.tar'))