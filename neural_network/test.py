import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
from data_loader import *
from model_nn import *
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)

torch.manual_seed(102)

net = NN()
print(net)
net.to(device)

params = net.state_dict()

# train_dataset = Loader('../data/', split="train")
# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

val_dataset = Loader('../data/', split="val")
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

loss_fn = RMSELoss()

def update_lr(optimizer, epoch):
	lr = 0
	if(epoch != 0):
		for param_group in optimizer.param_groups:
			param_group['lr'] *= 0.8
			lr = param_group['lr']
		print("LR Reduced to {}".format(lr))


resume_training = True
save_dir = 'weights/'
if(resume_training):
	net.load_state_dict(torch.load(save_dir + 'best_val.wts'))#chagefileloc
	print("weights Loaded")

file = open('nn.csv', 'w')
file.write('id,target\n')
file.close()

def filewrite(ids, target):
	file = open('nn.csv', 'a')
	file.write(ids)
	file.write(', ')
	file.write(str(target))
	file.write('\n')

net.eval()
with torch.no_grad():
	for i, data in enumerate(valloader):
		ids, market_feat, target = data
		market_feat, target = market_feat.to(device), target.to(device)
		out = net(market_feat)
		for j in range(out.shape[0]):
			filewrite(ids[j], out[j].item())
			print(i*64+j)
