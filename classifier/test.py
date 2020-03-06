import torch
import random
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
from data_loader import *
from model_nn import *
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

torch.manual_seed(102)

net = NN()

net.to(device)

params = net.state_dict()

test_dataset = Testloader('../data/')
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=3, pin_memory=True)

train_dataset = Loader("../data/", split = "test")
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=3, pin_memory=True)

save_dir = 'weights/'
# net.load_state_dict(torch.load(save_dir + 'best_val_1.wts', map_location=torch.device('cpu')))
net.load_state_dict(torch.load(save_dir + 'trained_1.wts'))
print("weights Loaded")

file = open('classif_1.csv', 'w')
file.write('id,target\n')
file.close()



def filewrite(ids, target):
	file = open('classif_1.csv', 'a')
	file.write(ids)
	file.write(', ')
	if(target == 1):
		target = 1.001
		# target += random.randint(0, 98)/10000
	else:
		target = 0.999
		# target += 1 + random.randint(-98, 0)/10000		
	file.write(str(target))
	file.write('\n')

mean=1.0001464
std=0.020134468
net.eval()

corr = 0
incorr = 0

with torch.no_grad():
	for i, data in enumerate(testloader):
		ids, market_features= data
		market_features = market_features.to(device)
		out = net(market_features)
		for j in range(out.shape[0]):
			prob = out[j].item()
			if(prob >= 0.5):
				prob = 1
			else:
				prob = 0
			filewrite(ids[j], prob)
			print(i*64+j)