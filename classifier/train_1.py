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

train_dataset = Loader('../data/', split="train")
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

val_dataset = Loader('../data/', split="val")
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# loss_fn = RMSELoss()
loss_fn = nn.BCELoss()

def update_lr(optimizer, epoch):
	lr = 0
	if(epoch != 0):
		for param_group in optimizer.param_groups:
			param_group['lr'] *= 0.85
			lr = param_group['lr']
		print("LR Reduced to {}".format(lr))


resume_training = True
save_dir = 'weights/'
if(resume_training):
	net.load_state_dict(torch.load(save_dir + 'best_val.wts'))#chagefileloc
	print("weights Loaded")

best_acc = 0.87

No_Epoch = 1000
print("Total Epochs : {} Iterations per Epoch : {}".format(No_Epoch, len(trainloader)))

for epoch in range(No_Epoch):
	net.train()
	update_lr(optimizer, epoch)
	running_loss = 0
	avg_loss = deque(maxlen=100)
	for i, data in enumerate(trainloader):
		data, target = data
		data, target = data.to(device), target.to(device)

		out = net(data)
		#print(out)
		loss = loss_fn(out, target)
		avg_loss.append(loss.item())
		#print(market_features.view(-1).size())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		# if(i % 10 == 0):
		# print(out[0].item(), target[0].item())
		if(i % 10 == 0):
			print("Epoch : {} iter : {} train_loss : {:.3} Avg_Loss : {:.3} best_acc : {:.3}".format(epoch, i+1, loss.item(), np.mean(avg_loss), best_acc))
	train_loss = running_loss/len(trainloader)
	torch.save(net.state_dict(), save_dir + 'trained_1.wts')#chagefileloc
	net.eval()
	corr = 0
	incorr = 0
	with torch.no_grad():
		for i, data in enumerate(valloader):
			data, target = data
			data, target = data.to(device), target.to(device)
			out = net(data)
			loss = loss_fn(out, target)
			running_loss += loss.item()
			for i in range(out.shape[0]):
				prob = out[i].item()
				expected = target[i].item()
				if(prob >= 0.5 and expected == 1):
					corr += 1
				elif(prob < 0.5 and expected == 0):
					corr += 1
				else:
					incorr += 1

	acc = corr*1.0/(corr + incorr)
	if(best_acc < acc):
		best_acc = acc
		torch.save(net.state_dict(), save_dir + 'best_val.wts')#chagefileloc
		print("Saved best loss weights")

	print("Epoch : {} Acc : {:.3} Best_acc : {:.3}".format(epoch, acc, best_acc))
