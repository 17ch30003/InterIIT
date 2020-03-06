import pandas as pd
import numpy as np
import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
from torch.utils import data
import re
train_data=pd.read_csv('/home/others/16EE10041/InterIIT/data/competition_data/train.csv')

import gzip

class Loader(data.Dataset):
	def __init__(self, root, split="train"):
		self.root = root
		self.split = split

		# self.mean = 0
		# self.std = 1
		# self.max = 1.0771028037
		# self.min = 0.9151968827

		if split == "train":
			self.ids = pd.read_csv(self.root+'competition_data/train.csv')['id'][0:]
		else:
			self.ids = pd.read_csv(self.root+'competition_data/train.csv')['id'][3400:]
		#elif split == "test":
			#self.ids = pd.read_csv(os.listdir(root+'test.csv'))['id']
		train = pd.read_csv(self.root + "competition_data/train.csv")
		# train = train.fillna(train.mean())
		self.train = train

	def __len__(self):
		return len(self.ids)

	def __getitem__(self, index):
		if(self.split == "train"):
			Id = self.ids[index]
		else:
			Id = self.ids[index + 3400]

		row = self.train[self.train['id'] == Id]
		lbl = np.array(row['target'])
		if(lbl >= 1):
			lbl = np.array([1])
		else:
			lbl = np.array([0])
		# lbl = (lbl - self.min)/(self.max - self.min)
		market_features=np.array(row.iloc[0,3:]).astype(float)#added

		lbl = torch.from_numpy(lbl).float()
		market_features = torch.from_numpy(market_features).float()#added

		return market_features, lbl

class Testloader(data.Dataset):
	def __init__(self, root):
		self.root = root
		self.ids = pd.read_csv(self.root+'competition_data/test.csv')['id']
		#elif split == "test":
			#self.ids = pd.read_csv(os.listdir(root+'test.csv'))['id']
		train = pd.read_csv(self.root + "competition_data/test.csv")
		# train = train.fillna(train.mean())
		self.train = train


	def __len__(self):
		return len(self.ids)

	def __getitem__(self, index):
		Id = self.ids[index]

		row = self.train[self.train['id'] == Id]

		market_features=np.array(row.iloc[0,2:]).astype(float)#added

		market_features = torch.from_numpy(market_features).float()#added

		return Id, market_features


if __name__ == '__main__':
	train_dataset = Loader('../data/')
	testloader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=3, pin_memory=True)
	print(len(testloader))

	for i, data in enumerate(testloader):
		seq, lbl = data
		print(seq.shape)
		print(lbl.shape)
		print(lbl)
		break
