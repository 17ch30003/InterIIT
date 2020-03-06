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
train_data=pd.read_csv('../data/competition_data/train.csv')

import gzip

class Loader(data.Dataset):
	def __init__(self, root, split="train"):
		self.root = root
		self.split = split
		self.mean=1.0001464
		self.std=0.020134468
		dict = {}
		with open("../dictionary.txt") as f:
			for line in f:
				(key, val) = line.split()
				dict[key] = int(val)
		self.dict = dict

		if split == "train":
			self.ids = pd.read_csv(self.root+'competition_data/train.csv')['id'][0:4000]
		else:
			self.ids = pd.read_csv(self.root+'competition_data/train.csv')['id'][4000:]
		#elif split == "test":
			#self.ids = pd.read_csv(os.listdir(root+'test.csv'))['id']
		self.train = pd.read_csv(self.root + "competition_data/train.csv")

	def __len__(self):
		return len(self.ids)

	def __getitem__(self, index):
		if(self.split == "train"):
			id = self.ids[index]
		else:
			id = self.ids[index + 4000]
		try:
			i=self.dict[id]
		except Exception as e:
			print(e)
			return torch.from_numpy(np.array([1])).float(), torch.from_numpy(np.array([1])).float(), torch.from_numpy(np.array([1])).float()
		"""chunk_data = pd.read_csv(self.root + "competition_data/text_features_/chunk_"+str(i)+".csv")
		required_data=chunk_data[chunk_data['id']==id]
		sequence = np.array(required_data.iloc[:,1:])"""
		l=re.sub('/','_',id)
		sequence=np.load(self.root + 'competition_data/text_features_npy/'+l+'.npy')
		train = self.train

		lbl = np.array(train[train['id']==id]['target'])
		lbl=(lbl-self.mean)/(self.std)
		train=train.fillna(train.mean())
		market_features=np.array(train[train['id']==id].iloc[0,3:]).astype(float)#added

		seq = torch.from_numpy(sequence).float()
		lbl = torch.from_numpy(lbl).float()
		market_features = torch.from_numpy(market_features).float()#added

		return seq, lbl, market_features

class Testloader(data.Dataset):
	def __init__(self, root, split="test"):
		self.root = root
		self.split = split
		dict = {}
		with open("../dictionary.txt") as f:
			for line in f:
				(key, val) = line.split()
				dict[key] = int(val)
		self.dict = dict
		self.ids = pd.read_csv(self.root + 'competition_data/test.csv')['id']
			

	def __len__(self):
		return len(self.ids)

	def __getitem__(self, index):
		id = self.ids[index]
		try:
			i=self.dict[id]
		except:
			return torch.from_numpy(np.array([1])).float(), torch.from_numpy(np.array([1])).float()
		"""chunk_data = pd.read_csv("data/competition_data/text_features_/chunk_"+str(i)+".csv")
		required_data=chunk_data[chunk_data['id']==id]
		sequence = np.array(required_data.iloc[:,1:])"""
		l=re.sub('/','_',id)
		sequence=np.load(self.root + 'competition_data/text_features_npy/'+l+'.npy')

		test = pd.read_csv(self.root + "competition_data/test.csv")
		market_features=np.array(test[test['id']==id].iloc[0,2:]).astype(float)#added

		seq = torch.from_numpy(sequence).float()
		market_features = torch.from_numpy(market_features).float()#added

		return id, seq, market_features


if __name__ == '__main__':
	train_dataset = Testloader('data/')
	testloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=3, pin_memory=True)
	print(len(testloader))

	for i, data in enumerate(testloader):
		seq, lbl = data
		print(seq)
		print(lbl)
		break
