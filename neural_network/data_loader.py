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


class Loader(data.Dataset):
	def __init__(self, root, split="train"):
		self.root = root
		self.split = split
		# self.mean=1.0001464
		# self.std=0.020134468
		# self.mean = 0.0
		# self.std = 1.0

		if(split == "train"):
			self.ids = pd.read_csv('train.csv')['id'][0:3400]
			train = pd.read_csv("train.csv")
		else:
			self.ids = pd.read_csv('train.csv')['id'][3400:]
			train = pd.read_csv("train.csv")
		# train = train.fillna(train.mean())
		self.train = train
		self.text_feat = pd.read_csv("../lstm_new/sig_features_train.csv")

	def __len__(self):
		return len(self.ids)

	def __getitem__(self, index):
		Id = self.ids[index]
		row = self.train[self.train['id'] == Id]
		lbl = np.array(row['target'])
		# text_feat = np.array(self.text_feat[self.text_feat['id'] == Id].iloc[0, 1:])
		text_feat = np.array(self.text_feat.set_index('id').loc[Id])

		# lbl=(lbl-self.mean)/(self.std)
		# lbl = (lbl - self.min)/(self.max - self.min)
		span = np.array([row.iloc[0, 1]]).astype(float)
		market_features=np.array(row.iloc[0,3:]).astype(float)#added
		market_features = np.concatenate([span, market_features, text_feat])
		lbl = torch.from_numpy(lbl).float()
		market_features = torch.from_numpy(market_features).float()#added

		return Id, market_features, lbl


if __name__ == '__main__':
	train_dataset = Loader('../data/')
	testloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=3, pin_memory=True)
	print(len(testloader))

	for i, data in enumerate(testloader):
		seq, feat, lbl = data
		print(i)
		print(seq.shape)
		print(lbl.shape)
		print(feat.shape)
