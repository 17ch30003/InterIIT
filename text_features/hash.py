import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import json

path = "../data/competition_data/text_features_npy/"

def hash(vec):
	num = np.array([i for i in range(1, 301)], dtype = np.float)
	return str(round((vec*num).mean(), 7))

df = pd.read_csv(path + "../train.csv")

dic_r = {}
dic_f = {}

for i in tqdm(range(df.shape[0])):
	id_ = df.iloc[i]['id']
	target = df.iloc[i]['target']
	name = id_.replace('/', '_') + '.npy'
	data = np.load(path + name)
	if(target >= 1):
		for j in range(data.shape[0]):
			hash_val = hash(data[j])
			if hash_val in dic_r:
				dic_r[hash_val] += 1
			else:
				dic_r[hash_val] = 1
	else:
		for j in range(data.shape[0]):
			hash_val = hash(data[j])
			if hash_val in dic_f:
				dic_f[hash_val] += 1
			else:
				dic_f[hash_val] = 1

json_r = json.dumps(dic_r)
json_f = json.dumps(dic_f)

file_r = open("vocab_r.json", 'w')
file_r.write(json_r)
file_r.close()

file_f = open("vocab_f.json", 'w')
file_f.write(json_f)
file_f.close()