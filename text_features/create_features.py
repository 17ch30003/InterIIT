import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
from tqdm import tqdm

def hash(vec):
	num = np.array([i for i in range(1, 301)], dtype = np.float)
	return str(round((vec*num).mean(), 7))

path = "../data/competition_data/text_features_npy/"
th = 13

score = json.loads(open('score.json', 'r').read())
score_filtered = {}
mapping = {}
counter = 0
for key in score.keys():
	if score[key] > th:
		score_filtered[key] = th
		mapping[key] = counter
		counter += 1

dim = counter

print("dict read")
split = "test"
train = pd.read_csv("../data/competition_data/" + split + ".csv")['id']

file = open("sig_features_" + split + ".csv", 'w')
file.write('id')
for i in range(dim):
	file.write(',' + str(i))
file.write('\n')

print("Dimension : {}".format(dim))

def filewrite(id_, vec):
	file.write(id_)
	for v in vec:
		file.write(','+str(v))
	file.write('\n')

for id_ in tqdm(train):
	data = np.load(path + id_.replace('/', '_') + ".npy")
	one_hot = np.zeros(dim, dtype = np.float)
	for i in range(data.shape[0]):
		hash_value = hash(data[i])
		if hash_value in mapping:
			one_hot[mapping[hash_value]] += 1.0

	filewrite(id_, one_hot)

file.close()