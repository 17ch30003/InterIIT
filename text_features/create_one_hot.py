import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
from tqdm import tqdm

path = "../data/competition_data/text_features_npy/"

d = json.loads(open('final_idf.json', 'r').read())
print("dict read")
train = pd.read_csv("../data/competition_data/test.csv")['id']

dic = json.loads(open('IDF_dict.json', 'r').read())
Map = {}
count = 0
for k in dic.keys():
	l = len(dic[k])
	if(l > 430 and l < 520):
		Map[k] = count
		count += 1

print(count)
file = open('text_features_test_idf.csv', 'w')
file.write('id')
for i in range(count):
	file.write(','+str(i))
file.write('\n')

for id_ in tqdm(train):
	one_hot = np.zeros(count)
	data = np.load(path + id_.replace('/', '_') + ".npy")
	idf = d[id_.replace('/', '_')+'.npy']
	for i in range(data.shape[0]):
		if(idf[i] < 520 and idf[i] > 430):
			m = data[i].mean()
			m = round(m, 7)
			one_hot[Map[str(m)]] = 1
	file.write(id_)
	for i in range(one_hot.shape[0]):
		file.write(','+str(one_hot[i]))
	file.write('\n')

# for id_ in tqdm(train):
# 	data = np.load(path + id_.replace('/', '_') + ".npy")
# 	idf = d[id_.replace('/', '_')+'.npy']
# 	vec = []
# 	for i in range(data.shape[0]):
# 		if(idf[i] < 520 and idf[i] > 430):
# 			vec.append(data[i])
# 	vec = np.array(vec)
# 	vec = np.mean(vec, axis=0)
# 	file.write(id_)
# 	for i in vec:
# 		file.write(',' + str(i))
# 	file.write('\n')

# npys = os.listdir(path)
# for npy in tqdm(npys):
# 	file = open("mean/" + npy.replace('npy','txt'), 'w')
# 	data = np.load(path + npy)
# 	mean = data.mean(axis = 1)
# 	for m in mean:
# 		m = round(m, 7)
# 		if(str(m) in dictionary):
# 			idf = len(dictionary[str(m)])
# 		else:
# 			print("missing")
# 			idf = 1
# 		if npy not in d:
# 			d[npy] = []
# 			d[npy].append(idf)
# 		else:
# 			d[npy].append(idf)
