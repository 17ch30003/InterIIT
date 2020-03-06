import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import json

path = "../data/competition_data/text_features_npy/"

dic_r = json.loads(open('vocab_r.json', 'r').read())
dic_f = json.loads(open('vocab_f.json', 'r').read())
print("dict read")
score = {}

for key in tqdm(dic_r.keys()):
	if key in dic_f:
		score[key] = max(dic_r[key]/dic_f[key], dic_f[key]/dic_r[key])
	else:
		score[key] = dic_r[key]

for key in tqdm(dic_f.keys()):
	if key not in score:
		score[key] = dic_f[key]


json = json.dumps(score)
file = open("score.json", 'w')
file.write(json)
file.close()
