#%%
import pandas as pd
import numpy as np
import re
#%% 1. Ajouter les noms du colonnes sur les données
colname = []
f = open("./data/kddcup.names", "r")
buffer = f.readlines()
f.close()
for line in buffer:
    result = re.match("(.*):.*", line)  # 使用正则表达式筛选每一行的数据,自行查找正则表达式
    if result is not None:
        t = (result.group(1))  # group(1)将正则表达式的提取出来
        colname.append(t)

colname.append('attack')
print(colname)
print("Nombre de colonne : {}".format(len(colname)))


#%% 2. Affecter les catégories sur les types attack différents
attack_type = {}
f = open("./data/training_attack_types", "r")
buffer = f.readlines()
f.close()
for line in buffer:
    line = line.strip()
    if not len(line):
        continue
    attack_type[line.split(' ')[0]] = line.split(' ')[1]
    f.close()

attack_type['normal'] = 'normal'
print(attack_type)

#%% 3. Récupérer les données
data = pd.read_csv("./data/kddcup.data_10_percent.csv", sep=",", names=colname)
data['category'] = data.attack.apply(lambda r: attack_type[r[:-1]])
data.info()
print("Total data has {} rows & {} columns".format(data.shape[0], data.shape[1]))
data.describe()

#%% 4. Analyse de catégories
print(data.category.value_counts())
print(data.attack.unique())

#%% 5. Néttoyage
print(data['num_outbound_cmds'].unique())


