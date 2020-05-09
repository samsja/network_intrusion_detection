import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# %%
def read_column_from_file(path):
    f = open(path, "r")
    colname = []
    buffer = f.readlines()
    f.close()
    for line in buffer:
        result = re.match("(.*):.*", line)  # 使用正则表达式筛选每一行的数据,自行查找正则表达式
        if result is not None:
            t = (result.group(1))  # group(1)将正则表达式的提取出来
            colname.append(t)
    colname.append('name_of_attack')
    print(colname)
    print("Nombre de colonne : {}".format(len(colname)))
    return colname

def read_lab_from_file(path):
    f = open(path, "r")
    attack_type = {}
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
    return attack_type
# %%
# normalisation
def normalize_features(train, test):
    x_train = train.copy()
    x_test = test.copy()
    # extraire the numerical columns
    cols_num = x_train.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    x_train[cols_num] = scaler.fit_transform(x_train[cols_num])
    x_test[cols_num] = scaler.transform(x_test[cols_num])

    return x_train, x_test

# %%
def standardize_features(train, test):
    x_train = train.copy()
    x_test = test.copy()
    # extraire the numerical columns
    cols_num = x_train.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    x_train[cols_num] = scaler.fit_transform(x_train[cols_num])
    x_test[cols_num] = scaler.transform(x_test[cols_num])

    return x_train, x_test

# %%
# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if 1 < nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if not np.issubdtype(type(columnDf.iloc[0]), np.number):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

# %%
def plot_correlation_matrix(df, graphWidth):
    df = df.dropna('columns')  # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80)
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Matrice de Corrélation ', fontsize=15)
    plt.show()


