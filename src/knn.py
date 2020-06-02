# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from src.pretraitement_fonc import pretraitement
from sklearn.decomposition import PCA
from src.pretraitement_fonc import data_formated
from src.pretraitement_fonc import remove_variable
from src.pretraitement_fonc import one_hot_encoding
from src.pretraitement_fonc import seperate_train_test
from src.utils import standardize_features

from src.utils import add_decision_boundary
from src.utils import read_column_from_file
from src.utils import read_lab_from_file
from src.utils import plot_clustering

import src.knn_cross_validation as kcv
import src.knn_validation as kv
import src.nearest_prototypes as nrp

path = "./data/kddcup.names"
col_name = read_column_from_file(path)

path = "./data/training_attack_types"
attack_type = read_lab_from_file(path)

data = pd.read_csv("./data/kddcup.data_10_percent.csv", sep=",", names=col_name)
testdata = pd.read_csv("./data/corrected.csv", names=col_name)

data, testdata = data_formated(attack_type, data, testdata)
data, testdata = remove_variable(data, testdata)
data, testdata = one_hot_encoding(data, testdata)
X_train, Y_train, X_test, Y_test = seperate_train_test(data)

X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)

# %%
X_train_10, Y_train_, X_test, Y_test = seperate_train_test(data)



# PCA


cls = PCA(n_components=5)
pcs_X_train = cls.fit_transform(X_train_scaled)
cls.explained_variance_ratio_
df_X_train = pd.DataFrame(pcs_X_train, columns=[f"PC{i}" for i in range(1, 11)])
sns.scatterplot(x="PC1", y="PC2", hue=Y_train, data=df_X_train)
plt.show()
plt.bar(["Axe 1", "Axe 2", "Axe 3", "Axe 4", "Axe 5","Axe 6", "Axe 7", "Axe 8", "Axe 9", "Axe 10"], cls.explained_variance_ratio_)
plt.show()

cls = KNeighborsClassifier(n_neighbors=1)
cls.fit(X_train_scaled, Y_train)
pred = cls.predict(X_test_scaled)
# Calcul de la précision avec `accuracy_score`
acc = accuracy_score(pred, Y_test)

n_neighbors_list = np.unique(np.round(np.geomspace(1, 500, 100)).astype(int))
X_train_partiel = X_train_scaled.iloc[1:5000]
Y_train_partiel = Y_train.iloc[1:5000]
X_test_partiel = X_test_scaled.iloc[1:1500]
Y_test_partiel = Y_test.iloc[1:1500]

gen = kv.knn_simple_validation(X_train_partiel, Y_train_partiel, X_test_partiel, Y_test_partiel, n_neighbors_list)
df = pd.DataFrame(gen, columns=["# neighbors", "accuracy", "degrés de liberté"])
sp = sns.lineplot(x="degrés de liberté", y="accuracy", data=df)
sp.set(xscale="log")
plt.show()

Kopt = df.loc[df["accuracy"].idxmax(), "# neighbors"]
cls = KNeighborsClassifier(n_neighbors = Kopt)


cls.fit(X, y)
plot_clustering(X, y)
add_decision_boundary(cls)
plt.show()

plot_clustering(X_train_partiel.iloc[:, 20:22], Y_train_partiel)
plt.show()