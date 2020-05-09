# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils import plot_correlation_matrix
from src.utils import read_column_from_file
from src.utils import read_lab_from_file
from src.utils import normalize_features
from src.utils import standardize_features
from sklearn.model_selection import train_test_split
import seaborn as sns

pd.set_option('display.max_columns', 500)  # 最大列数

# %% 1. Ajouter les noms du colonnes sur les données
path = "../data/kddcup.names"
col_name = read_column_from_file(path)

# %% 2. Affecter les catégories sur les types attack différents
path = "../data/training_attack_types"
attack_type = read_lab_from_file(path)

# %% 3. Récupérer les données
data = pd.read_csv("../data/kddcup.data_10_percent.csv", sep=",", names=col_name)
# %%
testdata = pd.read_csv("../data/kddcup.testdata.unlabeled_10_percent.csv", names=col_name)
# %%
data['type_of_attack'] = data.name_of_attack.apply(lambda r: attack_type[r[:-1]])
data['class'] = np.where(data['type_of_attack'] == 'normal', 'normal', 'attack')
data.info()
print("{} rows and {} columns".format(data.shape[0], data.shape[1]))
data.describe()

# %% 4. Search the missing value
data.isnull().any()
# résultat : aucune valeur manquante

# %% 5. Néttoyage
# Nous allons d'abord supprimer les variables qui ont des valeurs constantes
data_std = data.std()
data_std = data_std.sort_values(ascending=True)
data_std
col_dropped = ["is_host_login", "num_outbound_cmds"]

# %% Corrélation
plot_correlation_matrix(data, graphWidth=30)

corr_matrix = data.corr().abs()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df = corr_matrix.mask(mask)

to_drop = [col for col in tri_df.columns if any(tri_df[col] >= 0.95)]
# not sure to delete the count and srv_count or not
col_dropped = col_dropped + to_drop
data_reduced = data.drop(col_dropped, axis=1)

# We have already transformed the label column "name_of_attack" of our original data into "class"
# So we only have to keep the column "class" and drop the other two "name_of_attack" and "type_of_attack"
col_supp = ["name_of_attack", "type_of_attack"]
data_reduced = data_reduced.drop(col_supp, axis=1)

# %% 6. Séparation de Training set et Test set (Validation set)
X_data = data_reduced.drop(["class"], axis=1)
Y_data = data_reduced["class"]

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, train_size=0.9, random_state=42)
# %%
print("shape of X_train set : {} rows, {} columns".format(X_train.shape[0], X_train.shape[1]))
print("shape of X_test set : {} rows, {} columns".format(X_test.shape[0], X_test.shape[1]))
print("length of Y_train {}, and Y_test {}".format(len(Y_train), len(Y_test)))

# %% 7. Scaling We can choose either normalisation or standardization to scale the data according to different
# algorithm and the distribution of data
# %%
X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
X_train_scaled.describe()

# %% 8. One Hot Encoding for categorical
for feature_i in X_train_scaled.select_dtypes(include=['object']):
    X_train_scaled[feature_i] = X_train_scaled[feature_i].astype('category')
    X_test_scaled[feature_i] = X_test_scaled[feature_i].astype('category')
# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X_train_scaled['protocol_type'] = le.fit_transform(X_train_scaled['protocol_type'])
X_train_scaled['flag'] = le.fit_transform(X_train_scaled['flag'])

X_test_scaled['protocol_type'] = le.fit_transform(X_test_scaled['protocol_type'])

# %%
X_train.info()
# %%
len(data["service"].unique())
