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
path = "./data/kddcup.names"
col_name = read_column_from_file(path)

# %% 2. Affecter les catégories sur les types attack différents
path = "./data/training_attack_types"
attack_type = read_lab_from_file(path)

# %% 3. Récupérer les données
data = pd.read_csv("./data/kddcup.data_10_percent.csv", sep=",", names=col_name)
testdata = pd.read_csv("./data/corrected.csv", names=col_name)
# %%
data['type_of_attack'] = data.name_of_attack.apply(lambda r: attack_type[r[:-1]])
data['class'] = np.where(data['type_of_attack'] == 'normal', 'normal', 'attack')
data.info()
print("data: {} rows and {} columns".format(data.shape[0], data.shape[1]))
data.describe()
data.duplicated()
data.drop_duplicates()

testdata['class'] = np.where(testdata['name_of_attack'] == 'normal.', 'normal', 'attack')
testdata.info()
print("test data: {} rows and {} columns".format(testdata.shape[0], testdata.shape[1]))
testdata.describe()
testdata.duplicated()
testdata.drop_duplicates()
# %% 4. Search the missing value
data.isnull().any()
testdata.isnull().any
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

to_drop = [col for col in tri_df.columns if any(tri_df[col] >= 0.94)]
# not sure to delete the count and srv_count or not
col_dropped = col_dropped + to_drop
col_supp = ["name_of_attack", "type_of_attack"]
data_reduced = data.drop(col_dropped+col_supp, axis=1)
# We have already transformed the label column "name_of_attack" of our original data into "class"
# So we only have to keep the column "class" and drop the other two "name_of_attack" and "type_of_attack"
testdata_reduced = testdata.drop(col_dropped, axis=1)
testdata_reduced = testdata_reduced.drop('name_of_attack', axis=1)



# %% 8. One Hot Encoding for categorical
# Training set
print('Training set:')
for col_name in data_reduced.columns:
    if data_reduced[col_name].dtypes.name == "object":
        unique_cat = len(data_reduced[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

print('Test set:')
for col_name in testdata_reduced.columns:
    if testdata_reduced[col_name].dtypes.name == "object":
        unique_cat = len(testdata_reduced[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

# Label Encoder
# insert code to get a list of categorical columns into a variable, categorical_columns
categorical_columns = ['protocol_type', 'service', 'flag']
# Get the categorical values into a 2D numpy array
data_categorical_values = data_reduced[categorical_columns]
testdata_categorical_values = testdata_reduced[categorical_columns]
data_categorical_values.head()

# Make column names for dummies
# protocol type
unique_protocol = sorted(data_reduced.protocol_type.unique())
string1 = 'Protocol_type_'
unique_protocol2 = [string1 + x for x in unique_protocol]
# service
unique_service = sorted(data_reduced.service.unique())
string2 = 'service_'
unique_service2 = [string2 + x for x in unique_service]
# flag
unique_flag = sorted(data_reduced.flag.unique())
string3 = 'flag_'
unique_flag2 = [string3 + x for x in unique_flag]
# put together
dum_cols=unique_protocol2 + unique_service2 + unique_flag2
print(dum_cols)

# pareil pour le données du test
unique_service_test = sorted(testdata_reduced.service.unique())
unique_service2_test = [string2 + x for x in unique_service_test]
test_dum_cols = unique_protocol2 + unique_service2_test + unique_flag2

# Transformer les nominales features à entier en utilisant LabelEncoder()
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
data_categorical_values_int = data_categorical_values.apply(LabelEncoder().fit_transform)
print(data_categorical_values_int.head())
# test set
testdata_categorical_values_int = testdata_categorical_values.apply(LabelEncoder().fit_transform)

# One Hot Encoding
ohe = OneHotEncoder()
data_categorical_values_ohe = ohe.fit_transform(data_categorical_values_int)
data_dummy = pd.DataFrame(data_categorical_values_ohe.toarray(), columns=dum_cols)
# test set
testdata_categorical_values_ohe = ohe.fit_transform(testdata_categorical_values_int)
test_dummy = pd.DataFrame(testdata_categorical_values_ohe.toarray(), columns=test_dum_cols)

# Add one missing column from data to test data and also from test data to data
trainservice = data_reduced['service'].tolist()
testservice = testdata_reduced['service'].tolist()
difference = list(set(trainservice) - set(testservice))
string = 'service_'
difference = [string + x for x in difference]
difference

difference_test_data = list(set(testservice) - set(trainservice))
string = 'service_'
difference_test_data = [string + x for x in difference_test_data]
difference_test_data

for col in difference:
    test_dummy[col] = 0
test_dummy.shape

for col in difference_test_data:
    data_dummy[col] = 0
data_dummy.shape

# Ajouter encoded categorical dataframe à dataframe orignal
newdata = data_reduced.join(data_dummy)
newdata.drop('flag', axis=1, inplace=True)
newdata.drop('protocol_type', axis=1, inplace=True)
newdata.drop('service', axis=1, inplace=True)
# test data
newdata_test = testdata_reduced.join(test_dummy)
newdata_test.drop('flag', axis=1, inplace=True)
newdata_test.drop('protocol_type', axis=1, inplace=True)
newdata_test.drop('service', axis=1, inplace=True)
print(newdata.shape)
print(newdata_test.shape)






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
X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
X_train_scaled.describe()

