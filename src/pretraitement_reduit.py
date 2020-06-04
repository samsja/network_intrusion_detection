# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

from src.utils import read_column_from_file, read_lab_from_file,normalize_features,\
    standardize_features, plot_correlation_matrix, add_decision_boundary, \
    read_column_from_file, read_lab_from_file, plot_clustering
import src.knn_cross_validation as kcv
import src.knn_validation as kv
import src.nearest_prototypes as nrp


data = pd.read_csv("./data/Train_data.csv", sep=",")
testdata = pd.read_csv("./data/Test_data.csv", sep=",")

col_dropped = ["is_host_login", "num_outbound_cmds"]
corr_matrix = data.corr().abs()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df = corr_matrix.mask(mask)
to_drop = [col for col in tri_df.columns if any(tri_df[col] >= 0.94)]

data_reduced = data.drop(col_dropped+to_drop, axis=1)
testdata_reduced = testdata.drop(col_dropped+to_drop, axis=1)

categorical_columns = ['protocol_type', 'service', 'flag']
# Get the categorical values into a 2D numpy array
data_categorical_values = data_reduced[categorical_columns]
testdata_categorical_values = testdata_reduced[categorical_columns]
data_categorical_values.head()

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
value_class = data_reduced['class']
X_data_reduced = data_reduced.drop('class', axis=1)
newdata = X_data_reduced.join(data_dummy)
newdata.insert(0,'class', value_class)
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



X_data = newdata.drop(["class"], axis=1)
Y_data = newdata["class"]
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, train_size=0.9, random_state=42)

print("shape of X_train set : {} rows, {} columns".format(X_train.shape[0], X_train.shape[1]))
print("shape of X_test set : {} rows, {} columns".format(X_test.shape[0], X_test.shape[1]))
print("length of Y_train {}, and Y_test {}".format(len(Y_train), len(Y_test)))

X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)

print(X_train_scaled.std(axis=0))
print(X_test_scaled.std(axis=0))


# knn
n_neighbors_list = np.unique(np.round(np.geomspace(1, 500, 100)).astype(int))

gen = kv.knn_simple_validation(X_train_scaled, Y_train, X_test_scaled, Y_test, n_neighbors_list)
df = pd.DataFrame(gen, columns=["# neighbors", "accuracy", "degrés de liberté"])
sp = sns.lineplot(x="degrés de liberté", y="accuracy", data=df)
sp.set(xscale="log")
plt.show()

Kopt = df.loc[df["accuracy"].idxmax(), "# neighbors"]
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X_data_scaled = scaler.fit_transform(X_data)
cls = KNeighborsClassifier(n_neighbors = Kopt)
cls.fit(X_data, Y_data)
plot_clustering(X_data, Y_data)
add_decision_boundary(cls)
plt.show()

plot_clustering(X_train_partiel.iloc[:, 20:22], Y_train_partiel)
plt.show()
