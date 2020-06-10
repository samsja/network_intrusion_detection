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
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

pd.set_option('display.max_columns', 500)  # 最大列数


def data_formated(attack_type, data, testdata):

    data['type_of_attack'] = data.name_of_attack.apply(lambda r: attack_type[r[:-1]])
    data['class'] = np.where(data['type_of_attack'] == 'normal', 'normal', 'attack')
    data.drop_duplicates()
    print("data: {} rows and {} columns".format(data.shape[0], data.shape[1]))

    testdata['class'] = np.where(testdata['name_of_attack'] == 'normal.', 'normal', 'attack')
    print("test data: {} rows and {} columns".format(testdata.shape[0], testdata.shape[1]))
    testdata.drop_duplicates()
    return data, testdata


def remove_variable(data, testdata):
    col_dropped = ["is_host_login", "num_outbound_cmds"]
    corr_matrix = data.corr().abs()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    tri_df = corr_matrix.mask(mask)

    to_drop = [col for col in tri_df.columns if any(tri_df[col] >= 0.94)]
    # not sure to delete the count and srv_count or not
    col_dropped = col_dropped + to_drop
    col_supp = ["name_of_attack", "type_of_attack"]
    data = data.drop(col_dropped + col_supp, axis=1)


    testdata = testdata.drop(col_dropped, axis=1)
    testdata = testdata.drop('name_of_attack', axis=1)

    return data, testdata


def one_hot_encoding(data, testdata):
    categorical_columns = ['protocol_type', 'service', 'flag']
    data_categorical_values = data[categorical_columns]
    testdata_categorical_values = testdata[categorical_columns]

    unique_protocol = sorted(data.protocol_type.unique())
    string1 = 'Protocol_type_'
    unique_protocol2 = [string1 + x for x in unique_protocol]
    # service
    unique_service = sorted(data.service.unique())
    string2 = 'service_'
    unique_service2 = [string2 + x for x in unique_service]
    # flag
    unique_flag = sorted(data.flag.unique())
    string3 = 'flag_'
    unique_flag2 = [string3 + x for x in unique_flag]
    # put together
    dum_cols = unique_protocol2 + unique_service2 + unique_flag2

    unique_service_test = sorted(testdata.service.unique())
    unique_service2_test = [string2 + x for x in unique_service_test]
    test_dum_cols = unique_protocol2 + unique_service2_test + unique_flag2

    data_categorical_values_int = data_categorical_values.apply(LabelEncoder().fit_transform)
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
    trainservice = data['service'].tolist()
    testservice = testdata['service'].tolist()
    difference = list(set(trainservice) - set(testservice))
    string = 'service_'
    difference = [string + x for x in difference]


    difference_test_data = list(set(testservice) - set(trainservice))
    difference_test_data = [string + x for x in difference_test_data]

    for col in difference:
        test_dummy[col] = 0

    for col in difference_test_data:
        data_dummy[col] = 0

    # Ajouter encoded categorical dataframe à dataframe orignal
    value_class = data['class']
    X_data_reduced = data.drop('class', axis=1)
    newdata = X_data_reduced.join(data_dummy)
    newdata.insert(0,'class', value_class)
    newdata.drop('flag', axis=1, inplace=True)
    newdata.drop('protocol_type', axis=1, inplace=True)
    newdata.drop('service', axis=1, inplace=True)
    # test data
    value_class_test = testdata['class']
    X_testdata_reduced = testdata.drop('class', axis=1)
    newdata_test = X_testdata_reduced.join(test_dummy)
    newdata_test.insert(0,'class', value_class_test)
    newdata_test.drop('flag', axis=1, inplace=True)
    newdata_test.drop('protocol_type', axis=1, inplace=True)
    newdata_test.drop('service', axis=1, inplace=True)
    print(newdata.shape)
    print(newdata_test.shape)

    return newdata, newdata_test

def seperate_train_test(data):
    X_data = data.drop(["class"], axis=1)
    Y_data = data["class"]
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, train_size=0.9, random_state=42)

    print("shape of X_train set : {} rows, {} columns".format(X_train.shape[0], X_train.shape[1]))
    print("shape of X_test set : {} rows, {} columns".format(X_test.shape[0], X_test.shape[1]))
    print("length of Y_train {}, and Y_test {}".format(len(Y_train), len(Y_test)))

    return X_train, Y_train, X_test, Y_test

def pretraitement(attack_type, data, testdata):
    data, testdata = data_formated(attack_type, data, testdata)
    data, testdata = remove_variable(data, testdata)
    data, testdata = one_hot_encoding(data, testdata)
    X_train, Y_train, X_test, Y_test = seperate_train_test(data)
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    return X_train, Y_train, X_test, Y_test, X_train_scaled, X_test_scaled, data, testdata





