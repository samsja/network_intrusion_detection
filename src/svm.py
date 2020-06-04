# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


from src.pretraitement_reduit import X_train_scaled, X_train, Y_train, X_test, X_test_scaled, Y_test,\
    newdata, newdata_test

# %%
# SVM
clf = svm.SVC(kernel='linear') # Linear Kernel
clf.fit(X_train_scaled, Y_train)
y_pred = clf.predict(X_test_scaled)
Y_test.value_counts()
print("Accuracy:",accuracy_score(Y_test, y_pred))
print("Precision score:", precision_score(Y_test, y_pred, average="macro"))
print("Recall score:", recall_score(Y_test, y_pred, average="macro"))
print("F1-score:",f1_score(Y_test, y_pred, average="macro"))
print(confusion_matrix(Y_test, y_pred))


clf_rbf = svm.SVC(kernel='rbf')
clf_rbf.fit(X_train_scaled, Y_train)
y_pred = clf_rbf.predict(X_test_scaled)
print("Accuracy score:", accuracy_score(Y_test, y_pred))
print("Precision score:", precision_score(Y_test, y_pred, average="macro"))
print("Recall score:", recall_score(Y_test, y_pred, average="macro"))
print("F1-score:",f1_score(Y_test, y_pred, average="macro"))
print(confusion_matrix(Y_test, y_pred))

X_test_scaled[['wrong_fragment','urgent']].ax_matrix

plot_decision_regions(X_test_scaled[['wrong_fragment','urgent']].as_matrix(), Y_test.astype(np.integer).as_matrix(), clf=clf, legend=2)
