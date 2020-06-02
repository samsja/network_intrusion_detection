from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from src.knn_validation import accuracy
from sklearn.utils import check_X_y


def knn_cross_validation(X, y, n_folds, n_neighbors_list):
    """Génère les couples nombre de voisins et précisions correspondantes."""

    # Conversion en tableau numpy si on fournit des DataFrame par exemple
    X, y = check_X_y(X, y)

    def models_accuracies(train_index, val_index, n_neighbors_list):
        """Précision de tous les modèles pour un jeu de données fixé."""

        # Création de `X_train`, `y_train`, `X_val` et `y_val`
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_val = X[val_index, :]
        y_val = y[val_index]

        # Calcul des précisions pour chaque nombre de voisins présent
        # dans `n_neighbors`
        n = len(train_index)
        for n_neighbors in n_neighbors_list:
            yield (
                n_neighbors,
                accuracy(X_train, y_train, X_val, y_val, n_neighbors),
                n / n_neighbors
            )

    # Définition de `n_folds` jeu de données avec `KFold`
    kf = KFold(n_splits=n_folds, shuffle=True).split(X)

    # Calcul et retour des précisions avec `models_accuracies` pour
    # chaque jeu de données défini par `KFold`.
    for train_index, test_index in kf:
        yield from models_accuracies(train_index, test_index, n_neighbors_list)


def knn_cross_validation2(X, y, n_folds, n_neighbors_list):
    n = (n_folds - 1) / n_folds * len(y)
    for n_neighbors in n_neighbors_list:
        cls = KNeighborsClassifier(n_neighbors=n_neighbors)
        for err in cross_val_score(cls, X, y, cv=n_folds):
            yield (n_neighbors, err, n / n_neighbors)
