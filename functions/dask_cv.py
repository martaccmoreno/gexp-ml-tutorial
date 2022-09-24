import dask
import dask.array as da
from dask_ml.model_selection import KFold
from dask_ml.metrics import accuracy_score, r2_score


def fit_estimator(estimator, X_train: da.Array, y_train: da.Array, classes: list = None):
    """
    Fit the estimator to the training fold (portion) of a given split.
    A list of all unique class labels must be provided when fitting incremental learning estimators.
    """
    if classes is list:
        estimator.fit(X_train, y_train, classes=classes)
    else:
        estimator.fit(X_train, y_train)
    return estimator


def prediction_score(estimator, X_test: da.Array, y_test: da.Array, hpo: bool = False, regression: bool = False):
    """
    Compare the prediction of the fitted estimator versus the actual values for the untouched test fold of each split,
    and return the performance evaluation as a score.

    The returned score is accuracy for classification tasks, and the r² score for regression tasks.
    """
    if hpo:
        y_pred = estimator.best_estimator_.predict(X_test)
    else:
        y_pred = estimator.predict(X_test)

    if regression:
        return r2_score(y_test, y_pred)
    else:
        return accuracy_score(y_test, y_pred)


def dask_cv(estimator, X: da.Array, y: da.Array, cv_splits: int = 10, classes: list = None, hpo: bool = False,
            regression: bool = False) -> list[float]:
    """
    Calculate cross-validation scores in a parallelized fashion with Dask.
    This function is optimized for use in machine learning tasks involving gene expression data.

    :param estimator: A scikit-learn or Dask estimator for which to calculate cross-validation scores.
    :param X: features, namely a matrix of processed gene expression data.
    :param y: labels, namely the phenotypic classes to predict.
    :param cv_splits: number of data splits on which to conduct cross-validation.
    :param classes: list of unique classes to account for while fitting an incremental learning estimator.
    :param hpo: whether we're dealing with a meta-estimator for hyper-parameter optimization.
    :param regression: whether the prediction task at hand is a regression problem as opposed to classification.
    :return: a list of cross-validation scores for each fold, i.e. len(return) = cv_splits
    """
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    splits = kf.split(X, y)

    scores = []
    for train_split_idx, test_split_idx in splits:
        fitted_estimator = dask.delayed(fit_estimator)(estimator, X[train_split_idx], y[train_split_idx], classes)
        score = dask.delayed(prediction_score)(fitted_estimator, X[test_split_idx], y[test_split_idx], hpo, regression)
        scores.append(score)

    return scores


if __name__ == '__main__':
    # from dask_ml.datasets import make_classification
    # X, y = make_classification(n_samples=1000, n_features=20000, chunks=(100, 20000), random_state=42)

    from dask.distributed import Client
    client = Client()
    print(client)

    import dask.dataframe as dd
    X = dd.read_parquet('/home/marta/IdeaProjects/ML-DASK/data/brca_undersample/parquet/n1205_f20000_brca_fpkm.parquet')
    y = dd.read_parquet('/home/marta/IdeaProjects/ML-DASK/data/brca_undersample/parquet/n1205_brca_subtypes.parquet')

    X_arr = X.to_dask_array(lengths=True)
    y_arr = y.to_dask_array(lengths=True)
    from dask_ml.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_arr = le.fit_transform(y_arr)

    from xgboost import XGBClassifier
    bst = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)

    from dask.distributed import Client
    client = Client()
    resulting_scores = dask.compute(dask_cv(bst, X_arr, y_arr))
    print(resulting_scores)