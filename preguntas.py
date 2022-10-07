"""
Regresión Lineal Multiple
-----------------------------------------------------------------------------------------

En este laboratorio se entrenara un modelo de regresión lineal multiple que incluye la 
selección de las n variables más relevantes usando una prueba f.

"""
# pylint: disable=invalid-name
# pylint: disable=unsubscriptable-object

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error


def pregunta_01():
    """
    Carga de datos.
    -------------------------------------------------------------------------------------
    """
    df = pd.read_csv('insurance.csv')

    y = df['charges']
    X = df.copy()

    X = X.drop(columns=['charges'])

    return X, y


def pregunta_02():
    """
    Preparación de los conjuntos de datos.
    -------------------------------------------------------------------------------------
    """

    X, y = pregunta_01()

    (X_train, X_test, y_train, y_test,) = train_test_split(
        X,
        y,
        test_size=300,
        random_state=12345, # aleatorios es 12345
    )

    return X_train, X_test, y_train, y_test


def pregunta_03():
    """
    Especificación del pipeline y entrenamiento
    -------------------------------------------------------------------------------------
    """

    # Importe make_column_selector
    # Importe make_column_transformer
    # Importe SelectKBest
    # Importe f_regression
    # Importe LinearRegression
    # Importe GridSearchCV
    # Importe Pipeline
    # Importe OneHotEncoder
    from ____ import ____

    pipeline = ____(
        steps=[
            # OneHotEncoder a variables categóricas, 
            (
                "column_transfomer",
                make_column_transformer(
                    (
                        OneHotEncoder(),
                        make_column_selector(dtype_include=object),
                    ),
                    remainder='passthrough',
                ),
            ),
            # Paso 2: Selector de características que seleccione las K características,
            (
                "selectKBest",
                SelectKBest(score_func=f_regression),
            ),
            # Modelo de regresión lineal.
            (
                "linearregression",
                LinearRegression(fit_intercept=True),
            ),
        ],
    )

    # Cargua de las variables.
    X_train, X_test, y_train, y_test = pregunta_02()

    param_grid = {
        'selectKBest__k': np.arange(1, 11, 1), # Parámetros para el GridSearchCV.
    }

    gridSearchCV = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error', # Métrica de evaluación 
        refit= True,
        return_train_score= True,
    )

    return gridSearchCV.fit(X_train, y_train)


def pregunta_04():
    """
    Evaluación del modelo
    -------------------------------------------------------------------------------------
    """

    gridSearchCV = pregunta_03()

    X_train, X_test, y_train, y_test = pregunta_02()

    y_train_pred = gridSearchCV.predict(X_train)
    y_test_pred = gridSearchCV.predict(X_test)

    mse_train = mean_squared_error(
        y_train,
        y_train_pred,
    ).round(2)

    mse_test = mean_squared_error(
        y_test,
        y_test_pred,
    ).round(2)

    return mse_train, mse_test
