import os
import joblib
import warnings
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

def cargar_datos(path_Xtrain, path_ytrain, path_Xtest, path_ytest):
    """Carga los datos escalados desde archivos .pkl"""
    Xtrain = joblib.load(path_Xtrain)
    ytrain = joblib.load(path_ytrain)
    Xtest = joblib.load(path_Xtest)
    ytest = joblib.load(path_ytest)
    return Xtrain, ytrain, Xtest, ytest

def evaluar_modelos_regresion_mlflow(Xtrain, ytrain, Xtest, ytest, experiment_name='Regresores_GridSearch'):
    """Evalúa modelos con GridSearchCV y registra resultados con MLflow"""
    warnings.filterwarnings("ignore")
    mlflow.set_experiment(experiment_name)

    modelos_con_grid = {
        'Lasso': (Lasso(max_iter=30000), {'alpha': np.logspace(-5, 1, 20)}),
        'Ridge': (Ridge(), {'alpha': np.logspace(-5, 1, 20)}),
        'DecisionTree': (DecisionTreeRegressor(random_state=42), {'max_depth': [3, 5, 10, 20]}),
        'RandomForest': (RandomForestRegressor(random_state=42), {
            'n_estimators': [50, 100, 500, 1000], 'max_depth': range(3, 10)}),
        'Bagging': (BaggingRegressor(random_state=42), {
            'n_estimators': [50, 100, 500, 1000], 'max_samples': [0.6, 0.8, 1.0], 'max_features': [0.6, 0.8, 1.0]}),
        'XGBoost': (XGBRegressor(random_state=42, verbosity=0), {
            'n_estimators': [50, 100, 500, 1000], 'max_depth': range(3, 10)})
    }

    resultados = []

    # Guardar temporalmente los datasets
    joblib.dump(Xtrain, "Xtrain.pkl")
    joblib.dump(ytrain, "ytrain.pkl")
    joblib.dump(Xtest, "Xtest.pkl")
    joblib.dump(ytest, "ytest.pkl")

    for nombre, (modelo, params) in modelos_con_grid.items():
        with mlflow.start_run(run_name=nombre):
            grid = GridSearchCV(modelo, param_grid=params, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
            grid.fit(Xtrain, ytrain)
            mejor_modelo = grid.best_estimator_

            # Predicción y métricas
            y_pred_train = mejor_modelo.predict(Xtrain)
            y_pred_test = mejor_modelo.predict(Xtest)

            metrics = {
                "MSE_Train": mean_squared_error(ytrain, y_pred_train),
                "RMSE_Train": np.sqrt(mean_squared_error(ytrain, y_pred_train)),
                "R2_Train": r2_score(ytrain, y_pred_train),
                "MAPE_Train": mean_absolute_percentage_error(ytrain, y_pred_train),
                "MSE_Test": mean_squared_error(ytest, y_pred_test),
                "RMSE_Test": np.sqrt(mean_squared_error(ytest, y_pred_test)),
                "R2_Test": r2_score(ytest, y_pred_test),
                "MAPE_Test": mean_absolute_percentage_error(ytest, y_pred_test)
            }

            mlflow.log_metrics(metrics)
            mlflow.log_params(grid.best_params_)
            mlflow.sklearn.log_model(mejor_modelo, "modelo")

            # Log artifacts (datos)
            mlflow.log_artifact("Xtrain.pkl", artifact_path="datos")
            mlflow.log_artifact("ytrain.pkl", artifact_path="datos")
            mlflow.log_artifact("Xtest.pkl", artifact_path="datos")
            mlflow.log_artifact("ytest.pkl", artifact_path="datos")

            resultados.append([nombre, *metrics.values(), grid.best_params_])

    # DataFrame y log como CSV
    columnas = ['Modelo',
                'MSE_Train', 'RMSE_Train', 'R2_Train', 'MAPE_Train',
                'MSE_Test', 'RMSE_Test', 'R2_Test', 'MAPE_Test',
                'Mejores_Parametros']
    df_resultados = pd.DataFrame(resultados, columns=columnas)
    df_resultados.sort_values(by='RMSE_Test', inplace=True)
    df_resultados.to_csv("resultados_modelos.csv", index=False)

    # Log resumen final
    with mlflow.start_run(run_name="Resumen_Modelos"):
        mlflow.log_artifact("resultados_modelos.csv", artifact_path="resumen")

    for f in ["Xtrain.pkl", "ytrain.pkl", "Xtest.pkl", "ytest.pkl", "resultados_modelos.csv"]:
        if os.path.exists(f):
            os.remove(f)

    return df_resultados