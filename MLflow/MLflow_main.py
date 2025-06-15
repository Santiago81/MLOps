import argparse
from funciones_modelado import cargar_datos, evaluar_modelos_regresion_mlflow

def main():
    parser = argparse.ArgumentParser(description="Evaluación de modelos de regresión con MLflow")
    parser.add_argument("--path-datos", type=str, required=True,
                        help="Ruta al directorio que contiene los archivos Xtrain.pkl, ytrain.pkl, Xtest.pkl, ytest.pkl")
    parser.add_argument("--experimento", type=str, default="Regresores_GridSearch",
                        help="Nombre del experimento en MLflow")

    args = parser.parse_args()

    # Construir rutas de datos
    ruta_Xtrain = f"{args.path_datos}/XtrainScaled.pkl"
    ruta_ytrain = f"{args.path_datos}/y_train.pkl"
    ruta_Xtest = f"{args.path_datos}/XtestScaled.pkl"
    ruta_ytest = f"{args.path_datos}/y_test.pkl"

    # Ejecutar flujo
    Xtrain, ytrain, Xtest, ytest = cargar_datos(ruta_Xtrain, ruta_ytrain, ruta_Xtest, ruta_ytest)
    df_resultados = evaluar_modelos_regresion_mlflow(Xtrain, ytrain, Xtest, ytest, args.experimento)

    print("\n Resultados:")
    print(df_resultados)

if __name__ == "__main__":
    main()