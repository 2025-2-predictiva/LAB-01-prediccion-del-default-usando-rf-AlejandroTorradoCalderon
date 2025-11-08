import pandas as pd
import gzip
import json
import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path, index_col=False, compression="zip")
    df_test = pd.read_csv(test_path, index_col=False, compression="zip")

    print("Datos cargados exitosamente")

    return df_train, df_test


def preprocess_data(df, set_name):
    # Renombrar columna
    df = df.rename(columns={"default payment next month": "default"})

    # Remover columna ID
    df = df.drop(columns=["ID"])

    # Eliminar registros con informacion no disponible
    df = df.dropna()

    # Agrupar valores de EDUCATION > 4
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x < 4 else 4)

    print(f"Preprocesamiento de datos {set_name} completado")

    return df


def split_features_target(df, target_name):
    X = df.drop(columns=[target_name])
    y = df[target_name]

    print("División de características y target completada")

    return X, y


def pipeline_definition():
    # Definir las variables categóricas
    categorical_features = ["EDUCATION", "MARRIAGE", "SEX"]

    # Crear el preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",
    )

    # Crear el pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    print("Definición del pipeline completada")

    return pipeline


def hyperparameter_optimization(pipeline, X_train, y_train):
    param_grid = {
        "classifier__n_estimators": [100],
        "classifier__max_depth": [None],
        "classifier__min_samples_split": [10],
        "classifier__min_samples_leaf": [4],
        "classifier__max_features": [None],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        verbose=1,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    print("Optimización de hiperparámetros completada")
    
    return grid_search


def save_model(model, model_path):
    # Guardar el modelo comprimido con gzip
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with gzip.open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("Modelo guardado exitosamente")


def calculate_metrics(model, X, y, dataset_type):
    y_pred = model.predict(X)

    # Calcular métricas
    precision = precision_score(y, y_pred)
    balanced_acc = balanced_accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    metrics = {
        "type": "metrics",
        "dataset": dataset_type,
        "precision": precision,
        "balanced_accuracy": balanced_acc,
        "recall": recall,
        "f1_score": f1,
    }

    # Calcular matriz de confusión
    cm = confusion_matrix(y, y_pred)
    cm_dict = {
        "type": "cm_matrix",
        "dataset": dataset_type,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }

    print(f"Cálculo de métricas completado para el conjunto de {dataset_type}")

    return metrics, cm_dict


def save_metrics(metrics, metrics_path):
    # Guardar las métricas en un archivo json
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")

    print("Métricas guardadas exitosamente")


def main():
    df_train, df_test = load_data("files/input/train_data.csv.zip", "files/input/test_data.csv.zip")

    df_train = preprocess_data(df_train, "train")
    df_test = preprocess_data(df_test, "test")

    X_train, y_train = split_features_target(df_train, "default")
    X_test, y_test = split_features_target(df_test, "default")

    pipeline = pipeline_definition()
    grid_search = hyperparameter_optimization(pipeline, X_train, y_train)

    save_model(grid_search, "files/models/model.pkl.gz")

    train_metrics, train_cm = calculate_metrics(grid_search, X_train, y_train, "train")
    test_metrics, test_cm = calculate_metrics(grid_search, X_test, y_test, "test")
    
    metrics = [train_metrics, test_metrics, train_cm, test_cm]
    save_metrics(metrics, "files/output/metrics.json")


if __name__ == "__main__":
    main()