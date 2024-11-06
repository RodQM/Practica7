import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_wine
import seaborn as sns
import matplotlib.pyplot as plt

# Función auxiliar para cargar y procesar el dataset de Wine Quality
def load_wine_quality():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    X = df.drop('quality', axis=1)
    y = df['quality']
    return X, y

# Datasets
wine = load_wine()
wine_quality_X, wine_quality_y = load_wine_quality()
seeds_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
seeds_columns = ["area", "perimeter", "compactness", "length_kernel", "width_kernel", "asymmetry_coef", "groove_length", "type"]
seeds = pd.read_csv(seeds_url, sep='\s+', names=seeds_columns)

# Cargar datos de Iris (de scikit-learn)
iris = load_wine()
iris_X, iris_y = iris.data, iris.target

# Función principal para elegir el método de validación y ejecutar el clasificador
def main():
    print("Elige el método de validación:")
    print("1. Hold-Out 70/30 Estratificado")
    print("2. 10 Fold Cross-Validation Estratificado")
    print("3. Leave-One-Out Cross-Validation")
    opcion = int(input("Ingresa el número de la opción deseada: "))

    datasets = [
        (wine_quality_X, wine_quality_y, "Wine Quality"),
        (seeds.drop('type', axis=1), seeds['type'], "Seeds"),
        (iris_X, iris_y, "Iris")
    ]

    for X, y, dataset_name in datasets:
        print(f"\nProcesando dataset: {dataset_name}")

        # Escalar las características
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Crear el clasificador KNN
        knn = KNeighborsClassifier(n_neighbors=5)

        if opcion == 1:
            # Hold-Out 70/30 Estratificado
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            print(f"Accuracy: {accuracy}")
            print("Matriz de Confusión:")
            print(cm)
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
            plt.title(f'Matriz de Confusión - Hold-Out ({dataset_name})')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()

        elif opcion == 2:
            # 10 Fold Cross-Validation Estratificado
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            scores = cross_val_score(knn, X, y, cv=skf)
            print(f"Accuracy promedio con 10 Fold Cross-Validation: {scores.mean()}")
            print(f"Desviación estándar de la accuracy: {scores.std()}")

        elif opcion == 3:
            # Leave-One-Out Cross-Validation
            loo = LeaveOneOut()
            scores = cross_val_score(knn, X, y, cv=loo)
            print(f"Accuracy promedio con Leave-One-Out Cross-Validation: {scores.mean()}")

        else:
            print("Opción de validación no válida.")
            return

if __name__ == "__main__":
    main()
