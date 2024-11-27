#### Import de base
import json
from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle
from PIL import Image

# Chemin des fichiers
TRAIN_PATH = r".\data\train.csv"
MODEL_PATH = r".\mnist_model.pkl"


########### Charger et préparer les données


def load_and_prepare_data(train_path, test_size=0.2):
    """
    Charge les données d'entraînement, normalise et divise en ensembles d'entraînement et de test.
    :param train_path: Chemin vers le fichier CSV contenant les données d'entraînement.
    :param test_size: Taille de l'ensemble de test (entre 0 et 1).
    :return: X_train, X_test, y_train, y_test
    """
    train_data = pd.read_csv(train_path)
    X = train_data.iloc[:, 1:].values / 255.0  # Normalisation des pixels
    y = train_data.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test


########### Entraîner et sauvegarder le modèle


def train_and_save_model(X_train, y_train, model_path):
    """
    Entraîne un modèle Random Forest et le sauvegarde.
    :param X_train: Données d'entraînement.
    :param y_train: Labels d'entraînement.
    :param model_path: Chemin pour sauvegarder le modèle.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modèle entraîné et sauvegardé dans '{model_path}'.")


########### Évaluer le modèle


def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur les données de test.
    :param model: Modèle entraîné.
    :param X_test: Données de test.
    :param y_test: Labels de test.
    """
    y_pred = model.predict(X_test)
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))
    global report
    report = classification_report(y_test, y_pred, output_dict=True)
    with open("report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=4)



########### Charger et prétraiter une image


def preprocess_image(image_path):
    """
    Charge une image et la prétraite pour le modèle MNIST.
    :param image_path: Chemin de l'image à prédire.
    :return: Image prétraitée sous forme de vecteur.
    """
    image = Image.open(image_path).convert("L")  # Convertir en niveaux de gris
    image = image.resize((28, 28))  # Redimensionner à 28x28 pixels
    image_array = np.array(image) / 255.0  # Normaliser les pixels entre 0 et 1
    return image_array.flatten().reshape(1, -1)


########### Charger le modèle et prédire une image


def predict_image(image_path, model_path):
    """
    Charge le modèle et prédit le chiffre manuscrit dans une image.
    :param image_path: Chemin de l'image.
    :param model_path: Chemin vers le fichier du modèle sauvegardé.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    print(f"Le modèle prédit : {prediction[0]}")
    precision = "{:.2f}".format(report[str(prediction[0])]["precision"]*100)
    plt.imshow(processed_image.reshape(28, 28), cmap="gray")
    plt.title(f"Prédiction : {prediction[0]} ({precision} % de precision)")
    plt.axis("off")
    plt.show()


########### Point d'entrée principal


def main():
    """
    Point d'entrée principal du script.
    """
    # Charger et préparer les données
    X_train, X_test, y_train, y_test = load_and_prepare_data(TRAIN_PATH)

    # Entraîner et sauvegarder le modèle
    train_and_save_model(X_train, y_train, MODEL_PATH)

    # Charger le modèle et évaluer sur les données de test
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    evaluate_model(model, X_test, y_test)

    # Prédire une image
    for i in range(1, 4):  # De 1 à 3 inclus
        image_path = rf".\data\chiffre_test{i}.png"  # Chemin de l'image à prédire
        predict_image(image_path, MODEL_PATH)


if __name__ == "__main__":
    main()
