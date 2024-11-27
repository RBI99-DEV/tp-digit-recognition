import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Chemin du modèle et du fichier CSV
MODEL_PATH = r".\mnist_model.pkl"
DEFAULT_CSV_PATH = r".\data\test.csv"


########### Charger et prétraiter les données du fichier CSV


def preprocess_csv(file_path):
    """
    Charge un fichier CSV contenant des données d'images et les prétraite pour le modèle MNIST.
    :param file_path: Chemin du fichier CSV.
    :return: Données prétraitées sous forme de tableau.
    """
    data = pd.read_csv(file_path)
    images = data.values / 255.0  # Normaliser les pixels entre 0 et 1
    return images


########### Charger le modèle et prédire les données du CSV


def predict_csv(file_path, model_path):
    """
    Charge le modèle et prédit les chiffres manuscrits dans un fichier CSV.
    :param file_path: Chemin du fichier CSV.
    :param model_path: Chemin vers le fichier du modèle sauvegardé.
    """
    # Charger le modèle
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Prétraiter les données du CSV
    processed_images = preprocess_csv(file_path)

    # Faire des prédictions
    predictions = model.predict(processed_images)

    # Afficher les prédictions pour les premières images
    with open("report.json", "r", encoding="utf-8") as file:
        report = json.load(file)
        for i in range(min(100, len(processed_images))):  # Limiter à 20 images pour l'affichage
            precision = "{:.2f}".format(report[str(predictions[i])]["precision"]*100)
            plt.imshow(processed_images[i].reshape(28, 28), cmap="gray")
            plt.title(f"Prédiction : {predictions[i]} ({precision} % de precision)")
            plt.axis("off")
            plt.show()


########### Point d'entrée principal


def main():
    """
    Point d'entrée principal du script.
    """
    # Utiliser le fichier CSV par défaut
    csv_path = DEFAULT_CSV_PATH

    # Faire la prédiction
    predict_csv(csv_path, MODEL_PATH)


if __name__ == "__main__":
    main()
