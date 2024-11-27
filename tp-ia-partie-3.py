import json
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os  # Importation ajoutée

# Chemin du modèle et des images
MODEL_PATH = r".\mnist_model.pkl"
DEFAULT_IMAGE_PATHS = [rf".\data\chiffre_test{i}.png" for i in range(1, 4)]  # Liste des chemins d'images par défaut


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
    # Charger le modèle
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Prétraiter l'image
    processed_image = preprocess_image(image_path)

    # Faire une prédiction
    prediction = model.predict(processed_image)
    print(f"Le modèle prédit : {prediction[0]} pour l'image {os.path.basename(image_path)}")

    # Afficher l'image avec la prédiction
    
    with open("report.json", "r", encoding="utf-8") as file:
        report = json.load(file)
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
    # Prédire pour chaque image par défaut
    for image_path in DEFAULT_IMAGE_PATHS:
        predict_image(image_path, MODEL_PATH)


if __name__ == "__main__":
    main()
