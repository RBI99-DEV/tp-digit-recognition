import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_data = pd.read_csv(r".\data\train.csv")

# Afficher image quelconque


def afficher_image(index, dataset):
    """
    Affiche une image à partir des données MNIST.
    :param index: Index de l'image à afficher.
    :param dataset: Dataset contenant les labels et les pixels.
    """
    image = dataset.iloc[index, 1:].values.reshape(28, 28)  # Pixels de l'image
    label = dataset.iloc[index, 0]  # Label de l'image
    plt.imshow(image, cmap="gray")
    plt.title(f"Label : {label}")
    plt.axis("off")  # Supprime les axes pour une meilleure lisibilité
    plt.show()


# Afficher une image (par exemple, la première ou la deuxième)
afficher_image(1, train_data)

# Afficher de 0 à 9


def afficher_chiffres_0_9(dataset):
    """
    Affiche une image pour chaque chiffre de 0 à 9.
    :param dataset: Dataset contenant les labels et les pixels.
    """
    plt.figure(figsize=(10, 5))
    for chiffre in range(10):
        # Filtrer les lignes correspondant au chiffre
        image = (
            dataset[dataset.iloc[:, 0] == chiffre].iloc[0, 1:].values.reshape(28, 28)
        )
        plt.subplot(2, 5, chiffre + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"{chiffre}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# Afficher un exemple de chaque chiffre
afficher_chiffres_0_9(train_data)

# Affiche chiffre 7 de différentes façons


def afficher_chiffre_multiple(dataset, chiffre, n_images=9):
    """
    Affiche plusieurs images représentant un chiffre spécifique.
    :param dataset: Dataset contenant les labels et les pixels.
    :param chiffre: Le chiffre à afficher (par exemple, 7).
    :param n_images: Nombre d'images à afficher.
    """
    images = dataset[dataset.iloc[:, 0] == chiffre].iloc[:n_images, 1:].values
    plt.figure(figsize=(10, 5))
    for i, image in enumerate(images):
        plt.subplot(1, n_images, i + 1)
        plt.imshow(image.reshape(28, 28), cmap="gray")
        plt.title(f"{chiffre}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# Afficher les premières images du chiffre 7
afficher_chiffre_multiple(train_data, chiffre=7)

# Afficher la moyenne des chiffres


def afficher_chiffre_moyen(dataset):
    """
    Calcule et affiche une image moyenne pour chaque chiffre de 0 à 9.
    :param dataset: Dataset contenant les labels et les pixels.
    """
    plt.figure(figsize=(10, 5))
    for chiffre in range(10):
        # Moyenne des pixels pour toutes les images du même chiffre
        images = (
            dataset[dataset.iloc[:, 0] == chiffre]
            .iloc[:, 1:]
            .mean(axis=0)
            .values.reshape(28, 28)
        )
        plt.subplot(2, 5, chiffre + 1)
        plt.imshow(images, cmap="gray")
        plt.title(f"Moyen {chiffre}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# Afficher les représentants moyens pour chaque chiffre
afficher_chiffre_moyen(train_data)
