# tp-digit-recognition

TP IA Partie 1

Nous importons dans un premier temps un csv (données d'entrainement).
Ensuite en lançant le script :
Cela va afficher dans un premier temps une image :
afficher_image(1, train_data) > on peut modifier le 1 pour changer d'image.

Ensuite cela va afficher dans un second temps les chiffres de 0 à 9 en images.

Après cela va afficher un chiffre en différentes façons :
afficher_chiffre_multiple(train_data, chiffre=7) (on peut changer le 7 pour changer de chiffre)

Et enfin cela va afficher l'affiche "moyenne des chiffres".

TP IA Partie 2

Dans la partie 2, nous allons créer un modèle avec notamment model = RandomForestClassifier(n_estimators=100, random_state=42) model.fit(X_train, y_train).

Pour ensuite tester avec des images qui ont été créés sur Paint qui sont chiffre_testx (dans le script l'image est retraitée pour être adaptée MNIST), voir si cela réussit à identifier les chiffres.

Dans l'affichage, il y aura également le % de précision du modèle pour les différents chiffres. (On a enregistré les % dans le fichier report.json)

TP IA Partie 3

Dans la partie 3, nous avons importé directement le modèle qui se nomme mnist_model.pkl pour ne pas avoir à relancer à chaque fois le modèle pour tester d'identifier les chiffres.

Dans l'affichage, il y aura également le % de précision du modèle pour les différents chiffres. (récupéré depuis le fichier report.json)

TP IA Partie 4 

Dans la partie 4, nous avons utilisé le second fichier csv fourni (test.csv) pour tester le modèle à plus grande échelle et voir s'il fonctionne bien et avec des données directement d'un csv.

Dans l'affichage, il y aura également le % de précision du modèle pour les différents chiffres. (récupéré depuis le fichier report.json)


On peut également voir que le chiffre ou il se trompe le + est le 9 (93%). Probalement à cause de sa ressemblance avec d'autres chiffre (0,8)

Dans .gitignore, on a mis report.json et mnist_model.pkl pour ne pas les sauvegarder sur github car ils sont générés à l'éxécution.
