# I] Extraction des données


## DL video

Lien url dans functions.py avec la fonction read_video
DL de la video avec download_video() via pytube
detection des scene pour chaque video avec fin_scenes via scenedetect


## Création json

Save video dans valid_output
filtrage données : frame non-conservée si :
    1. > 1 personne
    2. 0 dans n'importe quel points qui n'est pas la tête
    3. mean(5pts les moins confiants)<0.5
si validation frame:
    1. go to valid_output/scene/number_frame
si valid_output/scenes is empty -> remove valid_output

On obtient un dossier valid_output avec autant de dossiers que de videos non vides (ie il existe au moins une frame dans la video exploitable), chaque dossier contient une video .avi avec openpose superposé et des json contennant les données associées à chaque frame

## Processing des data

De ces json on vient créer un numpy array en conservant que le keypoints (ie on supprime les valeurs de confiance associées à chaque points)

# II] Auto-Encoder

## Load des data

Pour attaquer le réseau de neurone, les données sont loadés depuis le .npy généré à l'étape précédente. On vient retirer la tête car celle-ci est mal détectée par openpose.
On obtient un tableau de taille (42x1) car il y a 21 points de 2 coordonnées (2D).
Puis on élimine les pieds pour donner plus de poids au bras. En effet c'est les mouvements du haut du corps qui nous intéresse et donc retirer des points dans le bas du corps permet de mieux apprendre le haut du corps.
Enfin on crée un tableau de taille (15x2) qui correspond au modèle de corps BODY-25 sans les pieds et la tête (aka BODY-25 NewBlack)
Cette fonction a un argument sequence qui permet d'organiser les data si on souhaite entrainer un réseau du séquence2séquence

## Normalize

Pour chaque frame on vient centrer les kp et les normaliser entre -1 et 1

## Encoder + Decoder

Sur les données normalisées on crée un AutoEncoder de 3 couches cachées pour le decode et l'encode. Entre chaque couche il ya un Batch Normalization dropout de 0.2 . Et relu comme fonction d'activation.

Pour éviter des pb d'écrétage ou de pertes de données du fait de relu on prend une fonction d'activation linéaire en entrée et sortie du réseau

On save le modèle de réseau entrainé dans un dossier model_'nkp' (nkp = nombre keypoints)

## Latent representation

Utilisation de la TSNE pour representer l'espace latent en 3D. Mais non-linéaire donc utilisation de la PCA pour pouvoir inverser et choisir le mouvement.
Fonction create_pca(nb_compo) renvoie l'objet pca et x le résultat de pca.fit_transform()

## Interpolation des mouvements

Pour representer un mouvement on choisit 2 poses aléatoirement dans x_train, on réalise un predict de l'encoder sur ces poses puis n visualise la pca. On fait une interpo (ie discrétisation d'un segment entre l'arrivée et la sortie sur 10 poses) dans l'espace latent via la pca.
Pour pas voir trop de données on fait un predict de l'encoder avec un pas de 50.


# Conversion données 2D vers 3D

# Représentation dans GRETA