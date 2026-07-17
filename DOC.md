MULTYLAYER PERCEPTRON: PROJET DE MACHINE LEARNING POUR APPRENDRE L'IMPLEMENTATION DES RESEAUX DE NEURONES


qu'est ce que un reseaux de neurones?

=> c'est un type d'algorithme utilise en machine learning, qui vise a
simuler le comportement du cerveau humain grace a des neurones aritificiels interconnectes entre eux, organises
en plusieurs couches


en quoi les reseaux de neurones sont differents du machine learning classique?

=> ils sont typiquements differents car ce sont des algorithme penses pour que le modele puisse s'ameliorer
tout seul sans l'intervention d'un humain, les modeles apprennent directement depuis les donnes qu'on leur donne,

dans le machine learning classique les features sont donnes a la main 


Comment ca marche?

=> chaque neurone represente une unite de compilation qui prend des inputs, fait des calculs basees sur ces inputs
et donne son output au prochain neurone qui va faire la meme chose, jusqu'a avoir un output.
Pendant que les donnes parcourent le reseaux, les connections entre les neurones se renforcent ou s'affaiblissent
par rapport aux patterns presents dans les donnees


poids => en machine learning en generale, un poids et l'importance que une certaine feature possede par rapport a
une prediction 

biais => constante donne a une formule pour decaler le resultat et empecher le modele de passer par l'origine
ce qui va polluer les sessions d'entrainement


Qu'est ce que le forward propagation?

=> c'est le process par lequel les donnes recues en input sont  traites dans le reseaux de neurones
pour generer un output finale
le output de chaque neurones dans chaque couche est processe en appliquant les poids et les biais aux inputs recus 
et puis en appliquant une fonction d'activation (sigmoide (voir le projet DSLR))


Qu'est ce que le back propagation?

=> c'est un algorithme tres utilise en machine learning, il calcule le gradient de descente de la fonction de perte
ce qui va permettre de regler les poids et d'avoir un entrainement plus pertinant et des predictions plus precises 



chaque noeud a une valeur numerique, pour construire un reseaux on a besoin de plusieurs noeuds
on connecte les noeuds entre eux en utilisant des poids, la somme de la multiplication entre les valeurs dun noeud et
son poids donne une valeur au prochain noeud

---------------------------------|
EXEMPLE:                         |
noed1 = 3, poids1 = 4            |
noed2 = 2, poids2 = 8            |
                                 |
noed3 = (3 * 4) + (2 * 8) = 28   |
 -> = 28                         |
---------------------------------|


UNE COUCHE DANS UN RESEAUX A 4 REGLES:

chaque noeud d'une couche, doit etre connecte a chaque noeud de la couche suivante, et les connections sont apelles des poids

chaque poids est valeur random predetermine (que au debut), on ne connait pas la valeur des noeuds  a part les noeuds dans la
couche input

seulement la premiere couche de noeuds (couche input) commence avec des valeurs

touts les autres noeud du reseaux ont une valeur qui a ete calcule grace aux valeurs des couches d'avant et leur poids


LES ETAPES POUR CREER UN RESEAUX DE NEURONES

1-> tu donnes la data a traiter a la premiere couche (la couche input), puis le reseaux utilise cette data + les poids
de chaque noed pour donner une valeur aux noeud dans les couches suivantes et ainsi de suivante

2-> ce process se propage dans chaque noed jusqu a chaque noed dans le reseaux a une valeur 

3-> la derniere couche nous donnera la prediction du modele, ensuite on va comparer cette prediction aux vraies valeurs
et on va appliquer la fonction de cout pour avoir une constante cout pour le reseaux

4-> grace a ce cout on va pouvoir applique la descente de gradient et modifier les poids et les biais

5-> repeter etapes 1-4 jusqua a avoir le plus petit cout possible


lorsque on travaille avec des couches il est essentiel de savori combien de couches et combien de noueds nous avons

on commence a compter a partir du premier "hidden layer" (couche/s entre les couches input et ouput qui s'occupent de faire
touts les calculs) jusqua la couche output

on represente les couches (layers en anglais) avec la variable L 

on exclu la couche input car ce sont seulement des donnes crues et non pas des calculs


pour compter les nombres de noeuds ont utilise la variable n avec un exposant, cet exposant va etre l'index de la couche

immaginons que nous avons un reseaux avec L= 3 (couche input + 2 hidden layers + couche output)

la couche input elle a 2 noeuds, les hiddens layer elles ont 3 et la couche output 1 seul

donc n ^ 0 = 2, n ^ 1 = 3, n ^ 2 = 3, n ^ 3 = 1


pour calculer le nombre de poids entre une couche et une autre on a juste a faire n ^ l * n ^ l - 1

mais c'est tres facile de se perdre lorsque on a bcp de noeuds et de poids, donc la meilleure facon de calculer ca c'est d'
utiliser des matrices

ce qui'il faut faire:

representer chaque couche comme une matrice 
representer chaque set de poids entre deux couches comme une matrices
faire le calcul des matrices 


Qu'est ce que le Feed forward?

-> le feed forward est le fait de calculer la valeur de chaque noed dans chaque couche, jusqu'a arriver a la couche output



EXPLORATION DES donnees

le data set, decrit les caracteristiques de celulles cancereuses

il y a 32 colonnes et 569 lignes


dans le dataset donne dans le sujet on a pas de headers pour les colonnes, suite a une recherche sur UCI machine learning repository
on a trouve le dataset: breast cancer Wisconsin (diagnostic) qui correpond au dataset donne dans le sujet

a partir de la on a donc pu trouver les noms des colonnes dans le dataset

qui sont:

![alt text](<immages/Screenshot 2026-07-13 144908.png>) 
![alt text](<immages/Screenshot 2026-07-13 144923.png>) 
![alt text](<immages/Screenshot 2026-07-13 144934.png>) 
![alt text](<immages/Screenshot 2026-07-13 144947.png>)


ici on peut donc voir que chaque ligne represente une cellule tumorale observe au microscope et les colonnes contiennent des
mesures extraites a partir des immages des cellules

le but du dataset est de predire si la cellule est B (begnin) ou M (malign)

EXPLORATION DES COLONNES:

1) la premiere colonne represente l'ID de la cellule (cette information ne va pas etre tres utile a l'entrainement du modele car
elle possede aucune information medicale)

2) cette colonne represente le diagnostic de la cellule (B/M), c'est donc la variable que l'on veut predire

3) a partir de cette colonne jusqu'a la colonne 32 nous avons les caracteristiqued des cellules, cependant nous avons que 10
caracteristiques reelles calcules pour chaque noyau de cellule

qui sont:

radius,
texture,
perimeter,
area,
smoothness,
compactness,
concavity,
concave points,
symmetry,
fractal dimension.

mais ces 10 caracteristiques sont calcules chacune 3 fois pour calculer:

la moyenne (mean), l'erreur standard (standard error) et la valeur maximale (worst)

donc 10 * 3 = 30

le sujet nous dit que on ne devrait pas donner les donnees brutes du dataset au model, ceci s'expliques par les ecarts entre les mesures
qui vont causer un mauvais entrainement pour regler ce probleme on doit proceder a une normalisation des donnes pour eviter tout type d'ecart
trop important, pour ce type de dataset une normalisation de type z-score est meilleure car on connait pas les extremees biologiques pour un min/max

et aussi on a des donnes avec des distributions basees sur les moyennes et les dispersions



pour eviter le data leakage, nous devions pas appliquer la normalisation sur tout le dataset mais seulement sur la partie train, sinon le modele va avoir
acces a des informations qu'il ne devrait pas avoir pendant la phase de training ce qui va causer du data leakage


QU'EST CE QUE LA FONCTION SOFTMAX?

la fonction softmax est une fonction mathematique utilise a la sortie d'un reseau de neurones pour produire une serie de probabilite dont la somme font 1,
le reseaux en sortie produit des nombres que on peut pas trop interpreter (positifs, negatifs, a virgule etc etc...), pour pouvoir les interpreter nous allons
appliquer la fonction softmax

Exemple simple

Imaginons un réseau qui doit reconnaître un animal parmi 3 classes :

Chat
Chien
Oiseau

Avant le softmax, le réseau produit :

Classe	Score (logit)
Chat	2.3
Chien	1.1
Oiseau	-0.7

Ces nombres ne veulent pas dire grand-chose.

Après le softmax, on obtient :

Classe	Probabilité
Chat	74 %
Chien	22 %
Oiseau	4 %

Maintenant les résultats sont interprétables.




comme nous devons appliquer la fonction softmax sur le layer output, on ne peut pas avoir un seul neurone en sortie car une probabilite sur un seul element
n'as pas de sens, donc ce qu'il faut faire c'est de faire en sorte a ce que le layer d'output soit compose de 2 neurones

donc comme on ne peut pas avoir un encodage binaire classique pour la colonne des diagnostic (B = 0 && M = 1), car notre output avec softmax est un vecteur
de dimensions 2 [0.73, 0.27] par exemple, et la fonction de cout a besoin d'un vecteur a 2 dimensions alle aussi, donc on peut pas lui donner un 0 ou un 1
donc nous devons les encoder d'une autre maniere a fin de faire comprendre nos donnes a notre modele, c'est ici que on va appliquer le one-hot encoding



QU'EST CE QUE LE ONE-HOT ENCODING?

c'est la solution a nos soucis, a la place de encoder la classe comme un entier (0 ou 1) on la transforme en vecteur, avec tant de dimensions que de classes,
ou une seule composante vaut 1 et le reste valent 0

par exemple:

B -> [1, 0]
M -> [0, 1]


EXPLORATION DES DONNES

nombre des M  = 212, nombre des B = 357,

suite a un .describe() nous pouvons constater un grand ecart entre certaines donnes, ce qui justifie la normalisation

area1 mean = 654.8 | smoothness1 mean = 0.09 => ecart trop grand, ca va fausser les entrainements

features significatives:

radius1
perimeter1
area1
compactness1
concavity1
concave_points1


a revoir:
texture1
smoothness1
symmetry1
fractal_dimension1
