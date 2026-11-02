## Projet étude: Requête des services IA

Fichier principal: `seg_cloth.ipynb`

Tester le service Hugging Face. Mon appel a renvoyé une erreur 402 caril faut des crédits. J’ai vérifié que le modèle pouvait tourner directement sur mon ordinateur (`test_token.py`).

Ensuite, j’ai complété le code de la page Hugging Face avec des affichages à chaque étape (`segformer_b3.py`), en résumer le modèle réduit l’image, calcule 18 scores par pixel, garde le meilleur, puis agrandit le résultat.

Puis, j’ai complété le notebook du précédent data scientist et ajouté une simulation de la réponse du service (`ancien_simulation_api.ipynb`). Pour utiliser le vrai service, il suffit de passer utiliser_api_reelle à True.

## Le notebook principal:

J’ai gardé le code de Hugging Face du modèle tel quel, sans le modifier, pour que mes résultats correspondent à ce que fait vraiment le modèle. J’ai seulement ajouté `torch.no_grad()` au moment de la prédiction car je n’entraine pas le modèle, donc il n’a pas besoin de préparer l’apprentissage, ce qui va plus vite et consomme moins de memoire. J’ai converti ensuite chaque photo en RGB pour qu’elle ait toujours trois canaux de couleur, même si un fichier PNG contient de la transparence.

Pour mesurer la *qualité*, j’ai choisi l’IoU plutôt que la simple précision, parce que le fond représente la majorité des pixels et j’ai ajouté le Dice parce que c’est l’autre score standard en segmentation. 
<br>Le calcule  prenait les scores en additionnant les pixels de toutes les images avant de diviser, plutôt qu’en faisant la moyenne image par image, car une classe rare comme l’écharppe donnerait sinon des chiffres très instables. Pour, les classes absentes d’un jeu d’images ne comptent pas dans la moyenne, pour ne pas la fausser avec de faux zéros.

J’ai découpé les 50 images en 60 %, 20 % et 20 % (30, 10 et 10), des proportions classiques. 
<br>Le tirage est au hasard, avec une graine fixée à 42 pour retrouver exactement le même découpage à chaque exécution (le chiffre 42 est arbitraire). Comme le modèle est déjà entraîné, le jeu d’entraînement ne sert pas ici, mais je l’ai gardé pour respecter la méthode demandée, et pour un éventuel fine-tuning. 
<br>La validation sert aux réglages et le test au score final.

Pour les visualisations, j’ai fixé la même échelle de couleurs pour toutes les images (une couleur par classe, de 0 à 17). 
<br>Sans cela, une même classe aurait des couleurs différentes d’une photo à l’autre et la légende serait fausse. J’ai enregistré les 50 comparaisons mais n’en affiche que 5 dans le notebook, pour ne pas le surcharger. 

Pour l’analyse, j’ai regroupé la gauche et la droite (chaussures, jambes, bras) parce que j'ai pensé que cette distinction n’a aucune valeur pour une marque, et pour vérifier mon hypothèse que le modèle trouve la bonne zone mais se trompe de côté. 
<br>Pour le graphique des vêtements présents, je compte un vêtement comme présent à partir de 0,2 % de la surface de l’image, afin d’ignorer quelques pixels isolés. Ce seuil est mon choix personelle, et il influence surtout les petits objets (je l'avais tester avec plusieurs autre seuil mais celui la était le mieux). Et je n’y garde que les vêtements et accessoires, car compter le visage n’aurait pas de sens.

Pour le coût, je pars du scénario de Sophia de l'énoncé: 500 000 images en 30 jours, soit 0,19 image à traiter par seconde. 
<br>Donc, j'ai mesurer le temps de traitement d’une image sur 11 images, en retirant la première, plus lente parce que le modèle s’initialise. J’en ai déduis le nombre d’instances nécessaires, puis arrondi à l’entier supérieur puisqu’on ne loue pas une demi-machine. 
<br>Le prix de 0,067$ de l’heure est celui d’une petite instance CPU de Hugging Face, que j’ai prise comme référence, c’est une hypothèse, car le prix exact du service pour ce modèle n’est pas publié. 
<br>Enfin, la latence est celle de mon ordinateur et elle varie un peu d’une mesure à l’autre, donc le coût à la consommation est une estimation d’environ 10 à 16$, et de 48$ avec une instance dédiée.