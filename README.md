# optimisation-equitable

## Binôme : 
**Ghiles OUHENIA** https://github.com/Ghiles-OUHENIA

## Résumé du projet : 

Le but de ce projet est de résoudre différents problèmes de partage de bien entre n agents de façon équitable.

Pour cela, il suffit de trouver une distribution des biens x maximisant $f(x) = \sum_{i} w_i * z_{(i)}(x)$. Comme $f(x)$ n’est pas une fonction linéaire, il faut donc la linéariser en procédant par plusieurs étapes.

Ce projet comprend une partie théorique dans laquelle un programme est construit afin de résoudre le problème du partage équilibré des actifs, qui sera adapté à trois problèmes concrets: 

- Partage équitable de `P` produits sur les `N` agents.
- Sélection multicritère de `P` projets parmis `N`.
- Recherche d’un chemin robuste dans un graphe.

