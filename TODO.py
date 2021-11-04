TODO = 1


# TODO --------------------------------------------------------------

#-------------------------------------------------------------------------
# Mission 1 : creer training set, ajouter au buffer, train la reward func, l'utiliser dans la outer loop, faut que train correctement l'agent et la reward #
#### DONE ####

# steps restants:
# training curves plot et visualisation
# vérifier avec ahmed les questions en dessous
# faire du versionning du code, checkpoint là ou ça marche


# trucs pas sur:
# problem with multi CPU sharing ? When the update should be made?
# check avec ahmed si les rewards sont bien computées
# architecture reward func
# infer more negative pairs with predicate manipulation
# when to put reward func model in gpu or cpu
# HER logique de code de computation des rewards checker que je fais bien
# demander a ahmed des données de training de GANGSTR avec plein de seeds

#-------------------------------------------------------------------------

#-------------------------------------------------------------------------

## Mission 2 : coder le demonstrator hard coced

# steps restants:
# tester a la mano chacun des scénarios pour voir s'il n'y a pas de bugs
# lancer un test global sur tout les goals et calculer le SR
# reflechir a comment faire du showing à partir de ça (travailler sur les trajectoire, notamment go XY et go Z doivent etre modifiés)

# trucs pas surs:

#-------------------------------------------------------------------------

#-------------------------------------------------------------------------


## Mission 3 : créer module d'encoding de demos, sample demos dans l'outer loop, encode demo, play goal conditioned policy,
## create feedback to construct reward dataset, train reward module, checker si l'agent apprend bien #

# steps restants:
# créer module d'encoding de demos > offline tester des archi avec un dataset
# insérer les demos dans l'outer loop
# modifier archi RL pour prendre en compte les nouveaux goals
# decider sur le feedback (combien? il en faut bcp)

# trucs pas surs:
# quel feedback est approprié?
# quel goal space?


#-------------------------------------------------------------------------

#-------------------------------------------------------------------------


