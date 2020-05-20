# Data analisys on  network intrusion detection

## Data:

* network_intrusion_detection : http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html


## School project:

* Université de Technologie de Compiègne (UTC) France

## courses: 

* SY09: Data analysis and  Data-Mining


## Remarque 
### Fichier pretraitement.py
#### Explication des variables : 
newdata : Les données total qui provient du fichier **kddcup.data_10_percent.csv**.
newdata_test : Les données total qui provient du fichier **corrected.csv**. (Nous n'utilisons pas dans l'apprentissage)
X_train_scaled : Training data qui contient les 90% données séparées de newdata; scaled; sans "class" 
X_test_scaled : Test(Validation) data qui contient les 10% données séparées de newdata aussi; scaled; sans "class"
Y_train : Liste de colonne "class" correspondante au training data **X_train_scaled** 
Y_test : Liste de colonne "class" correspondante au test data **X_test_scaled** 


Les deux variables suivantes sont des données non scaled: 
X_train : Training data qui contient les 90% données séparées de newdata; sans "class" 
X_test : Test(Validation) data qui contient les 10% données séparées de newdata aussi; sans "class"



## Students:

* Sami Jaghouar
* Yinong Qiu
* Loic Yvinec
 
