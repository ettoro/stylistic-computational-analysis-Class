# COLI-exam  
## Identifying Characters' Lines in Original and Translated Plays: The case of Golden and Horan's *Class*  

The main task of this project is to solve a multi-class classification problem in order to correctly identify the five characters (plus the stage directions) of the play *Class* through the stylometric analysis of each character's line. The classification is performed both for the original and for the translation in order to understand whether the same differentiation between characters is maintained or not.

### Subfolders

* __Preprocessing__:
  * Set-up and package installations
  * Functions to obtain a df with the necessary features for classification starting from a tsv file where each line contains the character's name and its dialogue line
  * Splitting of the train and test sets
  * Data normalization
  * Data augmentation to obtain a balanced dataset
* __Test (models trained)__: 
  * LogReg
  * Random Forest
  * knn
  * svc
  * neural network

### Data

The corpus is not publicly available due to copyright, additional information about the play can be found [here](https://www.nickhernbooks.co.uk/class)
