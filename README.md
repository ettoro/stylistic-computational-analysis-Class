# COLI analysis  
## Identifying Characters' Lines in Original and Translated Plays: The case of Golden and Horan's *Class*  

The main task of this project is to solve a multi-class classification problem in order to correctly identify the five characters (plus the stage directions) of the play *Class* through the stylometric analysis of each character's line. The classification is performed both for the original and for the translation in order to understand whether the same differentiation between characters is maintained or not. The classification is made using three different groups: six classes (one for each character), three classes (male characters, female characters, stage directions), three classes (adult characters, children, stage directions).

### Subfolders

* __Preprocessing__:
  * Set-up and package installations
  * Functions to obtain necessary features
  * Splitting of training and test sets
  * Data standardization
  * Data augmentation
* __Test (models trained)__: 
  * LogReg
  * Random Forest
  * knn
  * svc
  * neural network
* __Graphic Visualisations__:
  * PCA and t-SNE (2D)
  * box plots 
  * pairplots
  
### Data

The corpus is not publicly available due to copyright, additional information about the play can be found [here](https://www.nickhernbooks.co.uk/class)


### Full code (Colab Notebooks)

| [English](https://colab.research.google.com/drive/1Gd9Wec9k9CbpPfeQbj7OFNjCxrc48eCQ?usp=sharing) | [Italian](https://colab.research.google.com/drive/1WbrqXStHXht6PF1kvQktL21C-_WaWXZy?usp=sharing) |
|----|----|
|All characters|All characters|
|Male/Female/None|Male/Female/None|
|Adults/Children/None|Adults/Children/None|
