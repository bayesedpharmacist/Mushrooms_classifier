# Shrooms_project

This project comprises building a deep learning model to classify wild mushrooms.
The images were from Kaggle dataset. Many species in the original dataset that were polypores, i.e. grow on trees, were removed from the training dataset.
The final dataset comprises 214 species within 35 families that are common to Europe. Because of the dataset being polluted with images
from different stages of growth, meaning growing to decaying mushrooms,
I choose to include description data in form of simple text that describes the young mushroom appearance.
These are features characteristic for the family and the species:

1. Form of the hymenium.
2. Color of the hymenium in young exemplar.
3. Presence or lack thereof of ring on the stem.
4. Color of the cap in young exemplar.
5. Presence or lack thereof of scales on the cap.

These five features I annotated by myself, since I have university level of botany knowledge.
Main part of the features engineering was Categorical Naive Bayes encoding. Simply, mushrooms belonging to family Boletaceae
have pores, whereas mushrooms belonging to family Amanitaceae have gills. That is mushrooms from family Boletaceae have
98% probability of having pores and virtually 0% probability of having gills, and vice versa for family Amanitaceae. 
The probabilities themselves were used as one input to the model. Additionally,
these features were One Hot encoded as a unique signature input to the model. 
The final model has been trained on CNN for the input images,
and DNN for the input text description combined in one (Keras, Tensorflow). Until now the model has accuracy of ~75%.
Data on the edibility of these mushrooms is exclusively omitted. The classifier is for educational and
orientational purposes only.

### How to use
The model has been packaged as one file with PyInstaller in Docker manylinux environment. It works on Linux distributions 
after 2018. The executable file is called 'shroom' and can be downloaded from here: 
https://drive.google.com/file/d/16aHqIipZoYw343Rz6xNV1kiblwHvls76/view?usp=sharing

### Ongoing work and future perspectives
Currently, I am working on improving the model by adding custom layers to force hierarchical classification, i.e. first 
classifying the family and then the species. 

PS. The code in this repository, naturally, cannot be executed without the training dataset. 