# Classifiers
# Project description.
Educational project with a main focus on using a variety of classifiers and to improve code writing and learn theory behind classifiers.

Project consists of code.py, data folder (heart  and vowel datasets), gitignore and read.me files.

# Goals.
Explore binary form of classification, that helps to model data where response could take one of two states. And explore a multi-class classification where response could take any number of states.
Explore how to fit and evaluate models.

# Used libraries.
I used libraries in Python such as scikit-learn for machine learning models, and Pandas to import data as data frames.

# Classifiers that were used in the project.
Logistic Regression is a type of Generalised Linear Model (GLM) that uses a logistic function to model a binary variable based on any kind of independent variable.

Support Vector Machines (SVMs) are types of classification algorithms that are more flexible. They can do linear classification, but can use other non-linear basis functions. In the project I use a linear classifier to fit a hyperplane that separates the data into two classes.

Random Forests (RF) are an ensemble learning method that fit multiple Decision Trees on subsets of the data and average the results.

Neural Networks (NN) are machine learning algorithms that involve firing many hidden layers used to represent neurons that are connected with synaptic activation functions. I used sklearn for consistency, however libraries such as Tensorflow and Keras are more suited to fitting and customising neural networks.

In the train function to fit a binary and multi-class classifiers I used models from sklearn.

In the test function I use the predict method and score method to get the mean prediction accuracy.

# Datasets.
I apply models on data regarding coronary heart disease (CHD) in South Africa, that contains different variables such as tobacco usage, family history, ldl cholesterol levels, alcohol usage, obesity and more. And Vowel data used to determine which one of eleven vowel sounds were spoken. Datasets were taken from. The open source Elements of Statistical Learning.

# Results.
As expected result I should compare the predictive accuracy across the models. This will tell us which one is the most accurate for this specific dataset.
