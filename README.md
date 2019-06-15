| Name | Date |
|:-------|:---------------|
|Mayara Alves | 2019/06/14|

-----

### Resources
Your repository should include the following:

- Python script for your analysis
- Results figure/saved file
- Dockerfile for your experiment
- runtime-instructions in a file named RUNME.md

-----

## Research Question

Departing from 13 wine components is it possible to predict the wines classes?

### Abstract
The purpose of this project is to make an application of Big Data concepts like Machine Learning, Data Visualization and others. In this case was utilized a toy dataset with wine data to predict the classes of those wines. To make it possible and get in the final results was utilized an algoritm of Logistic Regression, with a great accuracy, making the solution reliable.
4 sentence longer explanation about your research question. Include:

### Introduction

The dataset is an UCI Wine Data with thirteen components of wines, (Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins, Color intensity, Hue, OD280/OD315 of diluted wines and Proline), from the same region in Italy, but from three different cultivars. It’s a toy dataset that can be used to training how to predict wine classes.


### Methods

The method utilized to predict the wine classes was Logistic Regression, an estimator from sklearn, and it was chosen because it's indicated to classification purposes.

Brief (no more than 1-2 paragraph) description about how you decided to approach solving it. Include:

- pseudocode for this method (either created by you or cited from somewhere else)
- why you chose this method

### Results

In the results we have an accuracy score of 98% which indicates a positive performance for the method. 
For the confusion matrix we have the following:

                | [19 | 0 | 0 |
                |  0  |27 | 1 |
                |  0  | 0 | 16]|

Where the 'Class 0' was correctly predicted 19 times, 'Class 1' have 27 correct data and 1 miss and in the 'Class 2' we have 16 correct predictions. And our f1-score is 0.9838, affirming that the model is working great for it's purpose.


                   


### Discussion
Brief (no more than 1-2 paragraph) description about what you did. Include:

- interpretation of whether your method "solved" the problem
- suggested next step that could make it better.

### References
https://archive.ics.uci.edu/ml/datasets/Wine
https://seaborn.pydata.org/examples/index.html
https://cce-bigdataintro-1160.github.io/CEBD-1160-spring-2019/8-ml.html
http://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_iris_knn.html


-------
