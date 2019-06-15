import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

wine_df = pd.read_csv('wine.data', sep=',', header=0)
wine_df.columns=['Class','Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of Ash', 'Magnesium', 'Total Phenols',
                'Flavanoids', 'Nonflavanoid Phenols', 'Proanthocyanins', 'Color Intensity', 'Hue',
                'OD Diluted Wines', 'Proline']
#Drop column class
wine_df2 = wine_df.drop(['Class'], axis=1)
os.makedirs('plots', exist_ok=True)

## Print dataset
print(wine_df2.to_string())
print()

# Print summary statistics
print(wine_df2.describe().to_string())

#Printing the pairplot to see the comportament between the features
sns.pairplot(wine_df, hue= 'Class', diag_kind='hist')
plt.savefig('plots/pairplot.png')
plt.show()


# Printing heatmap to see the correlation between the features
os.makedirs('plots/wine-seaborn_heatmap', exist_ok=True)
sns.set()
fig, ax=plt.subplots(figsize=(12,12))
sns.heatmap(wine_df2.corr(),annot=True, cmap='autumn')
ax.set_xticklabels(wine_df2.columns, rotation=45)
ax.set_yticklabels(wine_df2.columns, rotation=45)
plt.savefig('plots/heatmap.png')
plt.show()

#Seeing Alcohol, Ash and Color Intensity among the Classes
plt.style.use("ggplot")
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
axes.grid(axis='y', alpha=0.5)
axes.scatter(wine_df['Alcohol'], wine_df['Ash'])
axes.scatter(wine_df['Alcohol'], wine_df['Color Intensity'])
axes.scatter(wine_df['Alcohol'], wine_df['Class'])
axes.set_title(f'Alcohol comparisons')
axes.legend(['Ash', 'Color Intensity', 'Class'])
plt.savefig('plots/alcohol.ash.colorint.png')
plt.show()


#Starting LogisticRegression
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

wine = load_wine()
feature_names = wine.feature_names
X = wine.data
y = wine.target

#Splitting features and target datasets into: train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=4000)
lr.fit(X_train, y_train)

# Predicting the results for our test dataset
predicted_values = lr.predict(X_test)

# Printing the residuals: difference between real and predicted
for (real, predicted) in list(zip(y_test, predicted_values)):
   print(f'Value: {real}, pred: {predicted} {"is different" if real != predicted else ""}')

# Printing accuracy score(mean accuracy) from 0 - 1
print(f'Accuracy score is {lr.score(X_test, y_test):.2f}/1 \n')

# Printing the classification report
from sklearn.metrics import classification_report, confusion_matrix, f1_score
print('Classification Report')
print(classification_report(y_test, predicted_values))

# Printing the classification confusion matrix (diagonal is true)
print('Confusion Matrix')
print(confusion_matrix(y_test, predicted_values))
print('Overall f1-score')
print(f1_score(y_test, predicted_values, average="macro"))



# #Printing the colormap
# from matplotlib.colors import ListedColormap
# from sklearn import neighbors, datasets
# # Create color maps for 3-class classification problem, as with wine
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#
# wine = load_wine()
# X = wine.data[:, :2]
# y = wine.target
#
# knn = neighbors.KNeighborsClassifier(n_neighbors=1)
# knn.fit(X, y)
## x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
# y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
# xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
#                      np.linspace(y_min, y_max, 100))
# Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.figure()
# plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
## # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
# plt.xlabel('Alcohol')
# plt.ylabel('Color Intensity')
# plt.axis('tight')
#plt.savefig('plots/colormap.png')
# plt.legend()

#Plotting predicted vs Real
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette="inferno")
residuals = y_test - predicted_values
sns.scatterplot(y_test, residuals)
plt.plot([0, 5], [0, 5], '--')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.savefig('plots/project.png')
plt.show()

#Comparing Models
         
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")


# Scaling data (KNeighbors methods do not scale automatically!)
scaler = StandardScaler()
scaler.fit(X)
scaled_features = scaler.transform(X)

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.35, random_state=1)

f1_scores = []
error_rate = []

# Creating one model for each n neighbors, predicting and storing the result in an array
for Estimator in [KNeighborsClassifier, LogisticRegression, SGDClassifier, GaussianNB]:
    estimator = Estimator()
    estimator.fit(X_train, y_train)
    y_predicted = estimator.predict(X_test)
    f1 = f1_score(y_test, y_predicted, average="macro")
    error = np.mean(y_predicted != y_test)
    f1_scores.append(f1)
    error_rate.append(error)
    print(f'For {type(estimator)} the f1-score is {f1} and error rate is {error}')

# Plotting results
plt.plot(f1_scores, color='green', label='f1 score', linestyle='--')
plt.plot(error_rate, color='red', label='error rate', linestyle='--')
plt.xlabel('n neighbors parameter')
plt.ylabel('f1_score/error_rate')
plt.xticks(np.arange(4), ['KNeighborsClassifier', 'LogisticRegression', 'SGDClassifier', 'GaussianNB'])
plt.legend()
plt.savefig('plots/models f1score and error.png')
plt.show()
