from sklearn.cluster import KMeans # This line imports the KMeans class from the scikit-learn library, which is used for clustering data.

import matplotlib.pyplot as plt # This imports the pyplot module from the Matplotlib library, which is used for creating plots and visualizations.

import pandas as pandas #  capability to work with spreadsheet-like data enabling fast loading, aligning, manipulating, and merging

from sklearn.preprocessing import LabelEncoder # This imports the LabelEncoder class from scikit-learn, which is used for encoding categorical features into numerical values

# Store the Iris data from Iris.csv into a python dataframe.
df = pandas.read_csv("Iris.csv")
X = df[['SepalLengthCm', 'SepalWidthCm']]

# Insert the species into a dataframe named y.
y = df['Species']
print(y)

# Label Encoding to turn values into numerical.
le = LabelEncoder()
yEncoded = le.fit_transform(y)
print(yEncoded)

# This code creates a scatter plot using the SepalLengthCm and SepalWidthCm columns from X as the x and y coordinates, respectively. The color of each point on the scatter plot is determined by the corresponding value in yEncoded. It then sets the x and y axis labels and saves the plot as "plot.png".
plt.scatter(X['SepalLengthCm'],X['SepalWidthCm'], c = yEncoded, cmap = 'gist_rainbow')
plt.xlabel("Sepal Length", fontsize =18)
plt.ylabel("Sepal Width", fontsize = 18)
plt.savefig("plot.png")

# This creates a KMeans object named km with 3 clusters and sets the random state to 0 for reproducibility.
km = KMeans(n_clusters = 3, random_state = 0)

# This fits the KMeans model to the data X, clustering the data points into three clusters based on the specified parameters.
km.fit(X)

# This code predicts the cluster labels for the data points in X using the trained KMeans model and stores them in new_labels. It then creates a scatter plot similar to the previous one but with the predicted cluster labels as colors. Finally, it sets the x and y axis labels, adds a title to the plot, and saves it as "Prediction.png".
new_labels = km.labels_
plt.scatter(X['SepalLengthCm'],X['SepalWidthCm'], c = new_labels, cmap = 'gist_rainbow')
plt.xlabel('Sepal Length', fontsize = 18)
plt.ylabel('Sepal Width', fontsize = 18)
plt.title("Predicted", fontsize = 18)
plt.savefig("Prediction.png")