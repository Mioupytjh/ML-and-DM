import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
import numpy.linalg as la
import matplotlib.pyplot as plt

data: pd.DataFrame = pd.read_csv("data/diamond.csv")

Data = data.copy()

# Check for null values in the data set
null_count = Data.isnull().sum().sum()
print(f"Count null values in the data set: {null_count}")

# One-out-of K encoding
Data = pd.get_dummies(Data)


# Normalizing / Centering the data
Normalized_data = (Data - Data.mean()) / Data.std()
# print(Normalized_data)

# Perform PCA using sklearn
pca = PCA(n_components=2)  # Example: reduce to 2 components
principalComponents = pca.fit_transform(Normalized_data)

# print(data)
def plot_data(attribute):
    global data
    global principalComponents
    
    # Saves the color value of each data point
    label = data[attribute]

    # Create a DataFrame with the principal components
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

    principalDf[attribute] = label

    unique_colors = principalDf[attribute].unique()
    color_map = {color: idx for idx, color in enumerate(unique_colors)}

    colors = principalDf[attribute].map(color_map)

    print(principalDf)


    plt.figure(figsize=(10, 7))

    components = pca.components_
    feature_names = pca.feature_names_in_  # Original feature names

    # Scale for better visualization
    scaling_factor = 5  # Adjust for better visualization on the plot
    
    scatter = plt.scatter(
        principalDf['principal component 1'],
        principalDf['principal component 2'],
        c=colors, cmap='tab10', alpha=0.6
    )
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=color,
                                markerfacecolor=plt.cm.tab10(idx / len(unique_colors)), markersize=8)
                    for color, idx in color_map.items()]

    plt.legend(handles=legend_handles, title=f"Diamond {attribute.title()}")
    plt.title('PCA Plot of Diamond Dataset')
    plt.xlabel('principal component 1')
    plt.ylabel('principal component 2')
    plt.grid(True)
    for component in components.T[:2]:
        plt.arrow(
            0, 0,  # Starting point of the arrow (origin)
            component[0] * scaling_factor,  # Scaled x-component
            component[1] * scaling_factor,  # Scaled y-component
            color='red', width=0.01, head_width=0.15, alpha=0.8
        )

    plt.show()

plot_data("color")
plot_data("clarity")
plot_data("cut")

pca = PCA()  # Example: reduce to 2 components
principalComponents = pca.fit_transform(Normalized_data)

vals = pca.explained_variance_ratio_
vals_acc = np.add.accumulate(vals) # y
vals_num = np.arange(1, pca.explained_variance_.shape[0] + 1)  # Number of components

plt.figure(figsize=(12, 7))
plt.plot(vals_num, vals_acc, marker='o', linestyle='-', color='b', label='Cumulative Explained Variance')
plt.title('Explained Variance as a Function of Number of PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.legend()
plt.grid(True)
plt.show()