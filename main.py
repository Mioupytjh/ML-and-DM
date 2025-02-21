import numpy as np
import pandas as pd
import importlib_resources
import xlrd

filename = importlib_resources.files("DiamondData").joinpath("data/DiamondData.xls")
print("\nLocation of the iris.xls file: {}".format(filename))

doc = xlrd.open_workbook(filename).sheet_by_index(0)

# Extract data from the first column, including the header
raw_values = doc.col_values(0)  # Includes header row

spread_data = [row.split(',') for row in raw_values]

df = pd.DataFrame(spread_data)

for i in range(10):
    doc.put_cell(0, i, xlrd.XL_CELL_TEXT, f'Feature_{i+1}', 0)  # Set new headers from the split values
    for j, value in enumerate(df.iloc[:, i]):
        doc.put_cell(j, i, xlrd.XL_CELL_TEXT, value, 0)

attributeNames = doc.row_values(rowx=0, start_colx=0, end_colx=9)

classLabels = doc.col_values(0, 1, 53941)  # check out help(doc.col_values)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(len(classNames))))

# Extract vector y, convert to NumPy array
y = np.array([classDict[value] for value in classLabels])

# Preallocate memory, then extract data to matrix X
X = np.empty((classLabels, 9))
for i in range(9):
    X[:, i] = np.array(doc.col_values(i, 1, 151)).T

# Compute values of N, M and C.
N = len(y) #Number of observations
M = len(attributeNames) - 1 #Number of features (excluding the class label)
C = len(classNames) #number of classes

print(X.shape)