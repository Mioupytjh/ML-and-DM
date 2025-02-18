
import pandas as pd



df = pd.read_csv("./data/diamond.csv")

print(df.head())
print(df.describe().to_latex())