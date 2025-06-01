from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].apply(lambda i: iris.target_names[i])

df.head()
df.describe()
df.info()
df.isnull().sum()  # Check for missing values
