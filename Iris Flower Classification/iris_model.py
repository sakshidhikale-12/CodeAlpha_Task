
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv("Iris.csv")

print("First 5 rows:")
print(df.head())

print("\n dataset Info:")
print(df.info())

print("\n statistical summary:")
print(df.describe())

print("\n species Count:")
print(df['Species'].value_counts())


X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\n model accuracy:", accuracy)

sns.pairplot(df, hue='Species')

plt.show()
