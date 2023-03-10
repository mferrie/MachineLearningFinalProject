# Logistic regression practice
print("Importing libraries...")
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

print("Loading dataset...")
digits = load_digits()

print("Orgainizing training data...")
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

print("Training model...")
model = LogisticRegression(max_iter=100000)
model.fit(x_train, y_train)

y_predicted = model.predict(x_test)

model.score(x_test,y_test)

cm = confusion_matrix(y_test, y_predicted)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel("Prediction")
plt.ylabel("Reality")
plt.show()
