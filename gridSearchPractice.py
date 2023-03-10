from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import pandas as pd

print("Loading Dataset")
iris = datasets.load_iris()

clf = GridSearchCV(svm.SVC(gamma='auto'), {
    'C' : [i for i in range(1, 101)],
    'kernel' : ['rbf', 'linear', 'sigmoid', 'poly']
    })
print("Fitting model...")
clf.fit(iris.data, iris.target)

df = pd.DataFrame(clf.cv_results_)

print("Here's the parameters of the best model I found.")

df = df[['param_C', 'param_kernel', 'mean_test_score']].sort_values(by='mean_test_score', ascending=False)
print(df.head())

input("Press ENTER to exit...")
