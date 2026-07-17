import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import os

# load dataset
df = pd.read_csv('archive/Training.csv')
df.columns = df.columns.str.strip()

# fix NaN values
df = df.fillna(0)

X = df.drop('prognosis', axis=1)
y = df['prognosis']

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn.predict(X_test))
print(f"KNN Accuracy: {knn_acc*100:.1f}%")

# train SVM
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)
svm_acc = accuracy_score(y_test, svm.predict(X_test))
print(f"SVM Accuracy: {svm_acc*100:.1f}%")

# save models
os.makedirs('models', exist_ok=True)
pickle.dump((knn, X.columns.tolist()), open('models/knn_model.pkl', 'wb'))
pickle.dump((svm, X.columns.tolist()), open('models/svm_model.pkl', 'wb'))

print("Models saved successfully!")