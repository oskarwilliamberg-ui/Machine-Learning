import csv
with open("training_data.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

X = []
y = []

for row in rows:
    label = row[0]
    features = row[1:]
    features = [float(v) for v in features]
    X.append(features)
    y.append(label)

print(len(X), len(y))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
print(accuracy)

import joblib
joblib.dump(model, "gesture_model.joblib")