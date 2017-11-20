import numpy as np
import pandas as pd
from sklearn import ensemble


class Model:
	def __init__(self, model):
		self.model = model

	def train(self, X, y):
		self.model.fit(X, y)

	def predict(self, test_X):
		return self.model.predict(test_X)


def process_data(df):
	df["Sex"] = df["Sex"].apply(lambda x: x == "male")
	df["Fare"] = df["Fare"].apply(lambda x: 0 if np.isnan(x) else x)
	df["Age"] = df["Age"].apply(lambda x: -1 if np.isnan(x) else x)
	df["infant"] = df["Age"].apply(lambda x: x < 3 and x >= 0)
	df["child"] = df["Age"].apply(lambda x: x < 10 and x >= 3)
	df["teen"] = df["Age"].apply(lambda x: x < 18 and x >= 10)
	df["senior"] = df["Age"].apply(lambda x: x >= 50 )
	df["adult"] = df["Age"].apply(lambda x: x < 50 and x >= 18 )
	df["Embark_C"] = df["Embarked"].apply(lambda x: x == "C")
	df["Embark_S"] = df["Embarked"].apply(lambda x: x == "S")
	df["Embark_Q"] = df["Embarked"].apply(lambda x: x == "Q")
	df["family_size"] = df["SibSp"] + df["Parch"]

	return df

submit = True

features = ["SibSp","Fare", "Sex", "Pclass", "Age", 
"Embark_C", "Embark_S", "Embark_Q", "infant", "child", "senior", "adult", "teen", "Parch", "family_size"]

df = pd.read_csv("train.csv", doublequote=True)
df = process_data(df)
# print len(df.dropna(axis=1, how='any').index)
# df = df.dropna(axis=0, how='any')
if not submit:
	test_size = 250
	_ = df[:-test_size]
	test_df = df[-test_size:]
	df = _
else:
	test_df = process_data(pd.read_csv("test.csv", doublequote=True))

train_X, train_y = df.as_matrix(features), df["Survived"]

model = Model(ensemble.GradientBoostingClassifier(max_depth=3))
model.train(train_X, train_y)

predictions = model.predict(test_df.as_matrix(features))

if submit:
	test_df["Survived"] = predictions
	test_df.to_csv("predict.csv", columns = ["PassengerId", "Survived"], index=False)
else:
	correct = 0
	for i, p in enumerate(test_df["Survived"].values):
		if predictions[i] == p:
			correct += 1
	print correct, len(predictions)
