import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score

df_drug = pd.read_csv("drug200.csv")

label_encoder = LabelEncoder()

categorical_features = [feature for feature in df_drug.columns if df_drug[feature].dtypes == 'O']
for feature in categorical_features:
    df_drug[feature]=label_encoder.fit_transform(df_drug[feature])
    
X = df_drug.drop("Drug", axis=1)
y = df_drug["Drug"]

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

kfold = KFold(random_state=42, shuffle=True)
cv_results = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
print(cv_results.mean(), cv_results.std())

import pickle

pickle_file = open('model.pkl', 'ab')
pickle.dump(model, pickle_file)                     
pickle_file.close()