import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import  RandomForestRegressor
from sklearn.model_selection import  train_test_split

# load the csv file
df = pd.read_csv('finalDataa.csv',  sep='[;|_]', engine='python')
print(df.head())

# Select independent and dependent variable
X = df[["local", "type", "surface", "nbpiece"]]
y = df["prix"]
# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=250)


regressor = RandomForestRegressor(n_estimators=100, random_state=150)
regressor.fit(X_train, y_train)

# Score model
print(regressor.score(X_train, y_train))
# feature scaling

#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# instantiate the model
#classifier = RandomForestClassifier(n_estimators=300, random_state=0)

# fit the model
#classifier.fit(X_train, y_train)

#print(classifier.score(X_train, y_train))


#make pickle file of model
pickle.dump(regressor, open("model.pkl", "wb"))


