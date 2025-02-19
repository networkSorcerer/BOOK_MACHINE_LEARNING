import pandas as pd

fish = pd.read_csv('http://bit.ly/fish_csv_data')
fish.head()

print(pd.unique(fish['Species']))

fish_input = fish[['Weight', 'Length','Diagonal','Height','Width']].to_numpy()

print(fish_input[:5])

fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled,  train_target))
print(kn.score(test_scaled, test_target))

print(kn.classes_)
print(kn.predict(test_scaled[:5]))

import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))
