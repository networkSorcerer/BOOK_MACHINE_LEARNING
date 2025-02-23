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

distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])

import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-5,5,0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z,phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

char_arr = np.array(['A','B','C','D','E'])
print(char_arr[[True, False, True, False, False]])

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5]))

print(lr.predict_proba(train_bream_smelt[:5]))

print(lr.classes_)

# 로지스틱 ? 뉴스에서 어느 해당 기업을 검색 했을때 GOOD NEWS , BAD NEWS 나눌떄 써볼? 이진분류 
# 다항은 단위 값이 다른 데이터를 가지고 학습 할때
# 근접은 기업이 요구하는 기술 스택 + 회원의 기술 스택을 통해서 매칭 할때?

print(lr.coef_, lr.intercept_)

decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

from scipy.special import expit
print(expit(decisions))


lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled,test_target))

print(lr.predict(test_scaled[:5]))

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))

print(lr.classes_)

print(lr.coef_.shape, lr.intercept_.shape)

decisions = lr.decision_function(test_scaled[:5])
print(np.round(decision,decimals=2))

from scipy.special import softmax
proba = softmax(decisions, axis =1)
print(np.round(proba, decimals=3))
