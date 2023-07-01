#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 18:12:52 2023

@author: imin-u
"""

import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt

mp=sns.load_dataset('mpg')
mp.info()

#문제 3_1번: 결측치 여부
print("문제 3_1번")
mp_check=mp.isnull().sum()
print(mp_check)
#문제 3_2 : 데이턴 전처리
print("문제 3_2번")
mp.dropna(subset=['horsepower'], inplace=True)
mp_check=mp.isnull().sum()  
print(mp_check)
#문제 3_3
# 독립 변수(X)와 종속 변수(y) 분리
X = mp[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']]
y = mp['origin']

# 훈련용, 테스트용 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#문제 3_4번
# 결정트리 모델 생성
tree = DecisionTreeClassifier()

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
print("문제 3_4번 결정 트리 모델")
print(y_pred)
#문제 3_5
accuracy = accuracy_score(y_test, y_pred)
print("문제 3_5번")
print("분류 정확도:", accuracy)
#문제 3_6, 3_7
params={
'max_depth':[6,8,10,12,16,20,24]
}
grid_cv=GridSearchCV(tree ,param_grid=params,scoring='accuracy',cv=5,return_train_score=True)
grid_cv.fit(X_train,y_train)
print("문제 3_6번, 3_7번")
print('최고평균정확도:{0:.4f},최적하이퍼매개변수:{1}'.format(grid_cv.best_score_,grid_cv.best_params_))
#문제 3_8
best_dt_HAR=grid_cv.best_estimator_
y_pred_best = best_dt_HAR.predict(X_test)
print("문제 3_8번")
best_accuracy=accuracy_score(y_test, y_pred_best)
print('best결정트리예측정확도:{0:.4f}'.format(best_accuracy))
#문제 3_9번
feature_importance_values=best_dt_HAR.feature_importances_
feature_importance_values_s=pd.Series(feature_importance_values,index=X_train.columns)
feature_top5=feature_importance_values_s.sort_values(ascending=False)[:5]

plt.figure(figsize=(10,5))
plt.title('FeatureTop5')
sns.barplot(x=feature_top5,y=feature_top5.index)
plt.show()

#문제 3_10
label_name=mp.iloc[:,1].values.tolist()
feature_name=mp.iloc[:,1].values.tolist()
export_graphviz(best_dt_HAR,out_file="tree.dot", impurity=True,filled=True)

with open("tree.dot") as f:
    dot_graph=f.read()
    
graphviz.Source(dot_graph)


