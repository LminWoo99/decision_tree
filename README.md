# decision_tree
seaborn 라이브러리 ‘mpg’데이터에서 ‘origin 변수는 자동차 제조국으로 usa, europe, japan 세 가지 값을 갖음.<br>
mpg, cylinders, displacement, horsepower, weight, acceleration, model_year 변수를 이용하여 자동차 제조국 을 분류하는 결정트리 분석을 수행

<h2>데이터</h2>
seaborn 라이브러리 'mpg'데이터

<h2>진행 과정</h2>
  1. 데이터 전처리 여부 확인<br>
  2. 훈련용, 테스트용 데이터셋 분리<br>
  3. 결정트리 분류분석 모델 구축하여 모델을 생성하고 트리 모형 적합(훈련)하고 예측 수행<br>
  4. 생성된 결정 트리 모델의 분류 정확도 성능을 확인<br>
  5. GridSearchCV모듈을 사용하여 정확도를 검사하고 최적의 하이퍼 매개변수를 찾는 작업 수행<br>
  6. Decision Tree Classifier의 주요매개변수들을 이용하여 조정하면서 최고의 평균 정확도 찾기<br>
  7. 최적 모델 grid_cv.best_estimator_을 사용하여 테스트 데이터에 대한 예측을 수행<br>
  8. [feature_importances_ 속성을 사용하여 각 피처의 중요도를 알아내고 중요도가 높은 5개 피처를 찾아 그래프로 표시](https://github.com/LminWoo99/decision_tree/blob/main/featureTop5.png)<br>
  9. [Graphviz 패키지 : 결정트리 모델의 트리구조를 그림으로 시각화하기](https://github.com/LminWoo99/decision_tree/blob/main/%E1%84%80%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%90%E1%85%B3%E1%84%85%E1%85%B5.png)
  

<h2>모델 평가</h2>
해당 데이터프레임의 결측치 여부를 확인한 결과, horespower변수에 6개의 결측치가 있는걸 확인하여 drop하여 전처리 하였다.<br>
결정 트리 모델의 분류 정확도 성능은 0.810126582278481로 81%가량으로 나왔고, GridSearchCV를 사용하여 탐색한 하이퍼 매개변수 중에서 가장 높은 평균 정확도는 0.8243로 약 82.43%이고 이는 하이퍼 매개변수를 조합해 실행한 결과중 가장 높은 정확도를 나타내는 모델이다. 최적 하이퍼 매개변수는 max_depth가 6인 경우이다.<br> 결정 트리의 최대 깊이가 6인 경우가 가장 성능이 우수한것으로 판단된다. 최적모델의 테스트 데이터 예측 정확도는 0.7975로 약 79.75%의 정확도를 달성했다.<br> 전체적으로 정확도로 판단할 때, 이 모델의 성능은 충분히 좋다고 판단 할 수 있다. 중요도가 높은 5개 피처를 그래프로 확인해보면 displacement, horsepower, weight, acceleration, model_year순으로 높은 걸 확인 할 수있고, displacement 피처가 압도적으로 중요도가 높은 걸 확인 할 수 있다.
