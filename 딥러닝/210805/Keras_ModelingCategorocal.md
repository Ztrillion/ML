# Deep Nueral Network

## Categorical  Classification

### Overfitting  

- train 데이터에만 최적화된 상태

- 학습데이터가 부족하거나 

  - 더 많은 Training Data

- 모델의 capacity가 높아서(파라미터가 많고, 모델이 복잡)

  - Model_Capacity 

    - Hidden_layer 를 줄이거나 Layer의 Node의 수 줄이기

      

  - L2 Regularization(규제화)
    - 가중치의 제곱에 비례하는 노이즈를 cosfunction에 추가
    - 릿지, 라쏘 등 규제는 0~1사이값

  ``` python
  # ex) **kernel_regularizer=regularizers.l2(0.00001)**,
  mnist = models.Sequenctial()
  mnist.add(layes.Dense(512, activation='relu',
  	kernel_regularizer=regularizers.l2(0.00001),
  	input_shape=(28*28)))
  ```

  

  - Dropout
    - Network 연결 무작위로 끊기

  ```python
  # dropout ex)
  mnist = models.Sequenctial()
  mnist.add(layes.Dense(512, activation='relu',input_shape=(28*28)))
  mnist.add(layers.Dropout(0,4))
  ```

  

  - **Batch Normalization**
    - 레이어와 레이어를 통과하는 데이터를 전처리 함으로써 Capacity를 떨어뜨리지 않고, 오버피팅을 줄여줌

```python
# 변경 전

mnist = models.Sequential()
mnist.add(layers.Dense(512, activation='relu', input_shape(28*28)))
mnist.add(layers.Dense(256, activation = 'relu'))
mnist.add(layers.Dense(10, activation='softmax'))

# 변경 후

mnist = models.Sequential()
mnist.add(layers.Dense(512,input_shape(28*28)))
mnist.add(layers.BatchNormalization())
mnist.add(layers.Ativation='relu')
mnist.add(layers.Dense(256)
mnist.add(layers.BatchNormalization())
mnist.add(layers.Ativation='relu')
mnist.add(layers.Dense(10, activation='softmax'))
```



---

데이터가 불균형 일때 train_test_split(X,y,test_size, random_state, stratify) stratify옵션 주어야함.

모델 학습방법 

- 다중분류 : categorical_crossentropy