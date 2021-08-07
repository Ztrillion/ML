# Convolutional Neural Network

## 합성곱(Convolutional) 신경망 알고리즘

- 이미지 처리 작업에 주로 사용
- 합성곱 연산을 이용해 가중치(Weight)의 수를 줄이고 연산량 감소

- 여러 개의 **Fiter(Parameter Matrix)**로 이미지의 특징(Feature Map)추출
- Local connectivity % Parameter Sharing
- 말단에 Sigmoid 또는 Softmax 함수를 적용해 이미지 분류작업 수행
- DNN은 1차원으로만 Input을 받지만 CNN은 2,3차원도 Input가능

### CNN Hyperparameter

- Filter : 필터의 크기에 따라 input data의 크기가 줄어들 수 있다
- Stride : Filter를 적용하기 위해 이동하는 위치의 간격
  - Stride 값이 커지면 출력 특징 맵의 크기감소,일반적으로 1x1로 이동

- Pooling : 가로 , 세로 방향으로 크기를 줄이는 연산
  - MaxPooling , AveragePooling 중 Maxpooling을 주로씀

- Padding : 출력크기를 조정할 못적으로 사용
  - 합성곱 연산 수행 전 input_data 주변을 0으로 채움

- Channel : 
  - n차원 데이터, n차원 Filter를 사용 해 합성곱 연산수행
  - input_Data와 Filter의 채널 수는 같아야함

```python
from keras. rpeprocessing.image import ImageDataGenerator

train = ImageDataGenerator(rescale=1./255)
train_gen = train.flow_from_directory(
		# 디렉토리 명
    		train_dir,
    	# targetsize : 원하는 사이즈로 이미지 크기맞춤
			targetsize=()),
		# batch_size : 
    		batch_size = 20,
        # binary : 이진분류 0또는 1
        	class_mode = 'binary'
```



**Parameter 값을 똑같이 초기화 함으로써 모델의 결과가 다른 경우를 보완**

```python
from tensorflow.keras.initializers import HeNormal

initializer = HeNormal()
mnist = models.Sequential()
mnist.add(layers.Dense(512,activation='relu',input_shape=(28*28), kernel_initializer=initializer))
```

- 더 좋은 모델이라기 보다 어느 pc에서 모델링해도 일관성 있는 결과 도출