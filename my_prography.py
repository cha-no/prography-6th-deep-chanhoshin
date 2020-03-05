import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Add
from tensorflow.keras.applications import vgg16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# seed 값 설정
seed = 2020
np.random.seed(seed)
tf.random.set_seed(seed)

# 데이터 불러오기
(x_train, Y_train), (x_test, Y_test) = mnist.load_data()

# MNIST 데이터를 RGB 채널로 변경
X_train, X_test = [], []
for i in x_train:
    image = cv2.cvtColor(cv2.resize(i, (32,32)), cv2.COLOR_GRAY2BGR)
    X_train.append(image)
for i in x_test:
    image = cv2.cvtColor(cv2.resize(i, (32,32)), cv2.COLOR_GRAY2BGR)
    X_test.append(image)
X_train, X_test = np.array(X_train)/255, np.array(X_test)/255

# RGB 채널로 변경한 MNIST 데이터 plot
image = X_train[0]
image_R = np.array(image).copy()
image_R[:,:,1] = 0
image_R[:,:,2] = 0

image_G = np.array(image).copy()
image_G[:,:,0] = 0
image_G[:,:,2] = 0

image_B = np.array(image).copy()
image_B[:,:,0] = 0
image_B[:,:,1] = 0

plt.figure(figsize = (14,14))
plt.subplot(131)
plt.imshow(image_R)
plt.subplot(132)
plt.imshow(image_G)
plt.subplot(133)
plt.imshow(image_B)

# vgg-16 네트워크 구성
vgg = vgg16.VGG16(include_top=False, weights='imagenet')
for layer in vgg.layers:
    layer.trainable = False

vgg.summary()

# model initialization, inference 부분을 함수형태로 작성
def model_initialization(input_shape = (32,32,3)):
    return Input(shape=input_shape, name = 'mnist_input')

def model_inference(output):
    x = Flatten()(output)
    x = Dense(256,activation='relu')(x)
    x = Dense(10,activation='softmax',name='output')(x)
    return x

# conv2_1 입력을 첫번째 Dense 입력에 추가해주는 구조를 추가(skip connection 구조)
x1 = vgg.layers[1](vgg.input)
x = vgg.layers[2](x1)
x = Add()([x1,x])
for i in vgg.layers[3:]:
    x = i(x)
vgg = Model(inputs=vgg.input, outputs=x)
vgg.summary()

input_shape = (32,32,3)
my_input = model_initialization(input_shape)
output = vgg(my_input)
my_output = model_inference(output)
my_vgg = Model(inputs=my_input, outputs=my_output)
my_vgg.summary()

# epoch3번 동안 성능 안좋아지면 lr 절반으로 줄이고 5번동안 안좋아지면 학습종료
# 정확도 높은 모델 my_vgg_model.hdf5으로 저장
patience = 5
epochs = 20
file_path = "my_vgg_model.hdf5"
adam = tf.keras.optimizers.Adam(lr=0.001)
early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
check_point = ModelCheckpoint(file_path, monitor="val_loss", verbose=1,save_best_only=True, mode="min")
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=3)
my_vgg.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

history = my_vgg.fit(X_train, Y_train, validation_split=0.15,
                    batch_size=5000, epochs=epochs,
                    verbose=1, callbacks=[early_stop,check_point,reduce_lr])

# 학습된 결과를 plot
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('My_VGG Model', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,epochs+1))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, epochs+1, 10))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, epochs+1, 10))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
