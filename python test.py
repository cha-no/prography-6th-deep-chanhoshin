import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

(_, _), (x_test, Y_test) = mnist.load_data()

X_test = []
for i in x_test:
    image = cv2.cvtColor(cv2.resize(i, (32,32)), cv2.COLOR_GRAY2BGR)
    X_test.append(image)

X_test = np.array(X_test)/255

my_vgg = load_model("my_vgg_model.hdf5")
print("테스트 정확도: %f" % (my_vgg.evaluate(X_test, Y_test,verbose=0)[1]))
