from CoordAtt import CoordAtt
import tensorflow as tf
from keras.layers import Add
from keras import backend as K

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from keras import backend as K

feature_input = K.ones((100, 120, 120, 64))
coordinate_attention = CoordAtt(feature_input)  # coordinate_attention
feature_output = Add()([coordinate_attention, feature_input])  # skip connection
print(K.shape(feature_output))
