
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten


inputs = Input((28,28,3)) #--- For Image
h0 = Flatten()(inputs)
h1 = Dense(4,activation='relu', name = 'Hidden_ayer_1')(h0)
h2 = Dense(8,activation='relu', name = 'Hidden_ayer_2')(h1)
h3 = Dense(4,activation='relu', name = 'Hidden_ayer_3')(h2)
outputs = Dense(1, activation = 'sigmoid',  name= 'Output_layer')(h3)
model = Model(inputs, outputs)
model.summary(show_trainable = True)


#covulation layer