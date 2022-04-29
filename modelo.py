from keras.models import model_from_json
from matplotlib.pyplot import axis
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn import decomposition
import pandas as pd

def predecirGenero(lista):
    # cargamos los datos
    datos_array = np.array(lista, "float32")
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # cargar pesos al nuevo modelo
    loaded_model.load_weights("model.h5")
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    # Compilar modelo cargado y listo para usar usando los mismos parámetros.
    loaded_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])

    return int(loaded_model.predict(datos_array.reshape(1, 2)).round())


    


