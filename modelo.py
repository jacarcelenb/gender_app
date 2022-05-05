from keras.models import model_from_json
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn import preprocessing



def Preprocesamiento_Data(lista):
  admissions = pd.read_csv('min_max.csv')
  nueva_fila = pd.Series(lista, index=admissions.columns) # creamos un objeto Seris
  admissions = admissions.append(nueva_fila, ignore_index=True)
  datos=admissions.values
  # aplicar el preprocesamiento de los datos
  datos_min_max = preprocessing.MinMaxScaler().fit_transform(datos)
  # reducir dimensiones con PCA
  pca = decomposition.PCA(n_components=2)
  datos_min_max = pca.fit_transform(datos_min_max)
  data =datos_min_max[0,0:]
  data.shape = (1,2)
  return data



def predecirGenero(lista):
    datos_array = np.array(Preprocesamiento_Data(lista), "float32")
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # cargar pesos al nuevo modelo
    loaded_model.load_weights("model.h5")
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # Compilar modelo cargado y listo para usar usando los mismos parámetros.
    loaded_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
    return int(loaded_model.predict(datos_array.reshape(1, 2)).round())


    


