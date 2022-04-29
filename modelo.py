from keras.models import model_from_json
import tensorflow as tf
import numpy as np



def min_max_scaler(lista):
  listado = []
  listamin = [0 ,11.4,5.1,0,0,0,0]
  listamax = [1,15.5,7.1,1,1,1,1]

  valor_min_max = 0

  for i in range(len(lista)):
    valor_min_max = (lista[i] - listamin[i])/(listamax[i] - listamin[i])
    listado.append(valor_min_max)
  return listado

def predecirGenero(lista):
   
    datos_array = np.array(min_max_scaler(lista), "float32")
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # cargar pesos al nuevo modelo
    loaded_model.load_weights("model.h5")
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # Compilar modelo cargado y listo para usar usando los mismos parámetros.
    loaded_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])

    return int(loaded_model.predict(datos_array.reshape(1, 7)).round())


    


