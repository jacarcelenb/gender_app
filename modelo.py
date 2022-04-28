from keras.models import model_from_json
import tensorflow as tf
import numpy as np

def preprocesar(lista):
    datos = []

    return datos

def predecir(lista):
    # cargamos los datos
    datos_array = np.array(lista, "float32")
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


