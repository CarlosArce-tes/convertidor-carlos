from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

# Crear la aplicación Flask
app = Flask(__name__)

# Datos de entrenamiento
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Capa de una sola neurona
capa = tf.keras.layers.Dense(units=1, input_shape=[1])

# Modelo secuencial con la capa definida
modelo = tf.keras.Sequential([capa])

# Compilación del modelo
modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

# Entrenamiento del modelo
modelo.fit(celsius, fahrenheit, epochs=3500, verbose=False)
print("Entrenamiento completado.")

# Ruta de inicio con el formulario
@app.route('/')
def index():
    return render_template('index.html', resultado=None)

# Ruta para la conversión de Celsius a Fahrenheit
@app.route('/convert', methods=['POST'])
def convert():
    grados_celsius = float(request.form['celsius'])
    grados_fahrenheit = modelo.predict([grados_celsius])
    resultado = grados_fahrenheit[0][0]
    return render_template('index.html', resultado=resultado)

# Ejecutar la aplicación Flask
if __name__ == '__main__':
    app.run()
