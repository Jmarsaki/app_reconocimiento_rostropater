# Importar las librerías necesarias
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import torch
from model import GAN # Importar el modelo GAN desde el archivo model.py

# Crear una instancia de la aplicación Flask
app = Flask(__name__)

# Cargar el modelo GAN pre-entrenado
gan = GAN()
gan.load_state_dict(torch.load("gan.pth")) # Cargar los pesos del modelo desde un archivo
gan.eval() # Poner el modelo en modo de evaluación

# Definir una función que tome dos imágenes de padre o madre e hijos y las procese para obtener un tensor de entrada para el modelo GAN
def preprocess_images(img1, img2):
  # Convertir las imágenes a escala de grises
  img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  # Redimensionar las imágenes a 64x64 píxeles
  img1 = cv2.resize(img1, (64, 64))
  img2 = cv2.resize(img2, (64, 64))
  # Concatenar las imágenes horizontalmente
  img = np.hstack((img1, img2))
  # Normalizar las imágenes entre -1 y 1
  img = img / 127.5 - 1
  # Convertir las imágenes a tensor de PyTorch
  img = torch.from_numpy(img).float()
  # Añadir una dimensión de canal al tensor
  img = img.unsqueeze(0)
  # Añadir una dimensión de lote al tensor
  img = img.unsqueeze(0)
  # Devolver el tensor de entrada
  return img

# Definir una función que tome el tensor de entrada y lo pase por el modelo GAN para obtener un tensor de salida con la imagen generada del otro progenitor
def generate_image(img):
  # Pasar el tensor de entrada por el modelo GAN
  output = gan(img)
  # Eliminar la dimensión de lote del tensor de salida
  output = output.squeeze(0)
  # Eliminar la dimensión de canal del tensor de salida
  output = output.squeeze(0)
  # Devolver el tensor de salida
  return output

# Definir una función que tome el tensor de salida y lo convierta en una imagen y la guarde en la carpeta static
def save_image(img):
  # Desnormalizar el tensor de salida entre 0 y 255
  img = (img + 1) * 127.5
  # Convertir el tensor de salida a array de NumPy
  img = img.numpy()
  # Convertir el array de NumPy a imagen de OpenCV
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  # Guardar la imagen en la carpeta static con el nombre result.jpg
  cv2.imwrite("static/result.jpg", img)

# Definir una ruta / que muestre el archivo HTML con un formulario para subir las dos imágenes de padre o madre e hijos
@app.route("/", methods=["GET", "POST"])
def index():
  # Si el método es POST, significa que el usuario ha subido las imágenes
  if request.method == "POST":
    # Obtener las imágenes desde el formulario
    img1 = request.files["img1"]
    img2 = request.files["img2"]
    # Leer las imágenes como arrays de NumPy
    img1 = cv2.imdecode(np.frombuffer(img1.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img2.read(), np.uint8), cv2.IMREAD_COLOR)
    # Guardar las imágenes en la carpeta static con los nombres img1.jpg y img2.jpg
    cv2.imwrite("static/img1.jpg", img1)
    cv2.imwrite("static/img2.jpg", img2)
    # Redirigir a la ruta /result
    return redirect(url_for("result"))
  # Si el método es GET, significa que el usuario quiere ver el formulario
  else:
    # Mostrar el archivo HTML con el formulario
    return render_template("index.html")

# Definir una ruta /result que reciba las dos imágenes subidas, las procese con las funciones anteriores y muestre el archivo HTML con la imagen generada del otro progenitor
@app.route("/result")
def result():
  # Leer las imágenes desde la carpeta static
  img1 = cv2.imread("static/img1.jpg")
  img2 = cv2.imread("static/img2.jpg")
  # Procesar las imágenes con la función preprocess_images
  img = preprocess_images(img1, img2)
  # Generar la imagen con la función generate_image
  output = generate_image(img)
  # Guardar la imagen con la función save_image
  save_image(output)
  # Mostrar el archivo HTML con la imagen generada
  return render_template("result.html")

# Ejecutar la aplicación Flask
if __name__ == "__main__":
  app.run(debug=True)