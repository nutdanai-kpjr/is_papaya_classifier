from flask import Flask, render_template, request
import keras
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
# import keras


app = Flask(__name__)

model = load_model('modelNut.h5')
reconstructed_model = keras.models.load_model("modelNut.h5")
batch_size = 32
img_height = 64
img_width = 64
class_names=['Medium', 'Not_Ripe', 'Ripe']

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(img_height,img_width))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 64,64,3)
	p = model.predict(i)
	img = keras.preprocessing.image.load_img(
    img_path, target_size=(img_height, img_width)
)
	img_array = keras.preprocessing.image.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0) # Create a batch

	predictions = reconstructed_model.predict(img_array)
	score = tf.nn.softmax(predictions[0])

	print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
	)
	return [class_names[np.argmax(score)], str(round(100 * np.max(score), 2)) ]

@app.route('/',methods = ['GET', 'POST'])
def home():
    return render_template("index.html")
    
@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)



if __name__ == "__main__":
    app.run(debug=True)