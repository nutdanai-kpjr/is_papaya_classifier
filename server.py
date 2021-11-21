from typing import final
from flask import Flask, render_template, request
import keras
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
# import keras


app = Flask(__name__)
ispapaya_model = load_model('isPapayaModel.h5')
ispapaya_reconstructed_model = keras.models.load_model("isPapayaModel.h5")
ripeness_model = load_model('ripenessModel.h5')
ripeness_reconstructed_model = keras.models.load_model("ripenessModel.h5")
batch_size = 32
img_height = 64
img_width = 64
ripeness_class_names=['Medium', 'Not_Ripe', 'Ripe']
ispapaya_class_names=['Not Papaya','Papaya']

def final_predict(img_path):
	final_ans  = isPapaya_predict_label(img_path)
	if(final_ans[0]=='Papaya'):
		final_ans  = ripeness_predict_label(img_path)
	return final_ans

def ripeness_predict_label(img_path):
	i = image.load_img(img_path, target_size=(img_height,img_width))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 64,64,3)
	p = ripeness_model.predict(i)
	img = keras.preprocessing.image.load_img(
    img_path, target_size=(img_height, img_width)
)
	img_array = keras.preprocessing.image.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0) # Create a batch

	predictions = ripeness_reconstructed_model.predict(img_array)
	score = tf.nn.softmax(predictions[0])

	print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(ripeness_class_names[np.argmax(score)], 100 * np.max(score))
	)
	return [ripeness_class_names[np.argmax(score)], str(round(100 * np.max(score), 2)) ]

def isPapaya_predict_label(img_path):
	i = image.load_img(img_path, target_size=(img_height,img_width))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 64,64,3)
	p = ispapaya_model.predict(i)
	img = keras.preprocessing.image.load_img(
    img_path, target_size=(img_height, img_width)
)
	img_array = keras.preprocessing.image.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0) # Create a batch

	predictions = ispapaya_reconstructed_model.predict(img_array)

	score = tf.nn.softmax(predictions[0])

	print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(ispapaya_class_names[np.argmax(score)], 100 * np.max(score))
	)
		
	return [ispapaya_class_names[np.argmax(score)], str(round(100 * np.max(score), 2)) ]

@app.route('/',methods = ['GET', 'POST'])
def home():
    return render_template("index.html")
    
@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		# img_path = "static/test.png"
		img.save(img_path)

		p = final_predict(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)



if __name__ == "__main__":
    app.run(debug=True,port='5001')