from flask import Flask, jsonify, request, send_file, abort, redirect
from flasgger import Swagger
import cv2
from imutils import url_to_image

import time

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import keras
import numpy as np
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt

from PIL import Image

app = Flask(__name__)
Swagger(app)

face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

class_dog_cat = load_model('./models/cats_or_dogs.h5')

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

model_unet = load_model('./models/model_classic_unet.h5', custom_objects={'dice_coef': dice_coef})

@app.route('/')
def index():
    return redirect('/apidocs/')

@app.route('/01_semantic_segmentation',methods=['POST'])
def segmatic_segm():
    """Identifying the cells' nuclei
    How to use:
    1. Click the Try it out button.
    2. Replace URL or press Execute.
    More examples of images with cells: https://photos.app.goo.gl/wDcEdQTsYYq7S6Tv5
    Just replace url image.
    ---
    parameters:
      - name: url
        in: body
        required: true
        example: {'url':'https://lh3.googleusercontent.com/z0ZXuTSF_o_5gR-eqpU8w_kH3bXfBF2iiGVWqcA9pK10FaVfk7i919EC0TT5l5E2cdaADXlD34dli58xH2OXuSxiTjgRmLwi_mTRAJw4qArh5bkcYSa4Botv9ZQKzhRhaTR5zXJF8cMTdM5w7GE8KNLxtf-lE1JsoYX3QNvGAt6QnH88b9Loa6SjB6dtFq6G1sdqiBgQiyFSiu-elMOzvn870f5uPbdL2setkWJv0LY5jd6WvIkDcQYlGtrQWwgxzwTGkYFvURbCYdCPz19P_hoMm1odiY_FKHE25BH0mWJ8v0dikSxbi89YWRu58GuGTJJ2fg1v1fHpbeA529mLObA0gLlBtmExudBjgZzckTwsCgQFygf5yPHgdH-vWU_ZQn00g-jzgnfEP_YoNYM1W2vrm1CUJzFlJDHbmTyo50-8Ewwe7dkh6GO54PRlCr5jZr7lcU_cBdAPcNw3ayTpJlIuZ3aVXFbOjLYP-PSE5nCfoZ4Dpt_hRERtZLBLLanEVY_pmBTxg96Fn7hRWdy69LyV45XmyyAySWhxAIMubcvLlKNA_0h6yAVSzYTnnIps5UXZE9F5PUiEzhx4G52MbYTKcQObP65J89qyoBzWxgnkLTLQfQmCAol0KNnZvRYKbAgqp5gIzXTMQAkM3jDayYPwo911qJDuimpXWFvxr7KQk2s81meayDubxjJm=s360'}
    responses:
      200:
        description: image with detected nuclei
    """
       
    try:
      content = request.get_json()

      img = url_to_image(content['url'])
      sh_or = img.shape

      result = segmantation(img)
      result = np.squeeze(result)*255 
      result = cv2.resize(result, sh_or[:2], interpolation = cv2.INTER_AREA)

      timestr = time.strftime("%Y%m%d-%H%M%S")

      maskname = f'./static/mask{timestr}.jpg'      
      cv2.imwrite(maskname,result)

      origname = f'./static/orig{timestr}.jpg'
      cv2.imwrite(origname,img)

      concname = f'./static/conc{timestr}.jpg'

      merge_img(origname, maskname, concname)            

      return send_file(concname, mimetype='image/jpg')
    
    except Exception as e:
      print(e)
      return abort(400)



def segmantation(image_cells):
	
	img = cv2.resize(image_cells, (256, 256), interpolation = cv2.INTER_AREA)
	img = img.reshape(1,256,256,3)
	pred = model_unet.predict(img, verbose=1)
	pred_t = (pred > 0.5).astype(np.uint8)

	return pred_t

def merge_img(name_image1, name_image2, name_new_img):

  images = [Image.open(x) for x in [name_image1, name_image2]]
  widths, heights = zip(*(i.size for i in images))
  total_width = sum(widths)
  max_height = max(heights)
  new_im = Image.new('RGB', (total_width, max_height))
  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]
  new_im.save(name_new_img)

  return name_new_img

@app.route('/02_facedetect',methods=['POST'])
def face_detect():
    """Detect face on image from url
    Simple model for experience.
    ---
    parameters:
      - name: url
        in: body
        required: true
        example: {'url':'https://i.insider.com/5cb3c8e96afbee373d4f2b62?width=700'}
    responses:
      200:
        description: image with detected face
    """
       
    try:
      content = request.get_json()

      img = url_to_image(content['url'])
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)

      for (x,y,w,h) in faces:
          cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
          roigray = gray[y:y+h, x:x+w]
          roicolor = img[y:y+h, x:x+w]
      
      timestr = time.strftime("%Y%m%d-%H%M%S")
      
      fname = f'./static/file{timestr}.jpg'
      
      cv2.imwrite(fname,img)

      return send_file(fname, mimetype='image/jpg')
    
    except Exception as e:
      print(e)
      return abort(400)

@app.route('/03_dog_or_cat',methods=['POST'])
def cat_dog():
    """Determines if the image contains a cat or dog.
    Simple model for experience.
    ---
    parameters:
      - name: url
        in: body
        required: true
        example: {'url':'https://c.files.bbci.co.uk/12A9B/production/_111434467_gettyimages-1143489763.jpg'}
    responses:
      200:
        description: The answer to the question, cat or dog.
    """
       
    try:
      content = request.get_json()

      img_cat_dog = url_to_image(content['url'])

      result = CatOrDog_cl(img_cat_dog)
      

      return 'Its a ' + result 
    
    except Exception as e:
      print(e)
      return abort(400)

def CatOrDog_cl(image):

	image = cv2.resize(image, (150,150), interpolation = cv2.INTER_AREA)
	image = image.reshape(1,150,150,3) 
	res = str(class_dog_cat.predict_classes(image, 1, verbose = 0)[0][0])
	if res == "0":
		res = "Cat"
	else:
		res = "Dog"

	return res