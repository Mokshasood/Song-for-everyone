from django.shortcuts import render

from django.core.files.storage import FileSystemStorage

from keras.models import load_model
from keras.preprocessing import image
import json
from tensorflow import Graph
import tensorflow as tf 

model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model=load_model('./models/model_keras.h5')


img_height, img_width=128,128

labelInfo = ['angry','contempt','disgust','fear','happy','sad','surprise']

# Create your views here.
def index(request):
	context = {'a':1}
	return render(request,'index.html',context)

def predictImage(request):
	print (request)
	print (request.POST.dict())
	fileObj = request.FILES['filePath']
	fs = FileSystemStorage()
	filePathName = fs.save(fileObj.name, fileObj)
	filePathName = fs.url(filePathName)
	testimage='.'+filePathName
	img = image.load_img(testimage, target_size=(img_height, img_width))
	x = image.img_to_array(img)
	x=x/255
	x=x.reshape(1,img_height, img_width,3)
	with model_graph.as_default():
		with tf_session.as_default():
			predi=model.predict(x)

	import numpy as np
	predictedLabel=labelInfo[np.argmax(predi[0])]

	context={'filePathName':filePathName,'predictedLabel':predictedLabel}

	return render(request,'index.html',context)

def surprise(request):
	context = {'a':1}
	return render(request,'Surprise.html',context)

def happy(request):
	context = {'a':1}
	return render(request,'happy.html',context)

def sad(request):
	context = {'a':1}
	return render(request,'sad.html',context)

def fear(request):
	context = {'a':1}
	return render(request,'fear.html',context)

def angry(request):
	context = {'a':1}
	return render(request,'angry.html',context)

def contempt(request):
	context = {'a':1}
	return render(request,'contempt.html',context)

def disgust(request):
	context = {'a':1}
	return render(request,'disgust.html',context)

def viewDataBase(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html',context) 