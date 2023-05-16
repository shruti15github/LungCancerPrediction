from tensorflow.keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

from tensorflow.keras.models import load_model
lo_model=load_model('lu_prediction.h5')
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

img = image.load_img('Datasets/val/NORMAL/NORMAL2-IM-1431-0001.jpeg', target_size=(224, 224))
img = image.img_to_array(img)
img = img/255
imshow(img)
plt.axis('off')
img = np.expand_dims(img,axis=0)
answer = lo_model.predict(img)

if answer[0][0] > 0.5:
    print("The X-Ray is NORMAL")
else:
    print("The X-RAY indicates the presence of CANCER")