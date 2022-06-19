import encodings
from mimetypes import encodings_map
import os
import numpy as np
import cv2
import pickle
import face_recognition


tolerance  = 0.6
frame_thickness = 3
front_thickness = 2
model = "cnn"   # or hog 

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir,"image")


current_id =0 
label_id= {}
x_train=[]
y_label=[]

for root,dirs,files in os.walk(image_dir):
      for file in files:
        if (file.endswith("jpg")) or (file.endswith("jpeg")):
            path= os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            print(label)
            d = os.path.dirname(os.path.dirname(root))
            eco_path = os.path.join(d,"encodings\{}".format(label))

            img = cv2.imread(path)
            a =img.shape[0]
            b =img.shape[1] 
            c =img.shape[2]
            if b>1500:
               img = cv2.resize( img,(0,0) , None,0.4,0.4)
            else:
               img = cv2.resize( img,(0,0) , None,0.7,0.7)
            
            cv2.imshow("vf",img)
            cv2.waitKey(1) 

            encodings_image = face_recognition.face_encodings(img,num_jitters=20)[0]  # higher num_jitter more the accurate
            np.save("{}\enc_{}.npy".format(eco_path,os.path.basename(file).replace(".jpg","").replace(".jpeg","").lower()),encodings_image)
