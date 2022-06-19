# face_recognition-using-flask

    
   #### In this project we are going to make a model which predict new faces by using stored face data. we are using flask module in order to interact with html webpage.

   #### It is the basic model for face recognition system, you can enhance with your flask skill.
   
#### First we will make templates folder having index html for requirement of flask module. 
#### Second we will make image folder and store the similar image like 
```
  |-> image  (folder)
      |-> person1  (folder)
         -p1_image_1
         -p1_image_2
         -p1_image_3
         -p1_image_4
      
      |-> person2  (folder)
         -p2_image_1
         -p2_image_2
         -p2_image_3
         -p2_image_4
         
      |-> person3  (folder)
         -p3_image_1
         -p3_image_2
         -p3_image_3
         -p3_image_4
```
#### Third we will make encodings folder, and make empty folder similar to image.
```
 |-> encodings  (folder)
     |-> person1  (folder)
      
     |-> person2  (folder)
         
     |-> person3  (folder)

```

we are going to make our face data folder to store face encodings. face encodings is the number of face features (about 120) of image for example the distance between nose and eyes, between two eyes etc. to do that we are going to make face_encoding_creator.py , paste this belowe code and run automatically the code will store the image encodings in encodings folder.
```
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

```



    

