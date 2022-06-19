import encodings
from mimetypes import encodings_map
import os
import numpy as np
import cv2
import pickle
import face_recognition

# face_ca = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# recognizer = cv2.face.LBPHFaceRecognizer_create()


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
    # print(root,dirs,files)
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

            # with open("{}\enc{}.pickle".format(eco_path,os.path.basename(file).replace(".jpg","").lower()),'wb') as f:
            #     pickle.dump(encodings_image,f)
            

            # if not label in label_id:
            #     label_id['{}'.format(label)] = current_id
            #     current_id+=1
            # id = label_id['{}'.format(label)]  
            
            

            # faces = face_ca.detectMultiScale(gray_image,scaleFactor=1.4,minNeighbors=5)
            # for (a,b,c,d) in faces:
            #     print(path,a,b,c,d)
            #     abc = image_gray[b:b+d,a:a+c]
            #     x_train.append(abc)
            #     y_label.append(id)




# with open("labels.pickle",'wb') as f : #wb as writing bytes , f as files
#     pickle.dump(label_id,f)

# print(len(x_train) ,y_label)
# recognizer.train(x_train,np.array(y_label))    
# recognizer.save("trainer.yml")