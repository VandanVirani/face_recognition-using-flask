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

#### Now final step is to compare live face encodings with store face, if the result of comparision is true then the image is similar.
#### Make final.py , paste this below code .
```
from operator import add, mod
import pickle
from flask import Flask ,render_template,Response
import cv2,numpy as np 
import face_recognition
import os
# we will use deep learning model for image recognition 

address = "http://192.168.161.97:8080/video"
# if_face = cv2.CascadeClassifier("C:\\Users\\Ravi\\Desktop\\test_1\\haarcascade_frontalface_alt2.xml")
# labels = {}
# with open("C:\\Users\\Ravi\\Desktop\\test\\encodings\\ritik\\encr1.pickle",'rb') as f : #wb as writing bytes , f as files
#     encodings = pickle.load(f)




app = Flask(__name__)
camera =  cv2.VideoCapture(address)

paths = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(paths,"encodings")


encoding = []
name      = []
id = []
for root,dirs,files in os.walk(path):
    for file in files:
        
        paths = os.path.join(root,file)
        if str(os.path.basename(os.path.dirname(paths))) not in name:
            name.append(str(os.path.basename(os.path.dirname(paths))) )
            encoding.append( [np.load(paths)] )
        else:
            index = name.index(str(os.path.basename(os.path.dirname(paths)))  )
            encoding[index].append(np.load(paths))

def generate_frames():    
    global multi_frame
    multi_frame = [0,0]
    while 1:
        for i in range(2):
            id = np.zeros((len(name) ))
            succ , frame = camera.read()
            if not succ:
                break
            else:
                frame = cv2.resize(frame,(0,0),None,.5,.5)
                multi_frame[i] = frame
                ret,buffer=cv2.imencode('.jpg',frame)
                framee=buffer.tobytes()
            yield  (b'--frame\n'b'Content-Type: image/jpeg\n\n' + framee + b'\n') 



def identify():
    global multi_frame
    # print(np.shape(multi_frame))
    frame = multi_frame[0]
    def encodings(image):
        if len(face_recognition.face_locations(image))!=0:
            r = face_recognition.face_locations(image)
            enc = face_recognition.face_encodings(image,r,num_jitters=3)
            return True,enc,r
        else :
            return False,False,False
    succ_enco_v=[]
    test_enc=[]
    test_location=[]
    r =[ ]
    for d in range(len(multi_frame)):
        a,b,c=encodings(multi_frame[d])
        if d==0 and c!=False:
            print(c)
            h  = len(c)
        if a==True and h==len(c):
            succ_enco_v.append(a)
            test_enc.append(b)
            test_location.append(c)
        else:
            succ_enco_v.append(a)
            r.append(d)
    id = np.zeros((len(name)))   
    # print("fvfvf",test_location)
    if True in succ_enco_v:     
        # print("vdvf",np.shape(test_enc))
        if len(test_location[0])==1:  
            for k in range(len(test_enc)):
                id =np.add(id,  [sum(face_recognition.compare_faces(encoding[i],test_enc[k][0] , tolerance=0.6 )  )  for i in range(len(name)) ]   )
            d = np.where(id == np.max(id))
            re = name[d[0][0]]
            print(id,"   ",re)
            cv2.putText(frame,f'{re}',org=(test_location[0][0][3],test_location[0][0][0]),fontFace = cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(0,0,255),thickness=2)  
            cv2.rectangle(frame,(test_location[0][0][1],test_location[0][0][2]),(test_location[0][0][3],test_location[0][0][0]),(255,0,255),2)
            
        elif len(test_location[0])>1:
            # print(len(test_location))
            face_lloc = []
            face_ident =[]
            for o in range(len(test_location)):
                for k in range(len(test_enc)):
                    id =np.add(id , [sum(face_recognition.compare_faces(encoding[i],test_enc[k][o] , tolerance=0.6 )  )  for i in range(len(name)) ]  )
                face_lloc.append(test_location[o])
                d = np.where(id == np.max(id))
                face_ident.append(name[d[0][0]])
                print("vffv",id)
            for o in range(len(test_location[0])):
                cv2.putText(frame,f'{face_ident[o]}',org=(face_lloc[0][o][3],face_lloc[0][o][0]),fontFace = cv2.FONT_HERSHEY_COMPLEX,fontScale=1.5,color=(0,0,255),thickness=2)  
                cv2.rectangle(frame,(face_lloc[0][o][1],face_lloc[0][o][2]),(face_lloc[0][o][3],face_lloc[0][o][0]),(255,0,255),2)
    ret,buffer=cv2.imencode('.jpg',frame)
    frame=buffer.tobytes()
    yield  (b'--frame\n'b'Content-Type: image/jpeg\n\n' + frame + b'\n')   ## it will continuesly return the image



@app.route('/')

def html():
    return render_template('index.html')

@app.route('/video')

def videos():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route("/image_identify")
def buttons():
    return Response(identify(),mimetype='multipart/x-mixed-replace; boundary=frame')

if  __name__ == "__main__" :
    app.run(debug=True,host='127.0.0.1',port=2000)

```

### Now the result is :
<img src="https://user-images.githubusercontent.com/76767487/174471980-8f0b1d48-0643-4bac-b6b3-50953894e639.png" width=1000 height=500>
    

