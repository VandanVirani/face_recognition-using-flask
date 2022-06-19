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

os.chdir("C:\\Users\\Ravi\\Desktop\\test_1")


app = Flask(__name__)
camera =  cv2.VideoCapture(address)

# path = "C:\\Users\\Ravi\\Desktop\\test_1\\encodings"

# print(path)
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





print("half completed")





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



# # face_ca = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') # cascadeclassifier used to detect that their is face in image
# # recognizer = cv2.face.LBPHFaceRecognizer_create()
# # recognizer.read("trainer.yml")
# # for i in range(14):
# #     image = cv2.imread("C:\\Users\\Ravi\\Desktop\\test\\image\\ritik\\r{}.jpg".format(i+1))
# #     image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
# #     id,conf = recognizer.predict(image)
# #     if conf>=45 and conf<=85:
# #                 print(id,"    ",labels[id])


