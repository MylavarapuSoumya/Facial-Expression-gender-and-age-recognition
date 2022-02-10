# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 20:40:53 2021

@author: Soumya Mylavarapu
"""
import cv2
import math
import argparse
import os


from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


face_classifier = cv2.CascadeClassifier(r'C:\Users\ramyamylavarapu\Documents\miniProject\FER2\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
classifier =load_model(r"C:\Users\ramyamylavarapu\Documents\miniProject\FER2\Emotion_Detection_CNN-main\model.h5")
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']


from flask import Flask,render_template,Response

from flask import Flask,render_template,Response

padding=0
app=Flask(__name__)
camera=cv2.VideoCapture(0)
def generate_images():
    while True:
        success,frame=camera.read()
        
        if not success:
            break
        else:
            face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faces=face_classifier.detectMultiScale(frame,1.1,7)
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(int(x),int(y)),(x+w,y+h),(0,255,255),2)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



                if np.sum([roi_gray])!=0:
                    resultImg,faceBoxes=highlightFace(faceNet,frame)
                    for faceBox in faceBoxes:
                    
                        face=frame[max(0,faceBox[1]-padding):
                               min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                               :min(faceBox[2]+padding, frame.shape[1]-1)]
                        
                        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                        genderNet.setInput(blob)
                        genderPreds=genderNet.forward()
                        gender=genderList[genderPreds[0].argmax()]
                        
                        ageNet.setInput(blob)
                        agePreds=ageNet.forward()
                        age=ageList[agePreds[0].argmax()]
                
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)

                    prediction = classifier.predict(roi)[0]
                    label=emotion_labels[prediction.argmax()]+f'{gender}, {age}'
                    print(label)
                    label_position = (x,y)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                else:
                    cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                
            
                ret,buffer=cv2.imencode('.jpg',frame)
                frame=buffer.tobytes()
            
                yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture():
    return render_template('camera.html')


@app.route('/video')
def video():
    return Response(generate_images(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run()
