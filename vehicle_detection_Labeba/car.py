

import cv2
import numpy as np
import math

cascade_src = 'cars.xml'
video_src = 'dataset/video2.avi'
#video_src = 'dataset/video2.avi'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


wi=int(frameWidth)
hei=int(frameHeight)

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    myA=int((wi/2)-10)
    myB=int(hei-20)
    myC=int(wi/2)+10
    myD=int(hei)
    myE=myA+10
    myF=myB+10
    ftext="center is : ("+str(myE)+","+str(myF)+")"


    cv2.rectangle(img,(myA,myB),(myC,myD),(255,0,0),1)


    center1x=int((myA+myC)/2)
    center1y=int((myB+myD)/2)

    
    img2 = 255* np.ones(shape=[512, 512, 3], dtype=np.uint8)
    p=10
    q=40
    t1=18
    t2=18
    cv2.putText(img2, "object  distance   degree  position indicate",(10,10),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1,color=(0,0,0))
    for (x,y,w,h) in cars:
        
        center2x=int((x+x+w)/2)
        center2y=int((y+y+h)/2)
        cv2.line(img,(center1x,center1y), (center2x, center2y), (255,255,255),1)
        distance=pow(((center1x-center2x)**2)+((center1y-center2y)**2) ,0.5)
        if (center1x!=center2x):
            degree=math.degrees(math.atan((center2y-center1y)/(center2x-center1x)))
            if (degree<0):
                degree=180+degree

        if (degree>110):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            direc="left"
        
        elif (degree>70 and degree<110):
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
            direc="straight"
        
        else :
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2) 
            direc="right"
        
        range="safe"
        if (distance < 130 and (direc=="left" or direc=="straight")):
            range="inRange"
        cv2.putText(img2, "car      "+str(format(distance, '.2f'))+"    "+str(format(degree, '.2f'))+"  "+direc+"  "+range,(p,q),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1,color=(0,0,0))
        cv2.putText(img, "car",(x+2,y+2),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1,color=(255, 255, 255)) 
        cv2.line(img2,(10,t1), (600, t2), (0,0,0),1)
        q+=30
        t1+=30
        t2+=30
        

    
    cv2.imshow('video', img)
    cv2.imshow('video2', img2)
    
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()