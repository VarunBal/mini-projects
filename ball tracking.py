import cv2 as cv
import numpy as np
from collections import deque

camera=cv.VideoCapture(0)

tail=deque(maxlen=15)

while True:
    _,frame=camera.read()
    
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    yellow_lower=(20,150,60)
    yellow_upper=(40,255,255)

    hsv=cv.blur(hsv,(11,11))
    
    mask=cv.inRange(hsv,yellow_lower,yellow_upper)

    kernel = np.ones((4,4),np.uint8)

    mask=cv.erode(mask,None,iterations=2)
    mask=cv.dilate(mask,None,iterations=2)

    obj=cv.bitwise_and(frame,frame,mask=mask)

    cnts=cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[-2]
    centre=None
    

    if (len(cnts)>0):

        c=max(cnts,key=cv.contourArea)

        #for centre of the contour
        #M=cv.moments(c)
        #c_centre=(int(M['m10']/M['m00']),int(M['m01']/M['m00']))
        
        ((x,y),radius)=cv.minEnclosingCircle(c)
        centre=(int(x),int(y))

        if radius>5:
            cv.circle(frame,centre,int(radius),(255,0,0),3)
            cv.circle(frame,centre,2,(0,0,255),-1)
            #centre of contour
            #cv.circle(frame,c_centre,2,(0,255,0),-1)
    
    tail.appendleft(centre)

    for x in range(1,len(tail)):

        if tail[x-1]is None or tail[x]is None:
            continue

        #thickness=2
        thickness = int(np.sqrt(len(tail)/float(x+1))*2.5)
        cv.line(frame,tail[x-1],tail[x],(0,0,255),thickness)

    #for mirror effect
    frame=cv.flip(frame,1)
    
    cv.imshow('Tracks a Ball',frame)
    #cv.imshow('hsv',hsv)
    #cv.imshow('mask',mask)
    #cv.imshow('object',obj)

    if cv.waitKey(1) & 0xff==ord('q'):
        break
camera.release()
cv.destroyAllWindows()
