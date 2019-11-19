import cv2 as cv
import numpy as np

cam=cv.VideoCapture(0)

while True:
    _,frame=cam.read()
    final=frame

    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    skin_lower=(10,25,0)
    skin_upper=(30,255,255)
    
    grayed=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    #blurred=cv.GaussianBlur(grayed,(9,9),0)
    #blurred=cv.medianBlur(grayed,11)
    #blurred=cv.GaussianBlur(hsv,(31,31),0)
    blurred=cv.medianBlur(hsv,7)

    mask=cv.inRange(blurred,skin_lower,skin_upper)    
    #ret,mask=cv.threshold(blurred,254,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    mask=cv.erode(mask,(5,5),iterations=3)
    mask=cv.dilate(mask,(5,5),iterations=4)
    mask=cv.medianBlur(mask,7)

    contours=cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)[-2]

    if len(contours)>0:
                   cnt=max(contours,key=cv.contourArea)
                   hull=cv.convexHull(cnt)

                   M=cv.moments(cnt)
                   centre=(int(M['m10']/M['m00']),int(M['m01']/M['m00']))
                   
                   #cv.drawContours(final,[cnt],0,(255,0,0),2)
                   cv.circle(final,centre,5,(0,255,0),2)

                   epsilon=0.01*cv.arcLength(cnt,True);
                   cnt = cv.approxPolyDP(cnt,epsilon,True)

                   hull=cv.convexHull(cnt,returnPoints=False)
                   defects=cv.convexityDefects(cnt,hull)

                   for i in range(defects.shape[0]):
                       s,e,f,d=defects[i,0]
                       start=tuple(cnt[s][0])
                       end=tuple(cnt[e][0])
                       far=tuple(cnt[f][0])
                       cv.line(final,start,end,(0,0,255),2)
                       cv.circle(final,far,3,(0,255,255),-1)
                       fingers=i

                   print('No.of fingers',fingers)
                   fingers=0

    #for mirror effect
    final=cv.flip(final,1)

    #cv.imshow('original',frame)
    #cv.imshow('hsv',hsv)
    #cv.imshow('grayscale',grayed)
    #cv.imshow('blurred',blurred)
    cv.imshow('mask',mask)
    cv.imshow('final',final)

    if cv.waitKey(1) & 0xff==ord('q'):
        break


cam.release()
cv.destroyAllWindows()
