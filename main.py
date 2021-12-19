#pip intall opencv-contrib-python
import cv2 as cv
img=cv.imread("02-still-for-america-room-loop-superJumbo.jpg")
cv.imshow("lady",img)
count=0
#we have to convert it to gray as it  does not detect  color and it detect edges to determine a face

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)
# detcting faces
harr_cascade=cv.CascadeClassifier("haar_face.xml")
faces_rect=harr_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
print(f"number of faces= {len(faces_rect)}")
for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
    count+=1
    cv.putText(img,"face",(x,y),cv.FONT_HERSHEY_COMPLEX,1.1,(255,255,255),thickness=2)
img=cv.resize(img,(600,600))
print(count)
cv.imshow("detected_face",img)
cv.waitKey(0)