#pip intall opencv-contrib-python
import cv2 as cv
img=cv.imread("group 2.jpg")
cv.imshow("lady",img)

#we have to convert it to gray as it  does not detect  color and it detect edges to determine a face

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)
# detcting faces
harr_cascade=cv.CascadeClassifier("haar_face.xml")
faces_rect=harr_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
print(f"number of faces= {len(faces_rect)}")
for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow("detected_face",img)
cv.waitKey(0)