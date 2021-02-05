"""
Important Note: Make sure that the webcam is stationary, even slightest
moment in the webcam will make the program not work properly
"""
import cv2, time, pandas
from datetime import datetime

# img = cv2.imread("BigBang2Now1.png", 0)

first_frame = None
status_list = [None,None]
times=[]
df = pandas.DataFrame(columns=["Start", "End"])

video = cv2.VideoCapture(0)
"""
 Doc string done in this format
"""
while True:
    # Creating the first frame
    check, frame = video.read()
    # 0 is the symbol to denote that there is no motion in the current frame
    status = 0
    # Chsange the image colour to Gray
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # In Gaussian Blur operation, the image is blurred
    # This is done to reduce the noise and reduce detail
    gray = cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame,gray)

    # Thresholding:  is a non-linear operation that converts a gray scale
    # image to binary image where the two levels are assigned to pixel
    # that are below or above the specified threhold value.
    thresh_frame = cv2.threshold( delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    # Dilating is the frame
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)

    # Contours can be explained simply as a curve joining all the continuous
    # points (along the boundary), having same color or intensity. The contours
    # are a useful tool for shape analysis and object detection and recognition.

    (cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # image, cnts, hierarchy = cv2.findContours(thresh_frame.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        # checking if the window is bigger then 1000, the value can be
        # changed depending on what object to be captures
        if cv2.contourArea(contour) < 10000:
            continue
        # if it finds window > 1000 then status is changed to 1, means object detected
        status = 1

        # if the rectangle area is 1000 then the object will be shown in rectangle
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w ,y+h), (0,255,0), 3)

    status_list.append(status)
    # This is done to get only the last two status of the video frame to save memory
    status_list= status_list[-2:]


    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    cv2.imshow("Gray frame", gray)
    cv2.imshow("Delta frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Colour frame", frame)


    key= cv2.waitKey(1)

    # this is to print the arrays that store the image data
    # print(gray)
    # print(delta_frame)

    if(key == ord('q')):
        break

print(status_list)
print(times)

for i in range(0,len(times),2):
    df=df.append({"Start": times[i], "End" : times[i+1]},ignore_index=True)

# this line is sending the data frame to .csv file
df.to_csv("Times.csv")
video.release()
cv2.destroyAllWindows
