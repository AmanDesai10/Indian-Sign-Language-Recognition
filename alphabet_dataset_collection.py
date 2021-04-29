import cv2
import os
from string import ascii_uppercase

if not os.path.exists("dataset-alpha"):
    os.makedirs("dataset-alpha")
if not os.path.exists("dataset-alpha/train"):
    os.makedirs("dataset-alpha/train")
if not os.path.exists("dataset-alpha/test"):
    os.makedirs("dataset-alpha/test")

#making directory for alphabets

for i in ascii_uppercase:
    if not os.path.exists("dataset-alpha/train/"+i):
        os.makedirs("dataset-alpha/train/"+i)
    if not os.path.exists("dataset-alpha/test/"+i):
        os.makedirs("dataset-alpha/test/"+i)


mode = 'train/'
folder = 'dataset-alpha/'+mode
cam = cv2.VideoCapture(0)
while True:
    x,y=10,50
    _, frame = cam.read()
    frame = cv2.flip(frame,1)
    no_of_img = { }
    for i in ascii_uppercase:
        no_of_img[i] = len(os.listdir(folder + i))

    for i in ascii_uppercase:
        cv2.putText(frame, i+": " + str(no_of_img[i]), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        y = y+20
        if i == 'V':
            x = x+60
            y=50

    # cv2.putText(frame, "A: " + str(no_of_img['A']), (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    # cv2.putText(frame, "B: " + str(no_of_img['B']), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    # cv2.putText(frame, "C: " + str(no_of_img['C']), (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    # cv2.putText(frame, "D: " + str(no_of_img['D']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    # cv2.putText(frame, "E: " + str(no_of_img['E']), (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    cv2.rectangle(frame, (319, 9), (620 + 1, 309), (0, 255, 0), 1)
    roi = frame[10:300, 320:620]

    cv2.imshow("Frame", frame)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gaussblur = cv2.GaussianBlur(gray,(5,5),2)
    smallthres = cv2.adaptiveThreshold(gaussblur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,9,2.8)
    ret, final_image = cv2.threshold(smallthres, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    final_image = cv2.resize(final_image, (300, 300))
    cv2.imshow("BW", final_image)
    interrupt = cv2.waitKey(1) & 0xFF
    for j in ascii_uppercase:
        if interrupt == ord(j):
            cv2.imwrite(folder + j+'/' + str(no_of_img[j]) + '.jpg', final_image)
    if interrupt == 27:
        break
cam.release()
cv2.destroyAllWindows()