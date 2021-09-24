# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# pip install opencv-python

import cv2
import numpy as np

photo = np.zeros((300, 300, 3), dtype='uint8')
# RGB - standard, BGR - format in OpenCV
photo[:] = 119, 201, 105  # everything green
photo[100:150, 200:250] = 5, 5, 255  # red rectangle
cv2.rectangle(photo, (5, 5), (100, 100), (255, 10, 10), thickness=cv2.FILLED)
cv2.line(photo, (0, photo.shape[0]//2), (photo.shape[1], photo.shape[0]//2), (50, 50, 255))
cv2.circle(photo, (photo.shape[1]//2, photo.shape[0]//2), 50, thickness=cv2.FILLED, color=(255, 10, 10))
cv2.putText(photo, 'IMAGE', (100, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 3)
cv2.imshow('Photo', photo)
cv2.waitKey(0)

# image
img = cv2.imread('images/zebra.jpg')
# img = cv2.flip(img, -1)

def rotate(img_param, angle):
    height, width = img_param.shape[:2]
    point = (width//2, height//2)
    matrix = cv2.getRotationMatrix2D(point, angle, 1)
    return cv2.warpAffine(img, matrix, (width, height))

# img = rotate(img, 45)

def transform(img_param, x, y):
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(img_param, matrix, (img_param.shape[1], img_param.shape[0]))

# img = transform(img, 30, 50)

cv2.imshow('Two zebras', img)
print(img.shape)  # show resolution of image (height, width, layers (3 - RGB))
new_img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))  # resizing image: bigger in 2 times
new_img = cv2.GaussianBlur(new_img, (9, 9), 0)  # blur
new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)  # make it white-black
new_img = cv2.Canny(new_img, 100, 100)  # make it binary image (it's better after making it white-black)
kernel = np.ones((5, 5), np.uint8)  # matrix for dots
new_img = cv2.dilate(new_img, kernel, iterations=1)
new_img = cv2.erode(new_img, kernel, iterations=1)
cv2.imshow('Resized two zebra', new_img[0:500, 150:650])  # crop and show resized image
cv2.waitKey(0)  # show forever, 1000 - 1 second

# video
cap = cv2.VideoCapture('videos/balerina.mp4')
cap.set(3, 500)
cap.set(4, 300)

while True:
    success, img = cap.read()

    # new_img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))  # resizing image: bigger in 2 times
    img = cv2.GaussianBlur(img, (9, 9), 0)  # blur
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # make it white-black
    img = cv2.Canny(img, 100, 100)  # make it binary image (it's better after making it white-black)
    kernel = np.ones((5, 5), np.uint8)  # matrix for dots
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    cv2.imshow('Balerina dancing', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
