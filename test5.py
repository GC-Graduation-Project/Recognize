#기울어진 오선에 대한 보정 camera.py
import os
import numpy as np
import cv2
import functions as fs
import modules

# 이미지를 읽어옵니다.
resource_path = os.getcwd() + "/resources/"
src = cv2.imread(resource_path+"music10.jpg")

image = modules.deskew(src)
image_0, subimages = modules.camera_remove_noise(image)

# 2. 오선 제거
image_2, staves = modules.remove_staves(image_0)

# 3. 악보 이미지 정규화
image_3, staves = modules.normalization(image_2, staves, 10)

print(len(staves))

# 이미지 띄우기
cv2.imshow('image', image_2)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()