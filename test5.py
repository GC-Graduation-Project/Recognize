#기울어진 오선에 대한 보정 camera.py
import os
import cv2
import modules
from detect import detect

# 이미지를 읽어옵니다.
resource_path = os.getcwd() + "/resources/"
src = cv2.imread(resource_path+"music4.jpg")

image = modules.deskew(src)
image_0, subimages = modules.remove_noise(image)

# stave_list : 해당 악보의 모든 오선 정보를 담고 있는 리스트
# normalized_images : 오선 제거된 분할 이미지와 오선 정보에 대해 정규화된 이미지를 가지고있는 리스트

normalized_images, stave_list = modules.digital_preprocessing(image_0,subimages)
normalized_images[1] = cv2.bitwise_not(normalized_images[1])

result = detect(cv2.cvtColor(normalized_images[1], cv2.COLOR_GRAY2BGR))

print(result)
image_2 = cv2.bitwise_not(image_0)

# 이미지 띄우기
cv2.imshow('image2', normalized_images[1])
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()