#기울어진 오선에 대한 보정 camera.py
import os
import cv2
import modules
from detect import detect

# 이미지를 읽어옵니다.
resource_path = os.getcwd() + "/resources/"
src = cv2.imread(resource_path+"test.jpg")
stav_list = []

image = modules.deskew(src)
image_0, subimages = modules.remove_noise(image)

# stave_list : 해당 악보의 모든 오선 정보를 담고 있는 리스트
# normalized_images : 오선 제거된 분할 이미지와 오선 정보에 대해 정규화된 이미지를 가지고있는 리스트

normalized_images, stave_list = modules.digital_preprocessing(image_0,subimages)
for img in normalized_images:
    img = cv2.bitwise_not(img)
    results = detect(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)) # YOLO모델에는 BGR로 들어가야하기때문에 convert해서 넣어줌.
    temp_list = []
    # 각 탐지 결과에서 문자열 부분만 추출하여 result_list에 추가
    for item in results:
        note_type = item[0]  # 두 번째 요소 (음표 종류)
        temp_list.append(note_type)

    stav_list.append(temp_list)


print(results)

image_2 = cv2.bitwise_not(image_0)

# 이미지 띄우기
cv2.imshow('image2', normalized_images[1])
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()