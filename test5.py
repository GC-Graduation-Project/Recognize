#기울어진 오선에 대한 보정 camera.py
import os
import cv2
import modules
from pitchDetection import detect
import functions as fs

# 이미지를 읽어옵니다.
resource_path = os.getcwd() + "/resources/"
src = cv2.imread(resource_path+"music.jpg")
original_list = []
final_result = []
sentence = []
temp_dict = {}
ind = 0
image = modules.deskew(src)
image_0, subimages = modules.remove_noise(image)

# stave_list : 해당 악보의 모든 오선 정보를 담고 있는 리스트
# normalized_images : 오선 제거된 분할 이미지와 오선 정보에 대해 정규화된 이미지를 가지고있는 리스트

normalized_images, stave_list = modules.digital_preprocessing(image_0,subimages)
for img in normalized_images:
    img = cv2.bitwise_not(img)
    result = detect(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)) # YOLO모델에는 BGR로 들어가야하기때문에 convert해서 넣어줌.
    print(result)
    ori_tmp_list, note_tmp_list = fs.mapping_notes(stave_list[ind], result)
    original_list.append(ori_tmp_list)
    final_result.append(note_tmp_list)
    ind += 1

print(original_list)
print(final_result)
for notes in final_result:
    note_list = notes
    temp_dict = {}
    for note in note_list:
        positions = fs.get_number_upgrade(note)
        temp_dict[note] = positions
    ans = fs.calculate_efficient_positions(note_list, temp_dict)
    sentence.append(ans)


print(sentence)