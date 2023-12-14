#기울어진 오선에 대한 보정 camera.py
import os
import cv2
import modules
from pitchDetection import detect, detect1
import functions as fs

# 이미지를 읽어옵니다.
resource_path = os.getcwd() + "/resources/"
src = cv2.imread(resource_path+"music.jpg")
original_list = []
final_result = []
clef_list = []
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
    result2 = detect1(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    clef_list.append(result2)
    original_list.append(result)


for clef, notes in zip(clef_list, original_list):
    notes.insert(0, clef[0])
    notes = fs.add_dot(notes)
    note_tmp_list = fs.mapping_notes(stave_list[ind], notes)
    final_result.append(note_tmp_list)
    ind += 1


for notes in final_result:
    note_list = notes
    temp_dict = {}
    for note in note_list:
        positions = fs.get_guitar(note)
        temp_dict[note] = positions
    ans = fs.calculate_efficient_positions(note_list, temp_dict)
    sentence.append(ans)

