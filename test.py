from typing import List, Any

import cv2
import os
import numpy as np
import functions as fs
import modules as md


resource_path = os.getcwd() + "/resources/"
src = cv2.imread(resource_path + "music.jpg")
final_list = []

image = md.deskew(src)
image_0, subimages = md.remove_noise(image)

normalized_images, stave_list = md.digital_preprocessing(image_0, subimages)

rec_list, note_list, rest_list = md.beat_extraction(normalized_images)

note_list2, pitch_list = md.pitch_extraction(stave_list, normalized_images)

sharps, flats = fs.count_sharps_flats(rec_list[0])


for pitches in pitch_list:
    pitches = fs.modify_notes(pitches, sharps, flats)

for note2, note1 in zip(note_list2, note_list):
    note2[1:] = fs.update_notes(note2[1:], note1)

for list1, list2, list3 in zip(rec_list, note_list2, pitch_list):
    m_list = fs.merge_three_lists(list1, list2, list3)
    final_list.append(m_list)

print(final_list)

