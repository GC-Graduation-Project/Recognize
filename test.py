import cv2
import os
import functions as fs
import modules as md
import pitchDetection

resource_path = os.getcwd() + "/resources/"
src = cv2.imread(resource_path + "music1.jpg")
final_list = []

image = md.deskew(src)
image_0, subimages = md.remove_noise(image)

normalized_images, stave_list = md.digital_preprocessing(image_0, subimages)

rec_list, note_list, rest_list = md.beat_extraction(normalized_images)

print(note_list)

print(rec_list)

clef_list = pitchDetection.detect1(cv2.cvtColor(cv2.bitwise_not(image_0),cv2.COLOR_GRAY2BGR))

note_list2, pitch_list = md.pitch_extraction(stave_list, normalized_images, clef_list)

print(note_list2)
print(pitch_list)

for i, (rec, pitches) in enumerate(zip(rec_list, pitch_list)):
    sharps, flats = fs.count_sharps_flats(rec)
    temp_dict = {}
    modified_pitches = fs.modify_notes(pitches, sharps, flats)
    for pit in modified_pitches:
        positions = fs.get_guitar(pit)
        temp_dict[pit] = positions
    modified_pitches = fs.calculate_efficient_positions(modified_pitches, temp_dict)
    pitch_list[i] = modified_pitches  # Update the pitch_list with modified pitches

for note2, note1 in zip(note_list2, note_list):
    note2[1:] = fs.update_notes(note2[1:], note1)
print(note_list2)
for list1, list2, list3 in zip(rec_list, note_list2, pitch_list):
    m_list = fs.merge_three_lists(list1, list2, list3)
    final_list.append(m_list)

print(final_list)

sen = fs.convert_to_sentence(final_list)

print(sen)