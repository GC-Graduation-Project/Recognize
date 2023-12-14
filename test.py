import cv2
import os
import numpy as np
import functions as fs
import modules


resource_path = os.getcwd() + "/resources/"
src = cv2.imread(resource_path + "music.jpg")

image = modules.deskew(src)
image_0, subimages = modules.remove_noise(image)

normalized_images, stave_list = modules.digital_preprocessing(image_0, subimages)

list1,list2,list3 = modules.beat_extraction(normalized_images)

print(list1)
print(list2)
print(list3)