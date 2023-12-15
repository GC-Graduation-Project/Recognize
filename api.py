from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import modules as md
import functions as fs

app = FastAPI()
@app.get("/")
def HelloWorld():
    return FastAPI

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    final_list = []

    image_0, subimages = md.remove_noise(img)

    normalized_images, stave_list = md.digital_preprocessing(image_0, subimages)

    rec_list, note_list, rest_list = md.beat_extraction(normalized_images)

    note_list2, pitch_list = md.pitch_extraction(stave_list, normalized_images)

    for rec, pitches in zip(rec_list, pitch_list):
        sharps, flats = fs.count_sharps_flats(rec)
        pitches = fs.modify_notes(pitches, sharps, flats)

    for note2, note1 in zip(note_list2, note_list):
        note2[1:] = fs.update_notes(note2[1:], note1)

    for list1, list2, list3 in zip(rec_list, note_list2, pitch_list):
        m_list = fs.merge_three_lists(list1, list2, list3)
        final_list.append(m_list)

    print(final_list)

    sen = fs.convert_to_sentence(final_list)

    print(sen)

    # 결과를 JSON 형식으로 반환
    return JSONResponse(content={"results": sen})
