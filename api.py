from fastapi import FastAPI, UploadFile, File
from fastapi.responses import PlainTextResponse
import cv2
import numpy as np
import modules as md
import functions as fs

app = FastAPI()

@app.get("/")
def hello_world():
    return {"message": "Hello World"}

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # 파일 내용 읽기
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 이미지 처리 및 분석
        final_list = []
        image_0, subimages = md.remove_noise(img)
        normalized_images, stave_list = md.digital_preprocessing(image_0, subimages)
        rec_list, note_list, rest_list = md.beat_extraction(normalized_images)
        note_list2, pitch_list = md.pitch_extraction(stave_list, normalized_images)

        # 음표 처리 및 기타 탭 변환
        for i, (rec, pitches) in enumerate(zip(rec_list, pitch_list)):
            sharps, flats = fs.count_sharps_flats(rec)
            temp_dict = {}
            modified_pitches = fs.modify_notes(pitches, sharps, flats)
            for pit in modified_pitches:
                positions = fs.get_guitar(pit)
                temp_dict[pit] = positions
            modified_pitches = fs.calculate_efficient_positions(modified_pitches, temp_dict)
            pitch_list[i] = modified_pitches

        for note2, note1 in zip(note_list2, note_list):
            note2[1:] = fs.update_notes(note2[1:], note1)

        for list1, list2, list3 in zip(rec_list, note_list2, pitch_list):
            m_list = fs.merge_three_lists(list1, list2, list3)
            final_list.append(m_list)

        # 결과 문자열 생성
        sen = fs.convert_to_sentence(final_list)

        # 텍스트 형식으로 반환
        return PlainTextResponse(content=sen)
    except Exception as e:
        return PlainTextResponse(content=str(e), status_code=500)
