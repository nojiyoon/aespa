import base64
import io
import cv2
import keras
import numpy as np
from PIL import Image
# from keras.backend import tensorflow_backend as backend
import tensorflow.keras.backend as backend # tensorflow 2.0에서 변경
from django.conf import settings

from matplotlib import pyplot as plt
import numpy as np
from PIL import ExifTags
import piexif



def detect(upload_image):
    result_name = upload_image.name
    result_list = []
    result_img = ''

    # 설정에서 Cascade 파일의 경로 취득
    cascade_file_path = settings.CASCADE_FILE_PATH
    # 설정에서 모델 파일의 경로 취득
    model_file_path = settings.MODEL_FILE_PATH
    # Keras의 모델을 읽어오기
    model = keras.models.load_model(model_file_path)
    # 회전되는 이미지가 있다면 원래 방향으로 설정
    image = Image.open(upload_image)
    image = rotate(image)
    # 업로드된 이미지 파일을 메모리에서 OpenCV 이미지로 저장
    # image = np.asarray(Image.open(upload_image))
    image = np.asarray(image)
    # size 700 이상 이미지일때 비율 유지하면서 가로를 700으로 축소
    ratio = 700.0 / image.shape[1]
    dim = (700, int(image.shape[0] * ratio))
    if image.shape[0] > 700:
            image = cv2.resize(image, dim)
    # 화상을 OpenCV BGR에서 RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 화상을 RGB에서 GRAY로 변환
    image_gs = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # Cascade 파일을 읽어오기
    cascade = cv2.CascadeClassifier(cascade_file_path)
    # OpenCV를 이용해 얼굴 인식
    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1,
                                         minNeighbors=10, minSize=(64, 64))

    # 얼굴이 1개 이상인 경우
    if len(face_list) > 0:
        count = 1
        for (xpos, ypos, width, height) in face_list:
            # 인식한 얼굴을 잘라냄
            face_image = image_rgb[ypos:ypos+height, xpos:xpos+width]
            # 잘라낸 얼굴이 너무 작으면 스킵
            if face_image.shape[0] < 64 or face_image.shape[1] < 64:
                continue
            # 인식한 얼굴의 사이즈를 축소
            face_image = cv2.resize(face_image, (64, 64))
            # 인식한 얼굴 주변을 붉은 색으로 표시
            cv2.rectangle(image_rgb, (xpos, ypos),
                          (xpos+width, ypos+height), (0, 0, 255), thickness=2)
            # 인식한 얼굴을 1장의 화상 이미지로 합하는 배열로 변환
            face_image = np.expand_dims(face_image, axis=0)
            # 인식한 얼굴에서 이름을 추출
            name, result = detect_who(model, face_image)
            # 인식한 얼굴에 이름을 추가
            cv2.putText(image_rgb, f"{count}. {name}", (xpos, ypos+height+20),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            # 결과를 리스트에 저장
            result_list.append(result)
            count = count + 1

    # 화상을 PNG로 변환
    is_success, img_buffer = cv2.imencode(".png", image_rgb)
    if is_success:
        # 화상을 인메모리의 바이너리 스트림으로 전달
        io_buffer = io.BytesIO(img_buffer)
        # 인메모리의 바이너리 스트림에서 BASE64 인코드 변환
        result_img = base64.b64encode(io_buffer.getvalue()).decode().replace("'", "")

    # tensorflow의 백엔드 클리어
    backend.clear_session()
    # 결과 반환
    return (result_list, result_name, result_img)

def detect_who(model, face_image):
    # 예측
    predicted = model.predict(face_image)
    # 결과
    name = ""
    result = f"카리나일 가능성:{predicted[0][0]*100:.3f}% / 윈터일 가능성:{predicted[0][1]*100:.3f}% / 지젤일 가능성:{predicted[0][2]*100:.3f}% / 닝닝일 가능성:{predicted[0][3]*100:.3f}%"
    name_number_label = np.argmax(predicted)
    if name_number_label == 0:
        name = "Karina"
    elif name_number_label == 1:
        name = "Winter"
    elif name_number_label == 2:
        name = "Giselle"
    elif name_number_label == 3:
        name = "Ningning"
    return (name, result)



def rotate(img):
    if "exif" in img.info:
        exif_dict = piexif.load(img.info["exif"])
        #print(exif_dict)
        if piexif.ImageIFD.Orientation in exif_dict["0th"]:
            orientation = exif_dict["0th"].pop(piexif.ImageIFD.Orientation)
            exif_bytes = piexif.dump(exif_dict)
            #print('{} orientation value is {}'.format(filename,str(orientation)))
            
            if orientation == 2:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            elif orientation == 3:
                img = img.rotate(180)

            elif orientation == 4:
                img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)

            elif orientation == 5:
                img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)

            elif orientation == 6:
                img = img.rotate(-90, expand=True)

            elif orientation == 7:
                img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)

            elif orientation == 8:
                img = img.rotate(90, expand=True)

    return img