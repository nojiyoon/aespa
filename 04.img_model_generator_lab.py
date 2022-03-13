import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
# from keras.utils.vis_utils import plot_model
# tensorflow의 keras는 import에 pydot_ng, pydotplus를 import되는 것처럼 기술되어 있지만,
# keras(ver=2.2.0)은 import할 때, pydot만 import하는 것처럼 명시 되어 있어서 아래를 추가로 기술
from tensorflow.python.keras.utils.vis_utils import plot_model

def load_images(image_directory):
    image_file_list = []
    # 지정한 디렉토리내 파일 추출
    image_file_name_list = os.listdir(image_directory)
    print(f"대상 이미지 파일수:{len(image_file_name_list)}")
    for image_file_name in image_file_name_list:
        # 이미지 파일 경로
        image_file_path = os.path.join(image_directory, image_file_name)
        print(f"이미지 파일 경로:{image_file_path}")
        # 이미지 읽기
        image = cv2.imread(image_file_path)
        if image is None:
            print(f"이미지 파일[{image_file_name}]을 읽을 수 없습니다.")
            continue
        image_file_list.append((image_file_name, image))
    print(f"읽은 이미지 수:{len(image_file_list)}")
    return image_file_list

def labeling_images(image_file_list):
    x_data = []
    y_data = []
    for idx, (file_name, image) in enumerate(image_file_list):
        # 이미지를 BGR 형식에서 RGB 형식으로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 이미지 배열(RGB 이미지)
        x_data.append(image)
        # 이블 배열(파일명의 앞 2글자를 레이블로 이용)
        label = int(file_name[0:2]) - 1
        print(f"레이블:{label:02} 이미지 파일명:{file_name}")
        y_data = np.append(y_data, label).reshape(idx+1, 1)
    x_data = np.array(x_data)
    print(f"레이블링 이미지 수:{len(x_data)}")
    return (x_data, y_data)

def delete_dir(dir_path, is_delete_top_dir=True):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if is_delete_top_dir:
        os.rmdir(dir_path)

RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Outoput Model Only
OUTPUT_MODEL_ONLY = False
# Test Image Directory
TEST_IMAGE_DIR = "./test_image"
# Train Image Directory
TRAIN_IMAGE_DIR = "./face_scratch_image"
# Output Model Directory
OUTPUT_MODEL_DIR = "./model"
# Output Model File Name
OUTPUT_MODEL_FILE = "model.h5"
# Output Plot File Name
OUTPUT_PLOT_FILE = "model.png"

def main():
    print("===================================================================")
    print("Keras를 이용한 모델 학습 ")
    print("지정한 이미지 파일을 학습하는 모델 생성")
    print("===================================================================")

    # 디렉토리 작성
    if not os.path.isdir(OUTPUT_MODEL_DIR):
        os.mkdir(OUTPUT_MODEL_DIR)
    # 디렉토리 내 파일 삭제
    delete_dir(OUTPUT_MODEL_DIR, False)

    num_classes = 4
    batch_size = 32
    epochs = 30

    # 학습용 이미지 파일 읽기
    train_file_list = load_images(TRAIN_IMAGE_DIR)
    # 학습용 이미지 파일 레이블 처리
    x_train, y_train = labeling_images(train_file_list)

    # plt.imshow(x_train[0])
    # plt.show()
    # print(y_train[0])

    # 테스트용 이미지 파일 읽기
    test_file_list = load_images(TEST_IMAGE_DIR)
    # 테스트용 이미지 파일의 레이블 처리
    x_test, y_test = labeling_images(test_file_list)

    # plt.imshow(x_test[0])
    # plt.show()
    # print(y_test[0])

    # 이미지와 레이블의 배열 확인: 2차원 배열
    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)
    print("x_test.shape:", x_test.shape)
    print("y_test.shape:", y_test.shape)

    # 분류 레이블의 1-hot encoding처리(선형 분류를 쉽게 하기 위해)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # 이미지와 레이블의 배열 차수 확인 -> 2차원
    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)
    print("x_test.shape:", x_test.shape)
    print("y_test.shape:", y_test.shape)

    # 모델 정의
    model = Sequential()
    model.add(Conv2D(input_shape=(64, 64, 3), filters=32, kernel_size=(3,3), strides=(1,1), 
                    padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # 합성곱2
    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), 
                    padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.3))

    # 합성곱3
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), 
                    padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # 모델 시각화
    # 학습
    if OUTPUT_MODEL_ONLY:
        # 학습
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    else:
        # 학습 그래프
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                            verbose=1, validation_data=(x_test, y_test))
        # 일반화 정도의 평가/표시
        test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
        print(f"validation loss:{test_loss}\r\nvalidation accuracy:{test_acc}")

        # acc(정확도), val_acc(검증용 데이터 정확도) 그래프
        plt.plot(history.history["accuracy"], label="acuuracy", ls="-", marker="o")
        plt.plot(history.history["val_accuracy"], label="val_acuuracy", ls="-", marker="x")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accruracy")
        plt.legend(loc="best")
        plt.show()

        # 손실 기록 그래프
        plt.plot(history.history["loss"], label="loss", ls="-", marker="o")
        plt.plot(history.history["val_loss"], label="val_loss", ls="-", marker="x")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="lower right")
        plt.show()

    # 모델 저장
    model_file_path = os.path.join(OUTPUT_MODEL_DIR, OUTPUT_MODEL_FILE)
    model.save(model_file_path)

    return RETURN_SUCCESS

if __name__ == "__main__":
    main()