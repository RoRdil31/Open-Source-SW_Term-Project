import cv2

video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('style.xml')  # 얼굴 인식에 사용할 Cascade 분류기 로드
video_capture.set(3, 480)  # 프레임의 너비 설정
video_capture.set(4, 640)  # 프레임의 높이 설정MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return (age_net, gender_net)


def video_detector(age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX  # 글꼴 유형 설정
    while True:
        check, frame = video_capture.read()  # 비디오 프레임 읽기
        frame = cv2.flip(frame, 1)  # 프레임을 좌우 반전
        # 웹캠 피드를 그레이스케일로 변환 (OpenCV의 대부분 작업은 그레이스케일에서 수행됨)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 10, cv2.CASCADE_SCALE_IMAGE, (30, 30))  # 얼굴 탐지
        # print(check)
        # print(frame)
        # 얼굴 주변에 사각형 그리기
        for (x, y, w, h) in faces:
            # 얼굴 영역에 사각형 추가
            # print(x, y, w, h) # 얼굴의 좌표 출력
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            # 얼굴 영역을 행렬로 가져와 복사
            face_img = frame[y:y + h, h:h + w].copy()
            print(face_img)
            blob = cv2.dnn.blobFromImage(face_img, 1, (244, 244), MODEL_MEAN_VALUES, swapRB=True)  # Blob 생성
            # 성별 예측
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            # 나이 예측
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            overlay_text = "%s %s" % (gender, age)
            cv2.putText(frame, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)  # 텍스트 추가
            cv2.imshow('frame', frame)  # 프레임 표시
        key = cv2.waitKey(1)
        if key == 27:  # ESC 키가 눌리면 종료
            break
    video_capture.release()
    cv2.destroyAllWindows()


def main():
    age_net, gender_net = load_caffe_models()
    video_detector(age_net, gender_net)

if __name__ == "__main__":
    main()
