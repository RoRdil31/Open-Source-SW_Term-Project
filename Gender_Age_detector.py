import cv2

video_capture = cv2.VideoCapture(0)  # 비디오 캡처를 활성화
faceCascade = cv2.CascadeClassifier('style.xml')  # 얼굴 인식에 사용할 Cascade 분류기 로드
video_capture.set(3, 480)  # 프레임의 너비 설정
video_capture.set(4, 640)  # 프레임의 높이 설정
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  # 평균값(모델에 필요한 값)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']  # 나이 범위
gender_list = ['Male', 'Female']  # 성별 목록

def main():
    while True:
        # 비디오 프레임 읽기
        check, frame = video_capture.read()  
        frame = cv2.flip(frame, 1)  # 프레임을 좌우 반전
        cv2.imshow('frame', frame)  # 프레임 표시

        # ESC 키가 눌리면 종료
        if cv2.waitKey(1) & 0xFF == 27:  
            break

    # 리소스 해제
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()