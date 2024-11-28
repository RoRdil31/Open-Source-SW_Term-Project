import cv2

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 480)
video_capture.set(4, 640)

# 얼굴 인식에 사용할 Cascade 분류기 로드
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def main():
    while True:
        check, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        
        # 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 탐지
        faces = faceCascade.detectMultiScale(gray, 1.2, 10)
        
        # 얼굴 주변에 사각형 그리기
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
