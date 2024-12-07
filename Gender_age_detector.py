import cv2

video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return age_net, gender_net

def main():
    age_net, gender_net = load_caffe_models()
    
    while True:
        check, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 10)

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (244, 244), MODEL_MEAN_VALUES, swapRB=True)

            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            
            overlay_text = "%s %s" % (gender, age)
            cv2.putText(frame, overlay_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
