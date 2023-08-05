import cv2
import os
save_path = 'C:/Users/BABLU/Desktop/OUR PROJECT/newdataset'
person_name = input("Enter the student's rollno: ")
person_dir = os.path.join(save_path, person_name)
if not os.path.exists(person_dir):
    os.makedirs(person_dir)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
count = 0
while count < 500:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_img = frame[y:y+h, x:x+w]
        filename = os.path.join(person_dir, f'{person_name}_{count}.jpg')
        cv2.imwrite(filename, face_img)
        count += 1
    
    cv2.imshow('Capture Images', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print('Entered Successfully')
