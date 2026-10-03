import cv2

camera = cv2.VideoCapture(0) #[True,(480,640,3)]

while True:
    
    success, frame = camera.read()

    if not success:
        print("Camera could not be opened")
        break

    
    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1)  == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()