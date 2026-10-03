import cv2
camera = cv2.VideoCapture(0)
try:
    if not camera.isOpened():
        raise RuntimeError("Camera unavailable")
    while True:
        success, frame = camera.read()
        if not success:
            print("Could not read a frame")
            break
        cv2.imshow("Webcam - press Q", frame)
        if cv2.waitKey(1) == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()