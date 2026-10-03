import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")


camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame")
        break

    result = model.predict(
        source=frame,
        conf=0.35,
        verbose=False
    )[0]

 
    frame_with_boxes = result.plot()

    cv2.imshow("Object Detection - press Q", frame_with_boxes)

    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()