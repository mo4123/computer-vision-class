import cv2

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        print("Camera frame could not be captured")
        break

    frame = cv2.flip(frame, -1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    cv2.putText(
        frame,
        "Press q to quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Original Webcam", frame)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Edge Detection", edges)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
