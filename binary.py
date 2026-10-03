import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Camera could not be opened")

mode = 1

print("Press 1: Original")
print("Press 2: Grayscale")
print("Press 3: Binary threshold")
print("Press 4: Canny edges (more edges)")
print("Press 5: Canny edges (fewer edges)")
print("Press Q: Quit")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read a frame")
            break

        # Convert colour image into grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Reduce small details and camera noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Brightness threshold:
        # Above 127 becomes white
        # 127 or below becomes black
        _, binary = cv2.threshold(
            gray,
            127,
            255,
            cv2.THRESH_BINARY
        )

        # Lower settings: usually show more edges
        edges_more = cv2.Canny(blurred, 20, 80)

        # Higher settings: usually show fewer edges
        edges_fewer = cv2.Canny(blurred, 100, 200)

        # Select what to display
        if mode == 1:
            display = frame
            title = "Original"

        elif mode == 2:
            display = gray
            title = "Grayscale"

        elif mode == 3:
            display = binary
            title = "Binary Threshold: 127"

        elif mode == 4:
            display = edges_more
            title = "Canny Edges: 20, 80"

        else:
            display = edges_fewer
            title = "Canny Edges: 100, 200"

        cv2.imshow(title, display)

        key = cv2.waitKey(1) 

        if key == ord("q"):
            break
        elif key == ord("1"):
            mode = 1
            cv2.destroyAllWindows()
        elif key == ord("2"):
            mode = 2
            cv2.destroyAllWindows()
        elif key == ord("3"):
            mode = 3
            cv2.destroyAllWindows()
        elif key == ord("4"):
            mode = 4
            cv2.destroyAllWindows()
        elif key == ord("5"):
            mode = 5
            cv2.destroyAllWindows()

finally:
    camera.release()
    cv2.destroyAllWindows()