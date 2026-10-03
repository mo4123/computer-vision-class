import cv2
import numpy as np

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        break

    # 1. Convert BGR image to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Define red color range
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # 3. Find red pixels
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2

    # 4. Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # 5. Find largest red object
        largest_contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(largest_contour)

        # Ignore very small red objects/noise
        if area > 500:
            # Find enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)

            center = (int(x), int(y))
            radius = int(radius)

            # Draw circle
            cv2.circle(frame, center, radius, (0, 255, 0), 2)

            # Add text
            cv2.putText(
                frame,
                "Red Ball Detected",
                (center[0] - 80, center[1] - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("Red Ball Detection", frame)
    cv2.imshow("Red Mask", mask)

    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()