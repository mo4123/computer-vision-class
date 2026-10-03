import json
from pathlib import Path

import cv2
from ultralytics import YOLO


SOURCE = "https://ultralytics.com/images/bus.jpg"


TARGET = "person"


REQUIRED = 2


CONFIDENCE = 0.35



output_folder = Path("demo_results")
output_folder.mkdir(exist_ok=True)

annotated_path = output_folder / "annotated_detection.jpg"
report_path = output_folder / "detection_report.json"




print("Loading YOLO model...")

model = YOLO("yolo11n.pt")



print("\nSupported categories:")
print(model.names)



supported_names = list(model.names.values())

if TARGET not in supported_names:
    raise ValueError(
        f"'{TARGET}' is not a supported category.\n"
        f"Choose one of: {supported_names}"
    )



print("\nInspecting the image...")

results = model.predict(
    source=SOURCE,
    conf=CONFIDENCE,
    device="cpu"
)


result = results[0]




detections = []
labels = []

if result.boxes is not None and len(result.boxes) > 0:
    class_ids = result.boxes.cls.cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()
    coordinates = result.boxes.xyxy.cpu().tolist()

    for class_id, score, box in zip(
        class_ids,
        scores,
        coordinates
    ):
        label = model.names[int(class_id)]
        labels.append(label)

        x1, y1, x2, y2 = box

        detection = {
            "label": label,
            "confidence": round(float(score), 3),
            "box": {
                "x1": round(float(x1), 1),
                "y1": round(float(y1), 1),
                "x2": round(float(x2), 1),
                "y2": round(float(y2), 1)
            }
        }

        detections.append(detection)



count = labels.count(TARGET)

if count >= REQUIRED:
    message = (
        "Detected count meets the requirement; "
        "verify the picture."
    )
else:
    message = (
        "Detected count is below the requirement; "
        "review the picture."
    )



print("\nAll detected labels:")
print(labels)

print(f"\nTarget category: {TARGET}")
print(f"Detected count: {count}")
print(f"Required count: {REQUIRED}")
print(message)



print("\nDetection details:")

if detections:
    for number, detection in enumerate(detections, start=1):
        print(
            f"{number}. "
            f"{detection['label']} — "
            f"score {detection['confidence']} — "
            f"box {detection['box']}"
        )
else:
    print("No objects were detected above the confidence setting.")


annotated_image = result.plot()

saved = cv2.imwrite(
    str(annotated_path),
    annotated_image
)

if not saved:
    raise RuntimeError("Could not save the annotated image")



report = {
    "source": SOURCE,
    "model": "yolo11n.pt",
    "confidence_threshold": CONFIDENCE,
    "target": TARGET,
    "required": REQUIRED,
    "detected_count": count,
    "requirement_met": count >= REQUIRED,
    "message": message,
    "detections": detections
}

with report_path.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)


print(f"\nAnnotated image saved to: {annotated_path}")
print(f"JSON report saved to: {report_path}")



try:
    cv2.imshow(
        "Object detection - press Q to close",
        annotated_image
    )

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break

finally:
    cv2.destroyAllWindows()