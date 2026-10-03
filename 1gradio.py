import gradio as gr
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

def detect_objects(image):
    result = model.predict(
        image,
        conf=0.35,
        verbose=False
    )[0]

    return result.plot()

app = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(),
    outputs=gr.Image(),
    title="Object Detection Application"
)

app.launch()