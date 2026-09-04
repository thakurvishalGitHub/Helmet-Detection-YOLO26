from ultralytics import YOLO
import gradio as gr
import cv2

MODEL_PATH = "models/best.pt"
CONFIDENCE = 0.40
DEVICE = 0   # GPU use karega; agar issue ho to "cpu" kar dena

model = YOLO(MODEL_PATH)


def count_detections(result):
    helmet_count = 0
    no_helmet_count = 0

    if result.boxes is not None and result.boxes.cls is not None and len(result.boxes) > 0:
        class_ids = result.boxes.cls.tolist()
        names = result.names

        for class_id in class_ids:
            class_name = names[int(class_id)]

            if class_name == "Helmet":
                helmet_count += 1
            elif class_name == "No Helmet":
                no_helmet_count += 1

    return helmet_count, no_helmet_count


def detect_live(frame):
    if frame is None:
        return None, "Camera not started yet."

    # Gradio frame is RGB, OpenCV/YOLO flow ke liye BGR me convert
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = model.predict(
        source=frame_bgr,
        conf=CONFIDENCE,
        device=DEVICE,
        verbose=False
    )

    result = results[0]
    helmet_count, no_helmet_count = count_detections(result)

    annotated_frame = result.plot()   # BGR
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    summary = f"Helmet: {helmet_count} | No Helmet: {no_helmet_count}"

    return annotated_frame, summary


with gr.Blocks(title="Helmet Detection Live") as demo:
    gr.Markdown("# ⛑️ Helmet Detection System")
    gr.Markdown("Live webcam helmet / no-helmet detection")

    with gr.Row():
        input_cam = gr.Image(
            sources=["webcam"],
            type="numpy",
            streaming=True,
            label="Live Camera"
        )

        output_img = gr.Image(
            type="numpy",
            label="Live Detection"
        )

    summary_box = gr.Textbox(
        label="Detection Summary",
        value="Start webcam to begin detection"
    )

    input_cam.stream(
        fn=detect_live,
        inputs=input_cam,
        outputs=[output_img, summary_box],
        stream_every=0.1
    )

demo.launch(share=True)