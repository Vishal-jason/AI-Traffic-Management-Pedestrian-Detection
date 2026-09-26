from ultralytics import YOLO
import cv2

MODEL = "runs/detect/train/weights/best.pt"
VIDEO = "v1.mp4"
OUTPUT = "people_counted.mp4"

model = YOLO(MODEL)

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

counted_ids = set()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        verbose=False
    )

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.int().cpu().tolist()

        for track_id in ids:
            counted_ids.add(track_id)

    annotated = results[0].plot()

    cv2.putText(
        annotated,
        f"People: {len(counted_ids)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    writer.write(annotated)

cap.release()
writer.release()

print(f"Done. Unique tracked IDs: {len(counted_ids)}")
print(f"Saved to: {OUTPUT}")