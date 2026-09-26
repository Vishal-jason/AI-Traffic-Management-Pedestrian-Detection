import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "v7.mp4"
MODEL_PATH = "yolo11n.pt"

OUTPUT_PATH = "people_region.mp4"

# Person class in COCO
PERSON_CLASS = 0


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video:", VIDEO_PATH)
    exit()

# Original video information
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print()
print("========================================")
print("VIDEO INFORMATION")
print("========================================")
print("Width :", original_width)
print("Height:", original_height)
print("FPS   :", fps)
print("Frames:", total_frames)
print()


# ============================================================
# READ FIRST FRAME
# ============================================================

ret, first_frame = cap.read()

if not ret:
    print("ERROR: Could not read first frame.")
    cap.release()
    exit()

original_frame = first_frame.copy()


# ============================================================
# REGION DRAWING
# ============================================================

points = []

window_name = "Draw Region"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)


# ------------------------------------------------------------
# DISPLAY SIZE
#
# Change these if you want the drawing window larger/smaller.
# The actual video resolution is NOT changed.
# ------------------------------------------------------------

MAX_DISPLAY_WIDTH = 1500
MAX_DISPLAY_HEIGHT = 850


# This value will be updated
# whenever the frame is displayed.
current_scale = 1.0


# ============================================================
# CREATE DISPLAY FRAME
# ============================================================

def create_display_frame():

    global current_scale

    display = original_frame.copy()

    # --------------------------------------------------------
    # Draw points
    # --------------------------------------------------------

    for i, (x, y) in enumerate(points):

        cv2.circle(
            display,
            (x, y),
            6,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            display,
            str(i + 1),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    # --------------------------------------------------------
    # Draw lines between points
    # --------------------------------------------------------

    if len(points) >= 2:

        pts = np.array(
            points,
            dtype=np.int32
        )

        cv2.polylines(
            display,
            [pts],
            False,
            (0, 255, 255),
            3
        )

    # --------------------------------------------------------
    # Fill polygon when we have 3+ points
    # --------------------------------------------------------

    if len(points) >= 3:

        pts = np.array(
            points,
            dtype=np.int32
        )

        overlay = display.copy()

        cv2.fillPoly(
            overlay,
            [pts],
            (0, 255, 255)
        )

        display = cv2.addWeighted(
            overlay,
            0.15,
            display,
            0.85,
            0
        )

        cv2.polylines(
            display,
            [pts],
            True,
            (0, 255, 255),
            3
        )

    # --------------------------------------------------------
    # Calculate display scale
    # --------------------------------------------------------

    scale_x = MAX_DISPLAY_WIDTH / original_width
    scale_y = MAX_DISPLAY_HEIGHT / original_height

    current_scale = min(
        scale_x,
        scale_y,
        1.0
    )

    display_width = int(
        original_width * current_scale
    )

    display_height = int(
        original_height * current_scale
    )

    # --------------------------------------------------------
    # Resize ONLY the displayed copy
    #
    # The original frame remains full resolution.
    # --------------------------------------------------------

    display = cv2.resize(
        display,
        (
            display_width,
            display_height
        ),
        interpolation=cv2.INTER_AREA
    )

    return display


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global points
    global current_scale

    # --------------------------------------------------------
    # LEFT CLICK = ADD POINT
    # --------------------------------------------------------

    if event == cv2.EVENT_LBUTTONDOWN:

        # Convert display coordinates
        # back to original video coordinates.

        original_x = int(
            x / current_scale
        )

        original_y = int(
            y / current_scale
        )

        # Make sure coordinates stay inside frame

        original_x = max(
            0,
            min(
                original_x,
                original_width - 1
            )
        )

        original_y = max(
            0,
            min(
                original_y,
                original_height - 1
            )
        )

        points.append(
            (
                original_x,
                original_y
            )
        )

        print(
            f"Point {len(points)}: "
            f"({original_x}, {original_y})"
        )

    # --------------------------------------------------------
    # RIGHT CLICK = REMOVE LAST POINT
    # --------------------------------------------------------

    elif event == cv2.EVENT_RBUTTONDOWN:

        if len(points) > 0:

            removed = points.pop()

            print(
                "Removed point:",
                removed
            )


cv2.setMouseCallback(
    window_name,
    mouse_callback
)


# ============================================================
# DRAWING INSTRUCTIONS
# ============================================================

print("========================================")
print("DRAW REGION")
print("========================================")
print("LEFT CLICK  = Add point")
print("RIGHT CLICK = Remove last point")
print("R           = Reset region")
print("ENTER       = Finish region")
print("ESC         = Cancel")
print("========================================")
print()


# ============================================================
# DRAWING LOOP
# ============================================================

while True:

    display = create_display_frame()

    cv2.imshow(
        window_name,
        display
    )

    key = cv2.waitKey(20) & 0xFF

    # --------------------------------------------------------
    # ENTER
    # --------------------------------------------------------

    if key == 13:

        if len(points) < 3:

            print()
            print(
                "ERROR: Region needs at least "
                "3 points."
            )

            continue

        break

    # --------------------------------------------------------
    # R = RESET
    # --------------------------------------------------------

    elif key == ord("r"):

        points.clear()

        print(
            "Region reset."
        )

    # --------------------------------------------------------
    # ESC = CANCEL
    # --------------------------------------------------------

    elif key == 27:

        print(
            "Region drawing cancelled."
        )

        cap.release()
        cv2.destroyAllWindows()

        exit()


cv2.destroyWindow(
    window_name
)


# ============================================================
# CREATE REGION ARRAY
# ============================================================

region = np.array(
    points,
    dtype=np.int32
)

print()
print("========================================")
print("REGION CREATED")
print("========================================")
print("Number of points:", len(points))
print()
print(region)
print()


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(
    MODEL_PATH
)

print("Model loaded.")
print()


# ============================================================
# RESET VIDEO TO BEGINNING
# ============================================================

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    0
)


# ============================================================
# CREATE OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (
        original_width,
        original_height
    )
)

if not out.isOpened():

    print(
        "ERROR: Could not create output video."
    )

    cap.release()
    exit()


# ============================================================
# TRACKING VARIABLES
# ============================================================

unique_ids = set()

people_in_region = 0

frame_number = 0


# ============================================================
# PROCESS VIDEO
# ============================================================

print("========================================")
print("PROCESSING VIDEO")
print("========================================")
print()


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    annotated = frame.copy()

    # --------------------------------------------------------
    # YOLO TRACKING
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        classes=[PERSON_CLASS],
        conf=0.35,
        verbose=False
    )

    # Current frame region count
    people_in_region = 0

    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------

    if len(results) > 0:

        result = results[0]

        if result.boxes is not None:

            boxes = result.boxes

            # Get tracking IDs
            if boxes.id is not None:

                track_ids = (
                    boxes.id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                xyxy = (
                    boxes.xyxy
                    .cpu()
                    .numpy()
                )

                confidences = (
                    boxes.conf
                    .cpu()
                    .numpy()
                )

                # ------------------------------------------------
                # EACH PERSON
                # ------------------------------------------------

                for box, track_id, confidence in zip(
                    xyxy,
                    track_ids,
                    confidences
                ):

                    x1, y1, x2, y2 = map(
                        int,
                        box
                    )

                    # --------------------------------------------
                    # CENTER POINT
                    # --------------------------------------------

                    center_x = int(
                        (x1 + x2) / 2
                    )

                    center_y = int(
                        (y1 + y2) / 2
                    )

                    # --------------------------------------------
                    # CHECK REGION
                    # --------------------------------------------

                    inside = cv2.pointPolygonTest(
                        region,
                        (
                            center_x,
                            center_y
                        ),
                        False
                    )

                    is_inside = inside >= 0

                    # --------------------------------------------
                    # UNIQUE PEOPLE
                    # --------------------------------------------

                    unique_ids.add(
                        int(track_id)
                    )

                    # --------------------------------------------
                    # REGION COUNT
                    # --------------------------------------------

                    if is_inside:

                        people_in_region += 1

                    # --------------------------------------------
                    # BOX COLOR
                    # --------------------------------------------

                    if is_inside:

                        box_color = (
                            0,
                            255,
                            0
                        )

                    else:

                        box_color = (
                            255,
                            0,
                            0
                        )

                    # --------------------------------------------
                    # DRAW BOX
                    # --------------------------------------------

                    cv2.rectangle(
                        annotated,
                        (
                            x1,
                            y1
                        ),
                        (
                            x2,
                            y2
                        ),
                        box_color,
                        2
                    )

                    # --------------------------------------------
                    # DRAW CENTER
                    # --------------------------------------------

                    cv2.circle(
                        annotated,
                        (
                            center_x,
                            center_y
                        ),
                        5,
                        box_color,
                        -1
                    )

                    # --------------------------------------------
                    # TRACK ID + CONFIDENCE
                    # --------------------------------------------

                    label = (
                        f"id:{track_id} "
                        f"person "
                        f"{confidence:.2f}"
                    )

                    cv2.putText(
                        annotated,
                        label,
                        (
                            x1,
                            max(
                                y1 - 10,
                                20
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2
                    )

    # ========================================================
    # DRAW REGION
    # ========================================================

    cv2.polylines(
        annotated,
        [region],
        True,
        (
            0,
            255,
            255
        ),
        4
    )


    # ========================================================
    # DISPLAY COUNTERS
    # ========================================================

    cv2.putText(
        annotated,
        f"People in region: {people_in_region}",
        (
            30,
            60
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (
            0,
            255,
            0
        ),
        3
    )

    cv2.putText(
        annotated,
        f"Unique people: {len(unique_ids)}",
        (
            30,
            115
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (
            255,
            255,
            255
        ),
        3
    )


    # ========================================================
    # WRITE FRAME
    # ========================================================

    out.write(
        annotated
    )


    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_number % 30 == 0:

        print(
            f"Frame {frame_number}/{total_frames} | "
            f"In region: {people_in_region} | "
            f"Unique: {len(unique_ids)}"
        )


# ============================================================
# RELEASE
# ============================================================

cap.release()
out.release()

cv2.destroyAllWindows()


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("========================================")
print("Finished!")
print("========================================")
print(
    "Unique people:",
    len(unique_ids)
)
print(
    "People currently in region:",
    people_in_region
)
print(
    "Saved to:",
    OUTPUT_PATH
)
print("========================================")