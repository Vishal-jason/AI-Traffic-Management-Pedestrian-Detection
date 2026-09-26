import cv2
import numpy as np
from ultralytics import YOLO
import os


INPUT_VIDEO = r"D:\people\v4" \
".mp4"

# Use the model you already have in your project
MODEL_PATH = r"D:\people\yolo11n.pt"

OUTPUT_VIDEO = r"D:\people\people_crossing_time.mp4"


MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 800



NORMAL_RED_SIGNAL = 20.0

PEDESTRIAN_BUFFER_PER_PERSON = 2.0



drawing_points = []
drawing_done = False

original_width = 0
original_height = 0

display_width = 0
display_height = 0

scale_x = 1.0
scale_y = 1.0


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global drawing_points

    if drawing_done:
        return

    if event == cv2.EVENT_LBUTTONDOWN:

        # ----------------------------------------------------
        # IMPORTANT:
        # x,y are coordinates on the SMALL DISPLAY IMAGE.
        #
        # Convert them back to ORIGINAL VIDEO coordinates.
        # ----------------------------------------------------

        original_x = int(x / scale_x)
        original_y = int(y / scale_y)

        # Keep coordinates inside original frame
        original_x = max(
            0,
            min(original_x, original_width - 1)
        )

        original_y = max(
            0,
            min(original_y, original_height - 1)
        )

        drawing_points.append(
            (original_x, original_y)
        )

        print(
            f"Point {len(drawing_points)}: "
            f"({original_x}, {original_y})"
        )


# ============================================================
# DRAW REGION
# ============================================================

def draw_region(first_frame):

    global drawing_done
    global display_width, display_height
    global scale_x, scale_y

    frame_height, frame_width = first_frame.shape[:2]

    # --------------------------------------------------------
    # Calculate display size while keeping aspect ratio
    # --------------------------------------------------------

    scale = min(
        MAX_DISPLAY_WIDTH / frame_width,
        MAX_DISPLAY_HEIGHT / frame_height,
        1.0
    )

    display_width = int(
        frame_width * scale
    )

    display_height = int(
        frame_height * scale
    )

    scale_x = display_width / frame_width
    scale_y = display_height / frame_height

    print()
    print("=" * 60)
    print("DRAW REGION")
    print("=" * 60)
    print("Click points around the region.")
    print("ENTER = finish region")
    print("R     = reset")
    print("ESC   = exit")
    print()
    print(
        f"Original video size : "
        f"{frame_width} x {frame_height}"
    )
    print(
        f"Display size        : "
        f"{display_width} x {display_height}"
    )
    print()

    cv2.namedWindow(
        "Draw Region",
        cv2.WINDOW_NORMAL
    )

    # Force window to display size
    cv2.resizeWindow(
        "Draw Region",
        display_width,
        display_height
    )

    cv2.setMouseCallback(
        "Draw Region",
        mouse_callback
    )

    while True:

        # ----------------------------------------------------
        # Resize ONLY the image used for drawing.
        # Original frame remains untouched.
        # ----------------------------------------------------

        display_frame = cv2.resize(
            first_frame,
            (display_width, display_height),
            interpolation=cv2.INTER_AREA
        )

        # ----------------------------------------------------
        # Draw polygon using converted coordinates
        # ----------------------------------------------------

        if len(drawing_points) > 0:

            display_points = []

            for px, py in drawing_points:

                dx = int(
                    px * scale_x
                )

                dy = int(
                    py * scale_y
                )

                display_points.append(
                    (dx, dy)
                )

            # Draw points
            for i, point in enumerate(
                display_points
            ):

                cv2.circle(
                    display_frame,
                    point,
                    5,
                    (0, 255, 255),
                    -1
                )

                cv2.putText(
                    display_frame,
                    str(i + 1),
                    (
                        point[0] + 5,
                        point[1] - 5
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )

            # Draw connecting lines
            if len(display_points) >= 2:

                for i in range(
                    len(display_points) - 1
                ):

                    cv2.line(
                        display_frame,
                        display_points[i],
                        display_points[i + 1],
                        (0, 255, 255),
                        2
                    )

            # Close polygon preview
            if len(display_points) >= 3:

                cv2.line(
                    display_frame,
                    display_points[-1],
                    display_points[0],
                    (0, 255, 255),
                    2
                )

        # ----------------------------------------------------
        # Instructions
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            "Click points | ENTER = Start | R = Reset | ESC = Exit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Draw Region",
            display_frame
        )

        key = cv2.waitKey(20) & 0xFF

        # ENTER
        if key == 13:

            if len(drawing_points) >= 3:

                drawing_done = True

                print()
                print("Region selected.")
                print("Original-resolution points:")

                for i, point in enumerate(
                    drawing_points
                ):

                    print(
                        f"Point {i + 1}: {point}"
                    )

                break

            else:

                print(
                    "Please select at least 3 points."
                )

        # R = reset
        elif key == ord("r"):

            drawing_points.clear()

            print("Region reset.")

        # ESC
        elif key == 27:

            cv2.destroyAllWindows()

            return None

    cv2.destroyWindow(
        "Draw Region"
    )

    return np.array(
        drawing_points,
        dtype=np.int32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global original_width
    global original_height

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not os.path.exists(INPUT_VIDEO):

        print()
        print(
            "ERROR: Input video not found:"
        )
        print(INPUT_VIDEO)
        return

    if not os.path.exists(MODEL_PATH):

        print()
        print(
            "ERROR: YOLO model not found:"
        )
        print(MODEL_PATH)
        return

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        INPUT_VIDEO
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open video."
        )
        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    original_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    original_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = total_frames / fps

    print()
    print("=" * 60)
    print("VIDEO INFORMATION")
    print("=" * 60)

    print(
        f"Resolution : "
        f"{original_width} x {original_height}"
    )

    print(
        f"FPS        : {fps:.2f}"
    )

    print(
        f"Frames     : {total_frames}"
    )

    print(
        f"Duration   : {duration:.2f} seconds"
    )

    # ========================================================
    # READ FIRST FRAME
    # ========================================================

    ret, first_frame = cap.read()

    if not ret:

        print(
            "ERROR: Could not read first frame."
        )

        cap.release()
        return

    # ========================================================
    # DRAW REGION
    # ========================================================

    region = draw_region(
        first_frame
    )

    if region is None:

        cap.release()

        print("Cancelled.")
        return

    # ========================================================
    # RESET VIDEO TO FRAME 0
    # ========================================================

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    # ========================================================
    # LOAD YOLO
    # ========================================================

    print()
    print("=" * 60)
    print("LOADING YOLO MODEL")
    print("=" * 60)

    model = YOLO(
        MODEL_PATH
    )

    print(
        "Model loaded."
    )

    # ========================================================
    # OUTPUT VIDEO
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
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
        return

    # ========================================================
    # TRACKING DATA
    # ========================================================

    # Track IDs currently inside the region
    inside_ids = set()

    # Frame when each ID entered
    enter_frame = {}

    # Completed crossing times
    crossing_times = []

    # IDs that already completed a crossing
    completed_ids = set()

    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    frame_number = 0

    print()
    print("=" * 60)
    print("PROCESSING VIDEO")
    print("=" * 60)
    print()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        current_time = (
            frame_number / fps
        )

        # ====================================================
        # YOLO TRACKING
        # ====================================================

        results = model.track(
            frame,
            persist=True,
            classes=[0],       # person
            verbose=False
        )

        annotated = frame.copy()

        # ====================================================
        # DRAW REGION
        # ====================================================

        cv2.polylines(
            annotated,
            [region],
            True,
            (0, 255, 255),
            3
        )

        # ====================================================
        # DETECTION PROCESSING
        # ====================================================

        current_inside_ids = set()

        if (
            results
            and results[0].boxes is not None
            and results[0].boxes.id is not None
        ):

            boxes = results[0].boxes

            ids = (
                boxes.id
                .int()
                .cpu()
                .tolist()
            )

            xyxy = (
                boxes.xyxy
                .cpu()
                .numpy()
            )

            confs = (
                boxes.conf
                .cpu()
                .numpy()
            )

            for track_id, box, confidence in zip(
                ids,
                xyxy,
                confs
            ):

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                # ------------------------------------------------
                # Bottom-center point of person
                # ------------------------------------------------

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    y2
                )

                point = (
                    center_x,
                    center_y
                )

                # ------------------------------------------------
                # Check region
                # ------------------------------------------------

                inside = (
                    cv2.pointPolygonTest(
                        region,
                        point,
                        False
                    ) >= 0
                )

                # ------------------------------------------------
                # Draw bounding box
                # ------------------------------------------------

                if inside:

                    box_color = (
                        0,
                        255,
                        0
                    )

                    current_inside_ids.add(
                        track_id
                    )

                else:

                    box_color = (
                        255,
                        0,
                        0
                    )

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                # ------------------------------------------------
                # Draw center/bottom point
                # ------------------------------------------------

                cv2.circle(
                    annotated,
                    point,
                    5,
                    box_color,
                    -1
                )

                # ------------------------------------------------
                # ID label
                # ------------------------------------------------

                label = (
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    annotated,
                    label,
                    (
                        x1,
                        max(25, y1 - 8)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    box_color,
                    2
                )

                # =================================================
                # PERSON ENTERED REGION
                # =================================================

                if inside:

                    if track_id not in enter_frame:

                        enter_frame[
                            track_id
                        ] = frame_number

                        print(
                            f"ID {track_id} ENTERED "
                            f"region at "
                            f"{current_time:.2f} sec"
                        )

                    inside_ids.add(
                        track_id
                    )

                # =================================================
                # PERSON LEFT REGION
                # =================================================

                else:

                    if (
                        track_id in inside_ids
                        and track_id in enter_frame
                        and track_id not in completed_ids
                    ):

                        start_frame = (
                            enter_frame[
                                track_id
                            ]
                        )

                        crossing_time = (
                            frame_number
                            - start_frame
                        ) / fps

                        crossing_times.append(
                            crossing_time
                        )

                        completed_ids.add(
                            track_id
                        )

                        print(
                            f"ID {track_id} LEFT "
                            f"region at "
                            f"{current_time:.2f} sec | "
                            f"Crossing time: "
                            f"{crossing_time:.2f} sec"
                        )

        # ========================================================
        # CURRENT NUMBER OF PEOPLE
        # ========================================================

        people_currently_inside = len(
            current_inside_ids
        )

        # ========================================================
        # AVERAGE CROSSING TIME
        # ========================================================

        if len(crossing_times) > 0:

            average_crossing_time = (
                sum(crossing_times)
                / len(crossing_times)
            )

        else:

            average_crossing_time = 0.0

        # ========================================================
        # DYNAMIC RED SIGNAL CALCULATION
        # ========================================================

        # Number of pedestrians currently inside
        # the crossing region
        pedestrian_count = (
            people_currently_inside
        )

        # Calculate additional clearance buffer
        pedestrian_buffer = (
            pedestrian_count
            * PEDESTRIAN_BUFFER_PER_PERSON
        )

        # Recommended total red signal time
        #
        # Example:
        # Average crossing = 18 sec
        # Pedestrians = 4
        # Buffer = 4 x 2 = 8 sec
        #
        # Recommended = 18 + 8 = 26 sec
        #
        # Since normal red is 20 sec:
        # Extension = 26 - 20 = 6 sec

        recommended_red_signal = max(
            NORMAL_RED_SIGNAL,
            average_crossing_time
            + pedestrian_buffer
        )

        # Exact extension beyond normal red time
        extended_red_time = max(
            0.0,
            recommended_red_signal
            - NORMAL_RED_SIGNAL
        )

        # ========================================================
        # DISPLAY INFORMATION ON VIDEO
        # ========================================================

        cv2.putText(
            annotated,
            f"People in region: "
            f"{people_currently_inside}",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )

        cv2.putText(
            annotated,
            f"Completed crossings: "
            f"{len(crossing_times)}",
            (25, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"Average crossing time: "
            f"{average_crossing_time:.2f} sec",
            (25, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        # ========================================================
        # NEW: RECOMMENDED RED SIGNAL
        # ========================================================

        cv2.putText(
            annotated,
            f"Recommended Red Signal: "
            f"{recommended_red_signal:.2f} sec",
            (25, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        # ========================================================
        # NEW: EXTENSION TIME
        # ========================================================

        cv2.putText(
            annotated,
            f"Extended Time: "
            f"{extended_red_time:.2f} sec",
            (25, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        # ========================================================
        # SAVE ORIGINAL-RESOLUTION FRAME
        # ========================================================

        out.write(
            annotated
        )

        # ========================================================
        # OPTIONAL LIVE PREVIEW
        # ========================================================

        cv2.imshow(
            "People Crossing Analysis",
            annotated
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print()
            print(
                "Stopped by user."
            )

            break

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    out.release()

    cv2.destroyAllWindows()

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("FINISHED")
    print("=" * 60)

    print(
        f"Unique completed crossings: "
        f"{len(crossing_times)}"
    )

    if crossing_times:

        print()
        print("Crossing times:")

        for i, crossing_time in enumerate(
            crossing_times,
            1
        ):

            print(
                f"  Person {i}: "
                f"{crossing_time:.2f} seconds"
            )

        average = (
            sum(crossing_times)
            / len(crossing_times)
        )

        print()
        print(
            f"Average crossing time: "
            f"{average:.2f} seconds"
        )

        # ====================================================
        # FINAL RED SIGNAL RECOMMENDATION
        # ====================================================

        final_pedestrian_count = (
            len(crossing_times)
        )

        final_pedestrian_buffer = (
            final_pedestrian_count
            * PEDESTRIAN_BUFFER_PER_PERSON
        )

        final_recommended_red = max(
            NORMAL_RED_SIGNAL,
            average
            + final_pedestrian_buffer
        )

        final_extension = max(
            0.0,
            final_recommended_red
            - NORMAL_RED_SIGNAL
        )

        print()
        print("=" * 60)
        print("DYNAMIC SIGNAL RECOMMENDATION")
        print("=" * 60)

        print(
            f"Normal red signal      : "
            f"{NORMAL_RED_SIGNAL:.2f} sec"
        )

        print(
            f"Average crossing time  : "
            f"{average:.2f} sec"
        )

        print(
            f"Pedestrians completed  : "
            f"{final_pedestrian_count}"
        )

        print(
            f"Pedestrian buffer      : "
            f"{final_pedestrian_buffer:.2f} sec"
        )

        print(
            f"Recommended red signal: "
            f"{final_recommended_red:.2f} sec"
        )

        print(
            f"Extended red time     : "
            f"{final_extension:.2f} sec"
        )

        print()

        if final_extension > 0:

            print(
                f"RECOMMENDATION: "
                f"Extend the normal 20-second "
                f"red signal by "
                f"{final_extension:.2f} seconds."
            )

            print(
                f"Total recommended red signal "
                f"time = "
                f"{final_recommended_red:.2f} seconds."
            )

        else:

            print(
                "RECOMMENDATION: "
                "No extension required."
            )

    else:

        print()
        print(
            "No completed crossings detected."
        )

        print()
        print(
            f"Default red signal: "
            f"{NORMAL_RED_SIGNAL:.2f} seconds"
        )

    print()
    print(
        f"Saved to: {OUTPUT_VIDEO}"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()