from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2


# Initialize YOLO model
model = YOLO("besto_openvino_model/")


# Set up video capture
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize Object Counter with classes_names
line_pedestrian_1 = [(0, 190), (800, 190)]  # Define as per your need
counter = object_counter.ObjectCounter(
    count_reg_color=(0, 232, 255),
    view_img=True,
    reg_pts=line_pedestrian_1,
    classes_names=model.names,  # Pass classes_names to the constructor
)

# Set up video writer
output_path = "RESULT.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, im0 = cap.read()
    if not ret:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break

    # Track objects
    results = model.track(im0, persist=True, show=False)
    
    # Start counting based on tracks and your line
    im0 = counter.start_counting(im0, results)
    
    out.write(im0)
    
    if cv2.waitKey(1) == ord("q"):  # Press q to quit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
