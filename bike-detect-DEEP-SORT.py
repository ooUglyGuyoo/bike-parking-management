import cv2
import time
import numpy as np
from deep_sort import tracker as Tracker  # Import Tracker from deep_sort
from ultralytics import YOLO

# Load your existing YOLOv8 model
model = YOLO('best.pt')  # Replace with the path to your .pt file

# Initialize DeepSORT
deepsort = Tracker.Tracker(metric=0.5, max_iou_distance=0.7, max_age=30, n_init=3)

# Initialize video capture
cap = cv2.VideoCapture('test-bike-detect-trim.mp4')  # Replace with your video path

# Initialize variables for time tracking
parking_times = {}
last_seen_frames = {}
frame_count = 0

# Function to draw bounding boxes and display time
def draw_boxes(frame, detections, parking_times, last_seen_frames):
    for track in detections:
        bbox = track.to_tlbr()  # Convert to top-left, bottom-right format
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        track_id = track.track_id

        # Check if the bike is new or has been seen before
        if track_id not in parking_times:
            parking_times[track_id] = 0
            last_seen_frames[track_id] = frame_count

        # Update parking time if the bike is still in the frame
        if frame_count - last_seen_frames[track_id] <= 30:  # Adjust the threshold as needed
            parking_times[track_id] += 1
            last_seen_frames[track_id] = frame_count
        else:
            # Reset parking time if the bike is not seen for a while
            parking_times[track_id] = 0

        # Draw bounding box and display time
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'Time: {parking_times[track_id]}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Perform object detection using your YOLOv8 model
    results = model(frame)

    # Extract bounding boxes and confidence scores
    detections = results[0].xyxy

    # Convert detections to DeepSORT format
    detections = np.array(detections)
    detections = detections[:, :4].astype(np.int32)
    detections = detections.tolist()

    # Perform DeepSORT tracking
    tracked_detections = deepsort.update(detections, frame)

    # Draw bounding boxes and display time
    draw_boxes(frame, tracked_detections, parking_times, last_seen_frames)

    # Display the frame
    cv2.imshow('Parking Lot', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()