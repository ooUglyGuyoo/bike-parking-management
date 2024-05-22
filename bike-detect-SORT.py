import cv2
import numpy as np
from sort import *
from ultralytics import YOLO

# Initialize SORT tracker
mot_tracker = Sort()

# Load YOLOv8 model
model = YOLO('best.pt')  # Replace with the path to your YOLOv8 model

# Open video file
cap = cv2.VideoCapture('test-bike-detect-trim.mp4')

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writer for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

# Initialize bike detection and tracking
bike_detection_times = {}
detection_time_threshold = 10  # Seconds

# Manual Vanishing Point Selection
vanishing_point = (width // 2, -200)  # Center of the image

while True:
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        break

    # Detect bikes using YOLOv8 model
    results = model.predict(frame, verbose=False)

    # Convert results to numpy array
    dets = results[0].boxes.xyxy.cpu().numpy()

    # Track detected bikes using SORT
    tracked_objects = mot_tracker.update(dets)

    # Draw bounding boxes and update detection times
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj
        bike_id = int(track_id)

        # Check if bike has been detected before
        if bike_id not in bike_detection_times:
            bike_detection_times[bike_id] = 0

        # Set bounding box and text color based on detection time
        if bike_detection_times[bike_id] < detection_time_threshold:
            box_color = (0, 255, 0)  # Green
            text_color = (0, 255, 0)
        else:
            box_color = (0, 0, 255)  # Red
            text_color = (0, 0, 255)
        
        # Draw bounding box and display detection time
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 3)
        cv2.putText(frame, f"Bike {bike_id}: {bike_detection_times[bike_id]:.2f} s", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        # Calculate extension distance (10% of bounding box height)
        extension_distance = (y2 - y1) * 0.1

        # Calculate extended points (Handling negative coordinates)
        top_left_extended_x = int(x1 - extension_distance * ((x1 - vanishing_point[0]) / (y1 - vanishing_point[1])))
        top_left_extended_y = int(y1 - extension_distance)
        bottom_left_extended_x = int(x1 - extension_distance * ((x1 - vanishing_point[0]) / (y2 - vanishing_point[1])))
        bottom_left_extended_y = int(y2 - extension_distance)
        top_right_extended_x = int(x2 - extension_distance * ((x2 - vanishing_point[0]) / (y1 - vanishing_point[1])))
        top_right_extended_y = int(y1 - extension_distance)
        bottom_right_extended_x = int(x2 - extension_distance * ((x2 - vanishing_point[0]) / (y2 - vanishing_point[1])))
        bottom_right_extended_y = int(y2 - extension_distance)

        # Ensure extended points stay within the frame
        top_left_extended_x = max(0, top_left_extended_x)
        top_left_extended_y = max(0, top_left_extended_y)
        bottom_left_extended_x = max(0, bottom_left_extended_x)
        bottom_left_extended_y = max(0, bottom_left_extended_y)
        top_right_extended_x = min(width, top_right_extended_x)
        top_right_extended_y = max(0, top_right_extended_y)
        bottom_right_extended_x = min(width, bottom_right_extended_x)
        bottom_right_extended_y = max(0, bottom_right_extended_y)

        # Draw the lines connecting the bounding box to the vanishing point
        cv2.line(frame, (int(x1), int(y1)), (top_left_extended_x, top_left_extended_y), box_color, 2)
        cv2.line(frame, (int(x2), int(y1)), (top_right_extended_x, top_right_extended_y), box_color, 2)
        cv2.line(frame, (int(x1), int(y2)), (bottom_left_extended_x, bottom_left_extended_y), box_color, 2)
        cv2.line(frame, (int(x2), int(y2)), (bottom_right_extended_x, bottom_right_extended_y), box_color, 2)

        # Draw the back bounding box
        cv2.line(frame, (top_right_extended_x, top_right_extended_y), (top_left_extended_x, top_left_extended_y), box_color, 1)
        cv2.line(frame, (bottom_right_extended_x, bottom_right_extended_y), (top_right_extended_x, top_right_extended_y), box_color, 1)
        cv2.line(frame, (top_left_extended_x, top_left_extended_y), (bottom_left_extended_x, bottom_left_extended_y), box_color, 1)
        cv2.line(frame, (bottom_left_extended_x, bottom_left_extended_y), (bottom_right_extended_x, bottom_right_extended_y), box_color, 1)
        
        # Calculate vanishing point lines (adjust thickness as needed)
        # top_left_line = cv2.line(frame, (int(x1), int(y1)), vanishing_point, box_color, 1)
        # bottom_left_line = cv2.line(frame, (int(x1), int(y2)), vanishing_point, box_color, 1)
        # top_right_line = cv2.line(frame, (int(x2), int(y1)), vanishing_point, box_color, 1)
        # bottom_right_line = cv2.line(frame, (int(x2), int(y2)), vanishing_point, box_color, 1)
                
        # Update detection time
        bike_detection_times[bike_id] += 1 / fps

    # Write the frame to the output video
    out_video.write(frame)

    # Display the resulting frame (optional)
    cv2.imshow('Bike Detection and Tracking', frame)

    # Check for user input to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture, video writer, and close all windows
cap.release()
out_video.release()
cv2.destroyAllWindows()