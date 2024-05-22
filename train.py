# Import the necessary libraries
from ultralytics import YOLO

if __name__ ==  '__main__':
    # Set the path to your dataset
    dataset_path = r'C:\Users\ylian\Documents\Github\bike-parking-management\bike-detect-roboflow\data.yaml'

    # Create a YOLOv8 model
    model = YOLO("yolov8n.yaml") # build a new model from YAML
    model = YOLO("yolov8n.pt")  # Replace 'yolov8n.pt' with the desired model size (e.g., 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt')
    model = YOLO("yolov8n.yaml").load("yolov8n.pt") # build from YAML and transfer weights

    # Train the model
    print("=====================START TRAINING=====================")
    results = model.train(
        data=dataset_path,
        epochs=100,  # Number of training epochs
        batch=16,  # Batch size
        imgsz=640,  # Input image size
        device=0,  # Device to use for training (0 for GPU, -1 for CPU)
    )

    print("=====================START VALIDATION=====================")
    # Evaluate the model's performance on the validation set
    results = model.val()
    print(results)

    # print("=====================Perform on image=====================")
    # Perform object detection on an image using the model
    # results = model.predict('test.jpg')

    # Export the model to ONNX format
    success = model.export(format="onnx")