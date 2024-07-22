import cv2
import torch
import os
import shutil

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

def detect_and_save_objects(frame, frame_number, save_folder):
    results = model(frame)
    objects = results.pandas().xyxy[0]

    for index, obj in objects.iterrows():
        xmin, ymin, xmax, ymax = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
        label = obj['name']

        # Crop the detected object
        cropped_img = frame[ymin:ymax, xmin:xmax]

        # Save the cropped image
        save_path = os.path.join(save_folder, f"frame_{frame_number}_object_{index}_{label}.jpg")
        cv2.imwrite(save_path, cropped_img)
        print(f"Saved object {index} as {save_path}")  # Print the saved file path

def process_video(video_path, save_folder):
    # Create the save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Handle potential errors opening the video file
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return
    except Exception as e:
        print(f"Error opening video: {e}")
        return

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects and save cropped images
        detect_and_save_objects(frame, frame_number, save_folder)

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

# Specify the path to your video file manually
video_path = 'C:\\save_folder\\video1.mp4'  # Replace with your actual video path

# Specify the folder where you want to save detected objects
save_folder = 'C:\\save_folder'  # Detected objects will be saved in this folder

# Process the video
process_video(video_path, save_folder)

# Optionally, zip the detected objects folder for easier handling
shutil.make_archive('detected_objects', 'zip', save_folder)
print("All detected objects have been zipped into detected_objects.zip")
