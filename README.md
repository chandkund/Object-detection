# YOLOv5 Object Detection on Video

This project uses the YOLOv5 model to detect objects in video frames. Detected objects are cropped and saved as individual images. Additionally, all saved images are optionally compressed into a zip file for easier handling.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [License](#license)

## Project Overview

This project performs object detection on video files using the YOLOv5 model. For each frame in the video, objects are detected, cropped, and saved with filenames indicating the frame number and object label. The detected objects can be zipped into a single file for convenience.

## Installation

To run this project, you will need Python along with the following libraries:

- `opencv-python`
- `torch`
- `ultralytics`

Install the required packages using `pip`:

```bash
pip install opencv-python torch ultralytics
```

## Usage
- **Set Up Paths**:

Modify the video_path and save_folder variables in the script to specify the path to your video file and the folder where you want to save the detected objects.

```python
video_path = 'C:\\path\\to\\your\\video.mp4'  # Replace with your actual video path
save_folder = 'C:\\path\\to\\save_folder'  # Replace with your desired save folder
```
- **Run the Script**:
- Save the code in a file, e.g., object_detection.py, and run it using Python:
```bash
python object_detection.py
```
- **Optional - Zip Saved Images**:
  After processing, the detected objects are zipped into a file named detected_objects.zip for easier handling.

## Code Explanation
- **Import Libraries**:
```python
import cv2
import torch
import os
import shutil
```

- **Load YOLOv5 Model**:
```python
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()
```
- **Detect and Save Objects**:
```python
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
        print(f"Saved object {index} as {save_path}")

```
- **Process Video**:
```python
def process_video(video_path, save_folder):
    os.makedirs(save_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detect_and_save_objects(frame, frame_number, save_folder)
        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

```

- **Run and Zip**:
```python
video_path = 'C:\\path\\to\\your\\video.mp4'
save_folder = 'C:\\path\\to\\save_folder'

process_video(video_path, save_folder)

shutil.make_archive('detected_objects', 'zip', save_folder)
print("All detected objects have been zipped into detected_objects.zip")
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
