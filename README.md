
# People Counter using YOLOv8

## Project Overview

This project demonstrates a people counting application using the YOLOv8 model for object detection and tracking. The main goal is to detect individuals in a video and count the number of people entering and exiting a defined area. The project uses various tools and optimizations to improve performance.



## Key Features
- **Object Detection & Counting**: Utilizes YOLOv8 to detect people and assign unique IDs for tracking.
- **Custom Training**: A custom dataset was used for training the model, leveraging Roboflow for data annotation and Google Colab for training.
- **People Counting**: The system tracks the movement of individuals and determines whether they are entering or exiting the monitored area.

## Workflow

### Data Preparation
1. **Upload Dataset**: Use Roboflow to upload and manage the dataset. Frames are extracted from videos and annotated for training.
2. **Export Dataset**: After annotation, the dataset is exported for model training on Google Colab.

### Model Training
The model is trained using the provided code from Roboflow:
```bash
!mkdir {HOME}/datasets
%cd {HOME}/datasets
!pip install roboflow --quiet

from roboflow import Roboflow
rf = Roboflow(api_key= "your API HERE")
project = rf.workspace("YOUR WORKSPACE NAME").project("YOUR PROJECT NAME")
dataset = project.version(2).download("YOUR MODEL NAME")
```

Training command:
```bash
%cd {HOME}
!yolo task=detect mode=train model=model_1.pt data=/content/datasets/PEOPLE-COUNTER-2/data.yaml epochs=25 imgsz=800 plots=True
```

### Model Validation
```bash
!yolo task=detect mode=val model=/content/best.pt data=/content/datasets/PEOPLE-COUNTER-2/data.yaml
```

### People Counting
The counting logic is implemented using functions from the `object_counter` module of the Ultralytics library, adjusted for horizontal video orientation.

### Tracking
YOLOv8’s object tracking feature is used to assign unique IDs to objects and track their movement. Objects are tracked based on confidence scores and bounding boxes.

### Optimization
The model is optimized using Intel's OpenVINO for better performance on Intel GPUs:
1. Convert YOLO model to OpenVINO format (`.xml` and `.bin` files).
2. Run the converted model for faster preprocessing.

## How to Run

1. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the Application**:
    - Use the provided Python code to load the YOLOv8 model, process the video, and count people.

3. Optimization**:
    - To run the model optimized with OpenVINO, follow the instructions provided in the optimization section.

## Demonstration
Run the demonstration video to see the model in action.
