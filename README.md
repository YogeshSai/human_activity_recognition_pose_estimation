# Human Activity Recognition using Deep Learning and Pose Estimation


This Python code demonstrates human activity recognition using a pre-trained model and MediaPipe for pose estimation. It reads a video input or a live web-cam feed, captures frames, and predicts the activity in the captured frames using a deep neural network.

Requirements: 

To run this code, you need to have the following packages installed:

- collections
- numpy
- opencv-python
- mediapipe
- onnx

Download the model and place it in the model Directory

https://drive.google.com/file/d/1QPXJB6AHBFH957Q_CSCFkxkoFAYTIq1J/view?usp=share_link


Running the Code

To run the code, execute the following command in your terminal:

- python live.py

This will start capturing frames from your default webcam. If you want to use a video file instead of the webcam


- python video.py

Press the q key to exit the application.



Code Overview

The Parameters class initializes important paths and constants for the code.
- A double-ended queue named captures is created to store the captured frames.
- The pre-trained human activity recognition model is loaded using OpenCV's cv2.dnn.readNet function.
- The MediaPipe pose estimation is set up.
- The captured frames are processed and resized and are added to the deque.
- The code predicts the activity in the captured frames using the pre-trained model.
- The MediaPipe is used to estimate the pose in the captured frames.
- The predicted activity and pose are drawn on the captured frames.
- The captured frames are displayed on the screen.


Acknowledgments


This code is based on the tutorial by Adrian Rosebrock on PyImageSearch Human Activity Recognition with OpenCV and Deep Learning.
https://pyimagesearch.com/2019/11/25/human-activity-recognition-with-opencv-and-deep-learning/
