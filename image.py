import cv2
import mediapipe as mp

# Initialize Mediapipe pose detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load test image
img = cv2.imread('test/image/img.jpg')

# Convert image to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect pose in image
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    results = pose.process(img_rgb)
    
    # Draw pose landmarks on image
    annotated_image = img.copy()
    mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Classify pose as standing or sitting
    if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y > results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y:
        pose_class = 'sitting'
    else:
        pose_class = 'standing'

# Show image with pose landmarks and pose classification
#cv2.imshow('Pose Estimation', annotated_image)
cv2.putText(annotated_image, pose_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow('Pose Classification', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
