import os
import zipfile
import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# --- Configuration ---
ZIP_PATH = r"C:\Users\Shailesh Shukla\Desktop\New folder (2)\archive.zip"
EXTRACT_PATH = "gesture_data"
MODEL_PATH = "gesture_model.pkl"

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def extract_landmarks(image):
    """Converts an image into a flat list of 63 landmark coordinates (21 pts * x,y,z)."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        # Get only the first hand detected
        landmarks = results.multi_hand_landmarks[0].landmark
        return [val for lm in landmarks for val in [lm.x, lm.y, lm.z]]
    return None

# --- Step 1: Unzip and Prepare Data ---
if not os.path.exists(EXTRACT_PATH):
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

# --- Step 2: Feature Extraction & Training ---
data, labels = [], []
print("Processing images... this may take a minute.")

# Walk through the extracted folders
for category in os.listdir(EXTRACT_PATH):
    cat_path = os.path.join(EXTRACT_PATH, category)
    if os.path.isdir(cat_path):
        for img_name in os.listdir(cat_path):
            img_path = os.path.join(cat_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                landmarks = extract_landmarks(img)
                if landmarks:
                    data.append(landmarks)
                    labels.append(category)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Save the model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(clf, f)
print(f"Model trained and saved to {MODEL_PATH}!")

# --- Step 3: Real-time Recognition ---
print("Starting Webcam... Press 'q' to quit.")
cap = cv2.VideoCapture(0)
mp_draw = mp.solutions.drawing_utils

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Get landmarks for prediction
            lm_list = [val for lm in hand_lms.landmark for val in [lm.x, lm.y, lm.z]]
            prediction = clf.predict([lm_list])[0]
            
            # Draw UI
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Gesture: {prediction}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()