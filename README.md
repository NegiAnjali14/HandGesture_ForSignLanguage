📌 Project Description

Hand Gesture Sign Language Recognition is a computer vision project that detects and recognizes hand gestures using a webcam. The system uses Python, OpenCV, MediaPipe, and Machine Learning to identify hand landmarks and classify gestures in real time.

This project helps demonstrate how computer vision can assist in communication with people who use sign language.

🎯 Objectives
To detect hand gestures using a webcam.
To classify different hand signs such as Hello, Yes, No, Thank You, etc.
To build a real-time sign language recognition system.
To apply computer vision and machine learning techniques.

🛠 Technologies Used
Python
OpenCV
MediaPipe
TensorFlow / Keras
NumPy

📂 Project Structure
Hand-Gesture-Sign-Language

Data ├── Hello ├── I Love You ├── No ├── Ok ├── Please ├── Thank You └── Yes

model datacollection.py test.py README.md

⚙️ System Requirements
Python 3.8 or above
Webcam
Windows / Linux / macOS
Install required libraries:

pip install opencv-python pip install mediapipe pip install tensorflow pip install numpy

📸 Dataset Collection
The dataset for training the model is collected using the datacollection.py script. This script captures images from the webcam and stores them inside the Data folder according to gesture categories.

Example gesture classes:

Hello
I Love You
No
Ok
Please
Thank You
Yes

▶️ Running the Project
Step 1: Clone the Repository
git clone https://github.com/yourusername/hand-gesture-sign-language.git

Step 2: Open the Project Folder
cd hand-gesture-sign-language

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Run Gesture Detection
python test.py

The webcam will start and the system will detect hand gestures in real time.

📊 Features
Real-time hand detection
Gesture classification
Webcam-based interaction
Simple and lightweight implementation

🔮 Future Improvements
Support for more sign language gestures
Real-time text or speech output
Mobile application version
Higher accuracy using deep learning models

👩‍💻 Author
Anjali Negi

BCA Major Project
