import time
import Adafruit_PCA9685
from utility_reservoir_computer import *
import cv2

#Load Data
with open('Data_processed/X.pkl', 'rb') as f:
    X = pickle.load(f)

#Output dictionary
dict_y = {'C':0,'D':1,'E':2,'F':3,'G':4,'A':5,'B':6 }

#X Arrray
X_transcribed = note_to_matrix(X, dict_y, 7)                                                                                                                                                                                                  import time

def activate_servos_based_on_list(servo_status_list, duration=1, capture_dir='captured_images/', cap=None):
    """
    Activate servo motors based on the input list and capture images after movement.

    Parameters:
    - servo_status_list: List of servo statuses where True activates the corresponding servo.
    - duration: Duration to keep each activated servo in seconds.
    - capture_dir: Directory to save captured images.
    - cap: Opened OpenCV VideoCapture object.
    """
    # Ensure servo_status_list has at most 8 elements
    servo_status_list = servo_status_list[:8]

    # Initialise the PCA9685 using the default address (0x40).
    pwm = Adafruit_PCA9685.PCA9685()

    # Configure min and max servo pulse lengths
    servo_min = 150  # Min pulse length out of 4096
    servo_max = 450  # Max pulse length out of 4096

    # Set frequency to 60hz, good for servos.
    pwm.set_pwm_freq(60)

    # Activate servo motors based on the input list
    for index, status in enumerate(servo_status_list):
        if status:
            pwm.set_pwm(index, 0, servo_max)
            time.sleep(duration)
            pwm.set_pwm(index, 0, servo_min)

            # Capture image 0.3 seconds after the movement
            time.sleep(0.3)
            capture_image(index, capture_dir, cap)

def capture_image(image_index, capture_dir, cap):
    """
    Capture an image using OpenCV and save it to the specified directory.

    Parameters:
    - image_index: Index to include in the image filename.
    - capture_dir: Directory to save the captured image.
    - cap: Opened OpenCV VideoCapture object.
    """
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        return

    # Save the captured image
    image_filename = f"{capture_dir}image_{image_index}.jpg"
    cv2.imwrite(image_filename, frame)

    print(f"Image captured and saved as '{image_filename}'.")

# Open the camera outside the function
cap = cv2.VideoCapture(0)

music = input("Enter the music index to play: ")
music = X_transcribed[int(music)]

for i in music:
    activate_servos_based_on_list(i, cap=cap)
    activate_servos_based_on_list(i)

# Release the camera
cap.release()
