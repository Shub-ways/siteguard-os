import cv2
import numpy as np
import face_recognition
import os
import winsound
import csv
import time
import threading
import smtplib
from email.message import EmailMessage
from datetime import datetime
from PIL import Image 

# --- NEW: Phase 8 Helper Function for Math ---
def get_ear(eye_points):
    """Calculates the Eye Aspect Ratio (EAR) using Euclidean distance."""
    p = np.array(eye_points)
    # Vertical eye landmarks
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    # Horizontal eye landmarks
    C = np.linalg.norm(p[0] - p[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear
# ---------------------------------------------

class ConstructionSiteSecurity:
    def __init__(self, video_source=0, workers_dir="registered_workers"):
        self.video = cv2.VideoCapture(video_source)
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Email Configuration
        self.sender_email = " "
        self.sender_password = " "
        self.receiver_email = " "

        print(f"[INFO] Loading registered workers from '{workers_dir}'...")
        if not os.path.exists(workers_dir):
            os.makedirs(workers_dir)
            print(f"[WARNING] Folder '{workers_dir}' created. Please add worker photos and restart.")
            exit()
            
        for filename in os.listdir(workers_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(workers_dir, filename)
                try:
                    pil_image = Image.open(image_path).convert("RGB")
                    worker_image = np.array(pil_image)
                    encodings = face_recognition.face_encodings(worker_image)
                    if len(encodings) > 0:
                        self.known_face_encodings.append(encodings[0])
                        clean_name = os.path.splitext(filename)[0].replace("_", " ")
                        self.known_face_names.append(clean_name)
                        print(f"       -> Registered: {clean_name}")
                except Exception as e:
                    print(f"       -> [ERROR] Could not process {filename}. Reason: {e}")

        self.restricted_zone = np.array([
            [150, 100],  [500, 100],  [550, 400],  [100, 400]
        ], np.int32)
        
        self.log_dir = "logs"
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)
        
        self.last_log_time = 0
        self.log_cooldown = 10 
        self.last_email_time = 0
        self.email_cooldown = 60 
        
        self.attendance_file = "attendance_log.csv"
        self.logged_workers = {} 
        self.cooldown_seconds = 60 
        
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Date", "Time_of_Entry"])

        # --- NEW: Phase 8 Liveness Tracking Variables ---
        self.EAR_THRESHOLD = 0.22  # If ratio drops below this, eyes are closed
        self.EAR_CONSEC_FRAMES = 1 # How many frames eyes must be below threshold
        
        self.blink_counters = {}   # Tracks consecutive frames eyes are closed
        self.liveness_verified = {} # Tracks who has successfully blinked
        # ------------------------------------------------

        print("[INFO] System Active. Press 'q' to quit.")

    def send_alert_email(self, image_path, time_str):
        print("[INFO] Sending alert email in the background...")
        try:
            msg = EmailMessage()
            msg['Subject'] = f"URGENT: Security Breach Detected at {time_str}"
            msg['From'] = self.sender_email
            msg['To'] = self.receiver_email
            msg.set_content("An unknown individual has entered the restricted construction zone. See attached image.")
            with open(image_path, 'rb') as f:
                img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(self.sender_email, self.sender_password)
                smtp.send_message(msg)
            print(f"[SUCCESS] Email alert sent to {self.receiver_email}!")
        except Exception as e:
            print(f"[ERROR] Failed to send email: {e}")

    def run(self):
        process_this_frame = True

        while True:
            ret, frame = self.video.read()
            if not ret:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            frame = cv2.resize(frame, (640, 480))
            clean_frame = frame.copy() 
            
            cv2.polylines(frame, [self.restricted_zone], isClosed=True, color=(255, 165, 0), thickness=2) 
            cv2.putText(frame, "RESTRICTED ZONE", (160, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

            if process_this_frame:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                # --- NEW: Phase 8 Extract Facial Landmarks for the eyes ---
                face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)

            process_this_frame = not process_this_frame

            # Track who is in the current frame to reset liveness if they walk away
            current_frame_names = []

            for (top, right, bottom, left), face_encoding, face_landmarks in zip(face_locations, face_encodings, face_landmarks_list):
                face_center_x = left + ((right - left) // 2)
                face_center_y = top + ((bottom - top) // 2)
                
                is_inside_zone = cv2.pointPolygonTest(self.restricted_zone, (face_center_x, face_center_y), False) >= 0

                if is_inside_zone:
                    name = "UNKNOWN INTRUDER"
                    status_color = (0, 0, 255) 
                    alarm_triggered = True

                    if len(self.known_face_encodings) > 0:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
                        
                        if True in matches:
                            first_match_index = matches.index(True)
                            clean_worker_name = self.known_face_names[first_match_index]
                            current_frame_names.append(clean_worker_name)
                            
                            # --- NEW: Phase 8 Liveness Logic ---
                            left_eye = face_landmarks['left_eye']
                            right_eye = face_landmarks['right_eye']
                            
                            ear = (get_ear(left_eye) + get_ear(right_eye)) / 2.0
                            
                            # Check if eyes are closed
                            if ear < self.EAR_THRESHOLD:
                                self.blink_counters[clean_worker_name] = self.blink_counters.get(clean_worker_name, 0) + 1
                            else:
                                # If eyes open back up after being closed, it's a blink!
                                if self.blink_counters.get(clean_worker_name, 0) >= self.EAR_CONSEC_FRAMES:
                                    self.liveness_verified[clean_worker_name] = True
                                self.blink_counters[clean_worker_name] = 0

                            # Determine Access Status based on Liveness
                            if self.liveness_verified.get(clean_worker_name, False):
                                name = f"Authorized: {clean_worker_name}"
                                status_color = (0, 255, 0) # Green
                                alarm_triggered = False 

                                # Only log attendance AFTER liveness is verified
                                current_time = time.time()
                                if clean_worker_name not in self.logged_workers or (current_time - self.logged_workers[clean_worker_name]) > self.cooldown_seconds:
                                    now = datetime.now()
                                    with open(self.attendance_file, mode='a', newline='') as file:
                                        csv.writer(file).writerow([clean_worker_name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
                                    print(f"[SUCCESS] Attendance Logged: {clean_worker_name}")
                                    self.logged_workers[clean_worker_name] = current_time
                            else:
                                name = f"AWAITING BLINK: {clean_worker_name}"
                                status_color = (0, 255, 255) # Yellow
                                alarm_triggered = False # Don't alarm on our own workers, just wait for them to blink
                            # -----------------------------------

                    cv2.rectangle(frame, (left, top), (right, bottom), status_color, 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                    current_time = time.time()
                    if alarm_triggered:
                        cv2.putText(frame, "SECURITY BREACH!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                        winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC)
                        
                        if (current_time - self.last_log_time) > self.log_cooldown:
                            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            filepath = os.path.join(self.log_dir, f"intruder_{timestamp_str}.jpg")
                            cv2.imwrite(filepath, clean_frame) 
                            self.last_log_time = current_time
                            
                            if (current_time - self.last_email_time) > self.email_cooldown:
                                email_thread = threading.Thread(target=self.send_alert_email, args=(filepath, timestamp_str))
                                email_thread.start()
                                self.last_email_time = current_time

            # --- NEW: Reset liveness if the person leaves the frame ---
            keys_to_delete = [name for name in self.liveness_verified.keys() if name not in current_frame_names]
            for name in keys_to_delete:
                del self.liveness_verified[name]
                if name in self.blink_counters:
                    del self.blink_counters[name]
            # ----------------------------------------------------------

            cv2.imshow("Construction Site Monitor", frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ConstructionSiteSecurity(video_source=0) 
    detector.run()