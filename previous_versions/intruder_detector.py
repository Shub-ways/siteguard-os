import cv2
import time
from datetime import datetime
import os

class IntruderDetector:
    def __init__(self, video_source=0, min_area=5000):
        # We now pass the path to the video file instead of '0'
        self.video = cv2.VideoCapture(video_source)
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        self.min_area = min_area 
        
        self.last_log_time = 0
        self.log_cooldown = 3
        self.log_dir = "logs"
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        print(f"[INFO] Loading media from: {video_source}")
        print(f"[INFO] System Initialized with Area Threshold: {self.min_area}")
        print("[INFO] Press 'q' to quit.")

    def process_frame(self, frame):
        fg_mask = self.back_sub.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, fg_mask

    def run(self):
        prev_time = 0

        while True:
            ret, frame = self.video.read()
            
            # --- CHANGE 1: Video Looping Logic ---
            if not ret:
                print("[INFO] Video ended. Rewinding for presentation loop...")
                # Reset the video position to the first frame
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue # Skip the rest of the loop and grab the first frame again
                
            frame = cv2.resize(frame, (640, 480))
            clean_frame = frame.copy() 
            
            contours, fg_mask = self.process_frame(frame)
            
            status = "System Armed"
            status_color = (0, 255, 0)
            intruder_detected = False

            for contour in contours:
                if cv2.contourArea(contour) < self.min_area:
                    continue 
                
                status = "ALERT: INTRUDER"
                status_color = (0, 0, 255)
                intruder_detected = True
                
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), status_color, 3)
                
                center_x = x + (w // 2)
                center_y = y + (h // 2)
                cv2.drawMarker(frame, (center_x, center_y), status_color, cv2.MARKER_CROSS, 20, 2)

            current_time = time.time()
            if intruder_detected:
                if (current_time - self.last_log_time) > self.log_cooldown:
                    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filepath = os.path.join(self.log_dir, f"intruder_{timestamp_str}.jpg")
                    cv2.imwrite(filepath, clean_frame)
                    print(f"[ALERT] Intruder Logged: {filepath}")
                    self.last_log_time = current_time

            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time
            timestamp = datetime.now().strftime("%A %d %B %Y %I:%M:%S %p")

            cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Security Feed", frame)
            cv2.imshow("Foreground Mask", fg_mask)

            # We use waitKey(30) to simulate normal video playback speed (~30 FPS)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- CHANGE 2: Feed the video file into the detector ---
    # Make sure the filename matches exactly what you saved!
    video_file_path = "demo_media/demo_video.mp4" 
    
    detector = IntruderDetector(video_source=video_file_path, min_area=5000) 
    detector.run()