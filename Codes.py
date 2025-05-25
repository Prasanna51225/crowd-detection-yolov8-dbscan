import cv2
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from google.colab import files, drive
from IPython.display import HTML, display
from base64 import b64encode
import base64
import shutil

# Mount Google Drive
drive.mount('/content/drive')

class ReliableCrowdDetector:
    def __init__(self):
        # Initialize YOLOv8 model
        self.model = YOLO('yolov8s.pt')  # More accurate than 'nano' version

        # Detection parameters (optimized for crowd detection)
        self.conf_threshold = 0.6  
        self.iou_threshold = 0.45   

        # Crowd parameters (strictly following requirements)
        self.proximity_thresh = 110
        self.min_crowd_size = 3
        self.persistence_frames = 10 # Must persist for 10 frames

        # Background subtractor for static crowd detection
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

        # Tracking variables
        self.active_crowds = defaultdict(dict)
        self.logged_events = []
        self.crowd_snapshots = {}  # Stores one snapshot per crowd signature
        self.snapshot_counter = 1   # Counter for naming crowd snapshots

        # Visualization settings
        self.colors = [(0,255,0), (0,0,255), (255,0,0)]  # Different colors for different crowds
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def detect_people(self, frame):
        """Detect people using YOLOv8 with optimized parameters"""
        results = self.model(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[0],  # Only detect people
            imgsz=640     # Higher resolution for better detection
        )

        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf)
                if conf > self.conf_threshold:
                    boxes.append((x1, y1, x2, y2))

        return boxes

    def detect_static_people(self, frame):
        """Detect static people using background subtraction"""
        fgmask = self.fgbg.apply(frame)
        fgmask = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)[1]

        # Morphological operations to clean up detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:  # Only consider larger areas
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append((x, y, x+w, y+h))

        return boxes

    def cluster_people(self, boxes, frame_height):
        """Improved clustering with adaptive proximity threshold"""
        if len(boxes) < self.min_crowd_size:
            return []

        # Calculate centers of bounding boxes
        centers = np.array([[(x1+x2)//2, (y1+y2)//2] for (x1, y1, x2, y2) in boxes])

        # Adaptive proximity based on frame height
        adaptive_thresh = max(100, min(200, int(frame_height * 0.2)))

        # DBSCAN clustering with adaptive parameters
        clustering = DBSCAN(
            eps=adaptive_thresh,
            min_samples=self.min_crowd_size
        ).fit(centers)

        # Group people by cluster
        crowds = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label != -1:  # Ignore noise points
                crowds[label].append(idx)

        return list(crowds.values())

    def save_crowd_snapshot(self, frame, crowd_boxes, output_dir, crowd_signature):
        """Save exactly one snapshot per unique crowd"""
        # Return existing snapshot if this crowd was already captured
        if crowd_signature in self.crowd_snapshots:
            return self.crowd_snapshots[crowd_signature]

        # Create snapshot directory if it doesn't exist
        snapshot_dir = os.path.join(output_dir, 'crowd_snapshots')
        os.makedirs(snapshot_dir, exist_ok=True)

        # Get crowd bounding box with some padding
        x1 = min(box[0] for box in crowd_boxes) - 20
        y1 = min(box[1] for box in crowd_boxes) - 20
        x2 = max(box[2] for box in crowd_boxes) + 20
        y2 = max(box[3] for box in crowd_boxes) + 20

        # Ensure coordinates are within frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        # Crop the crowd region
        crowd_region = frame[y1:y2, x1:x2]

        # Generate unique filename
        snapshot_name = f"crowd_{len(self.crowd_snapshots)+1}.jpg"

        # Save the snapshot
        snapshot_path = os.path.join(snapshot_dir, snapshot_name)
        cv2.imwrite(snapshot_path, crowd_region)

        # Store the mapping
        self.crowd_snapshots[crowd_signature] = snapshot_name

        return snapshot_name

    def track_crowds(self, frame_num, crowds, boxes, frame, output_dir):
        """Track crowd persistence across frames and save snapshots"""
        current_crowds = set()

        for crowd in crowds:
            # Create crowd signature based on relative positions
            center_x = sum((boxes[i][0] + boxes[i][2])/2 for i in crowd)/len(crowd)
            center_y = sum((boxes[i][1] + boxes[i][3])/2 for i in crowd)/len(crowd)
            crowd_signature = (round(center_x/50)*50, round(center_y/50)*50, len(crowd))
            current_crowds.add(crowd_signature)

            # Update or initialize crowd tracking
            if crowd_signature in self.active_crowds:
                self.active_crowds[crowd_signature]['count'] += 1
                self.active_crowds[crowd_signature]['last_frame'] = frame_num
            else:
                # Save snapshot only when crowd is first detected
                snapshot_name = self.save_crowd_snapshot(
                    frame,
                    [boxes[i] for i in crowd],
                    output_dir,
                    crowd_signature
                )

                self.active_crowds[crowd_signature] = {
                    'start_frame': frame_num,
                    'last_frame': frame_num,
                    'count': 1,
                    'person_count': len(crowd),
                    'snapshot': snapshot_name
                }

        # Check for expired crowds
        expired = []
        for crowd_sig in list(self.active_crowds.keys()):
            if crowd_sig not in current_crowds:
                crowd_data = self.active_crowds[crowd_sig]

                # Only log if crowd persisted for required frames
                if crowd_data['count'] >= self.persistence_frames:
                    self.logged_events.append({
                        'frame_number': crowd_data['start_frame'],
                        'person_count': crowd_data['person_count'],
                        'end_frame': crowd_data['last_frame'],
                        'duration_frames': crowd_data['count'],
                        'snapshot': crowd_data['snapshot']
                    })
                expired.append(crowd_sig)

        # Remove expired crowds
        for crowd_sig in expired:
            del self.active_crowds[crowd_sig]

    def visualize_results(self, frame, frame_num, boxes, crowds):
        """Clear visualization of detection results"""
        # Draw all detected people
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Draw and annotate crowds
        for i, crowd in enumerate(crowds):
            if len(crowd) >= self.min_crowd_size:
                crowd_boxes = [boxes[idx] for idx in crowd]
                x1 = min(box[0] for box in crowd_boxes)
                y1 = min(box[1] for box in crowd_boxes)
                x2 = max(box[2] for box in crowd_boxes)
                y2 = max(box[3] for box in crowd_boxes)

                # Use different color for each crowd
                color = self.colors[i % len(self.colors)]

                # Draw thick bounding box around crowd
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Draw crowd count
                cv2.putText(frame, f'Crowd: {len(crowd)}', (x1, y1 - 10),
                           self.font, 0.8, color, 2)

                # Draw semi-transparent overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

        # Add frame information
        info_y = 40
        cv2.putText(frame, f'Frame: {frame_num}', (20, info_y),
                   self.font, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f'People: {len(boxes)}', (20, info_y + 40),
                   self.font, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f'Crowds: {len(crowds)}', (20, info_y + 80),
                   self.font, 0.8, (255, 255, 255), 2)

        return frame

    def cleanup_snapshots(self, output_dir):
        """Remove any snapshots not referenced in the final CSV"""
        snapshot_dir = os.path.join(output_dir, 'crowd_snapshots')
        if not os.path.exists(snapshot_dir):
            return

        # Get all snapshots referenced in CSV
        referenced_snapshots = {event['snapshot'] for event in self.logged_events}

        # Delete any snapshots not in the final CSV
        for filename in os.listdir(snapshot_dir):
            if filename not in referenced_snapshots:
                os.remove(os.path.join(snapshot_dir, filename))

    def process_video(self, input_path, output_dir='/content/drive/MyDrive/CrowdDetection'):
        """Main processing function with reliable crowd detection"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Prepare output paths
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f'{base_name}_output.mp4')
        csv_path = os.path.join(output_dir, f'{base_name}_crowds.csv')

        # Open video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Process frames with progress bar
        progress_bar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect people using both methods
            yolo_boxes = self.detect_people(frame)
            static_boxes = self.detect_static_people(frame)

            # Combine detections (prioritize YOLO boxes)
            all_boxes = yolo_boxes + static_boxes

            # Cluster people into crowds
            crowds = self.cluster_people(all_boxes, frame_height)

            # Track crowd persistence and save snapshots
            self.track_crowds(frame_num, crowds, all_boxes, frame.copy(), output_dir)

            # Visualize results
            frame = self.visualize_results(frame, frame_num, all_boxes, crowds)
            out.write(frame)

            frame_num += 1
            progress_bar.update(1)

        progress_bar.close()
        cap.release()
        out.release()

        # Clean up any snapshots not in final CSV
        self.cleanup_snapshots(output_dir)

        # Save results to CSV
        if self.logged_events:
            df = pd.DataFrame(self.logged_events, columns=[
                'frame_number',
                'person_count',
                'end_frame',
                'duration_frames',
                'snapshot'
            ])

            # Add snapshot paths
            df['snapshot'] = df['snapshot'].apply(
                lambda x: f'=HYPERLINK("crowd_snapshots/{x}", "{x}")'
            )

            df.to_csv(csv_path, index=False)
            print(f"\n Crowd data saved to: {csv_path}")

            # Create a zip file with all snapshots for download
            snapshot_dir = os.path.join(output_dir, 'crowd_snapshots')
            if os.path.exists(snapshot_dir):
                shutil.make_archive(os.path.join(output_dir, 'crowd_snapshots'), 'zip', snapshot_dir)
                print(f" Crowd snapshots saved to: {os.path.join(output_dir, 'crowd_snapshots.zip')}")
        else:
            print("\n No crowds detected meeting the criteria (3+ people for 10+ frames)")
            # Create empty CSV with correct columns
            pd.DataFrame(columns=[
                'frame_number',
                'person_count',
                'end_frame',
                'duration_frames',
                'snapshot'
            ]).to_csv(csv_path)

        print(f" Output video saved to: {output_path}")
        return output_path, csv_path

def upload_video():
    """Handle video upload in Colab"""
    print("Please upload your video file (MP4/AVI/MOV):")
    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded!")
        return None

    for filename in uploaded.keys():
        temp_path = f"/content/{filename}"
        with open(temp_path, 'wb') as f:
            f.write(uploaded[filename])
        print(f"Uploaded: {filename}")
        return temp_path
    return None

def show_video(video_path):
    """Display video in notebook"""
    try:
        mp4 = open(video_path, 'rb').read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        return HTML(f"""
        <div style="display:flex; justify-content:center;">
            <video width="80%" controls>
                <source src="{data_url}" type="video/mp4">
            </video>
        </div>
        """)
    except Exception as e:
        print(f" Error displaying video: {e}")
        return HTML("<p>Video display error</p>")

def main():
    print("""
     Reliable Crowd Detection System
    ---------------------------------
    Detects groups of 3+ people close together
    for 10+ consecutive frames (both moving and static)
    """)

    # Install requirements
    print(" Installing required packages...")
    !pip install ultralytics opencv-python numpy pandas scikit-learn tqdm --quiet

    # Initialize detector
    detector = ReliableCrowdDetector()

    # Upload video
    input_path = upload_video()
    if not input_path:
        return

    # Process video
    try:
        print("\n Analyzing video (this may take several minutes)...")
        output_path, csv_path = detector.process_video(input_path)

        # Show results
        print("\n Processing complete! Results:")
        display(show_video(output_path))

        # Show CSV preview
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if not df.empty:
                print("\n Crowd Events Detected:")
                display(df)

                # Display sample snapshots if available
                snapshot_dir = os.path.join(os.path.dirname(csv_path), 'crowd_snapshots')
                if os.path.exists(snapshot_dir):
                    sample_snapshots = [f for f in os.listdir(snapshot_dir) if f.endswith('.jpg')][:3]
                    if sample_snapshots:
                        print("\Sample Crowd Snapshots:")
                        for snapshot in sample_snapshots:
                            snapshot_path = os.path.join(snapshot_dir, snapshot)
                            display(HTML(f"""
                            <div style="margin:10px; display:inline-block;">
                                <img src="{snapshot_path}" width="300">
                                <p style="text-align:center;">{snapshot}</p>
                            </div>
                            """))
            else:
                print("\n No crowds detected meeting the criteria")
        print("\n Output saved to Google Drive")

    except Exception as e:
        print(f" Error during processing: {e}")

if __name__ == "__main__":
    main()
