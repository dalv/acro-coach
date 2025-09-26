import cv2
import numpy as np
from ultralytics import YOLO
import os

class AcrobaticPoseProcessor:
    def __init__(self, model_size='m'):
        """
        Initialize the pose processor with YOLOv11
        model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
        Smaller models are faster, larger models are more accurate
        """
        self.model = YOLO(f'yolo11{model_size}-pose.pt')
        
        # COCO pose keypoints (17 keypoints)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Define skeleton connections for drawing
        self.skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9),          # Left arm
            (6, 8), (8, 10),                 # Right arm  
            (5, 11), (6, 12), (11, 12),      # Torso
            (11, 13), (13, 15),              # Left leg
            (12, 14), (14, 16)               # Right leg
        ]
        
        # Colors for different people (BGR format)
        self.person_colors = [
            (0, 255, 0),    # Green for person 1
            (255, 0, 0),    # Blue for person 2
            (0, 0, 255),    # Red for person 3
            (255, 255, 0),  # Cyan for person 4
            (255, 0, 255),  # Magenta for person 5
        ]
        
    def draw_keypoints_and_skeleton(self, frame, keypoints, person_id=0, confidence_threshold=0.3):
        """Draw keypoints and skeleton on the frame"""
        color = self.person_colors[person_id % len(self.person_colors)]
        
        # Draw skeleton connections
        for connection in self.skeleton:
            kpt1_idx, kpt2_idx = connection
            if (kpt1_idx < len(keypoints) and kpt2_idx < len(keypoints) and 
                len(keypoints[kpt1_idx]) >= 3 and len(keypoints[kpt2_idx]) >= 3):
                
                x1, y1, conf1 = keypoints[kpt1_idx]
                x2, y2, conf2 = keypoints[kpt2_idx]
                
                # Only draw if both keypoints have sufficient confidence
                if conf1 > confidence_threshold and conf2 > confidence_threshold:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw keypoints
        for i, keypoint in enumerate(keypoints):
            if len(keypoint) >= 3:
                x, y, conf = keypoint
                if conf > confidence_threshold:
                    # Draw keypoint circle
                    cv2.circle(frame, (int(x), int(y)), 4, color, -1)
                    # Draw keypoint border
                    cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), 1)
        
        return frame
    
    def add_person_label(self, frame, bbox, person_id, confidence):
        """Add person ID label above bounding box"""
        x1, y1, x2, y2 = bbox
        color = self.person_colors[person_id % len(self.person_colors)]
        
        # Create label text
        label = f"Person {person_id + 1} ({confidence:.2f})"
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw background rectangle
        cv2.rectangle(frame, (int(x1), int(y1) - text_height - 10), 
                     (int(x1) + text_width, int(y1)), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        return frame
    
    def process_video(self, input_path, output_path, confidence_threshold=0.3):
        """Process the entire video and add pose overlays"""
        if not os.path.exists(input_path):
            print(f"Error: Input video '{input_path}' not found!")
            return False
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video '{input_path}'")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        # Process video with tracking for consistent person IDs
        results = self.model.track(input_path, stream=True, verbose=False, tracker="bytetrack.yaml")
        
        for result in results:
            frame = result.orig_img.copy()
            
            if result.boxes is not None and result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                keypoints = result.keypoints.xy.cpu().numpy()  # Shape: [num_people, num_keypoints, 2]
                keypoint_confidences = result.keypoints.conf.cpu().numpy()  # Shape: [num_people, num_keypoints]
                
                # Get track IDs if available
                track_ids = result.boxes.id
                if track_ids is not None:
                    track_ids = track_ids.cpu().numpy().astype(int)
                else:
                    track_ids = list(range(len(boxes)))
                
                # Process each detected person
                for i in range(len(boxes)):
                    bbox = boxes[i]
                    confidence = confidences[i]
                    person_keypoints = keypoints[i]
                    kpt_confidences = keypoint_confidences[i]
                    person_id = track_ids[i] if i < len(track_ids) else i
                    
                    # Combine keypoint coordinates with confidences
                    keypoints_with_conf = []
                    for j in range(len(person_keypoints)):
                        x, y = person_keypoints[j]
                        conf = kpt_confidences[j]
                        keypoints_with_conf.append([x, y, conf])
                    
                    # Draw pose overlay
                    frame = self.draw_keypoints_and_skeleton(
                        frame, keypoints_with_conf, person_id, confidence_threshold
                    )
                    
                    # Add person label
                    frame = self.add_person_label(frame, bbox, person_id, confidence)
            
            # Write frame
            out.write(frame)
            frame_count += 1
            
            # Progress indicator
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"✅ Processing complete! Output saved to: {output_path}")
        return True

def main():
    # Initialize processor
    print("🚀 Initializing YOLOv11 Pose model...")
    processor = AcrobaticPoseProcessor(model_size='x')  # Use 's' for better accuracy, 'n' for speed
    
    # Input and output paths
    input_video = "acro.mov"
    output_video = "acro_with_pose.mp4"
    
    # Check if input video exists
    if not os.path.exists(input_video):
        print(f"❌ Input video '{input_video}' not found!")
        print("Please make sure 'acro.mov' is in the same directory as this script.")
        return
    
    print(f"📹 Processing video: {input_video}")
    print("⏳ This may take a few minutes depending on video length and your CPU...")
    
    # Process the video
    success = processor.process_video(
        input_path=input_video,
        output_path=output_video,
        confidence_threshold=0.3  # Adjust this to filter low-confidence keypoints
    )
    
    if success:
        print(f"🎉 Done! Check out your pose-annotated video: {output_video}")
        print("\n💡 Tips for better results:")
        print("- For higher accuracy, change model_size to 'm' or 'l' (slower but more accurate)")
        print("- Adjust confidence_threshold (0.1-0.8) to show more/fewer keypoints")
        print("- For GPU acceleration later, add device='cuda' to YOLO initialization")
    else:
        print("❌ Processing failed. Please check the error messages above.")

if __name__ == "__main__":
    main()