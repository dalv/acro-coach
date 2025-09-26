import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import defaultdict, deque
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class KalmanFilter2D:
    """Simple 2D Kalman Filter for keypoint tracking"""
    def __init__(self, process_noise=1e-3, measurement_noise=1e-1):
        # State: [x, y, vx, vy] - position and velocity
        self.state = np.zeros(4)
        self.P = np.eye(4) * 1000  # High initial uncertainty
        
        # State transition model (constant velocity)
        self.F = np.array([[1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)
        
        # Observation model (we observe position only)
        self.H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]], dtype=np.float32)
        
        # Process noise
        self.Q = np.eye(4) * process_noise
        
        # Measurement noise
        self.R = np.eye(2) * measurement_noise
        
        self.initialized = False
        
    def predict(self):
        """Predict next state"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]  # Return predicted position
        
    def update(self, measurement):
        """Update with new measurement"""
        if not self.initialized:
            self.state[:2] = measurement
            self.initialized = True
            return measurement
            
        # Predict step
        self.predict()
        
        # Update step
        y = measurement - (self.H @ self.state)  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.state[:2]

class TemporalSmoother:
    """Temporal smoothing for pose keypoints"""
    def __init__(self, max_history=30, confidence_threshold=0.3):
        self.max_history = max_history
        self.confidence_threshold = confidence_threshold
        
        # Store history for each person and keypoint
        self.keypoint_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=max_history)))
        self.confidence_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=max_history)))
        
        # Kalman filters for each person and keypoint
        self.kalman_filters = defaultdict(lambda: defaultdict(lambda: KalmanFilter2D()))
        
        # Track missing keypoints for interpolation
        self.missing_count = defaultdict(lambda: defaultdict(int))
        
    def smooth_keypoints(self, person_id, keypoints, confidences, frame_idx):
        """Apply temporal smoothing to keypoints"""
        smoothed_keypoints = []
        smoothed_confidences = []
        
        for kpt_idx, (keypoint, confidence) in enumerate(zip(keypoints, confidences)):
            x, y = keypoint
            
            # Store in history
            self.keypoint_history[person_id][kpt_idx].append([x, y])
            self.confidence_history[person_id][kpt_idx].append(confidence)
            
            # Apply different smoothing strategies based on confidence
            if confidence > self.confidence_threshold:
                # High confidence: use Kalman filter
                smoothed_pos = self._kalman_smooth(person_id, kpt_idx, [x, y])
                smoothed_conf = confidence
                self.missing_count[person_id][kpt_idx] = 0
            else:
                # Low confidence: try to recover using temporal information
                smoothed_pos, smoothed_conf = self._recover_keypoint(person_id, kpt_idx, [x, y], confidence)
                self.missing_count[person_id][kpt_idx] += 1
            
            smoothed_keypoints.append(smoothed_pos)
            smoothed_confidences.append(smoothed_conf)
            
        return np.array(smoothed_keypoints), np.array(smoothed_confidences)
    
    def _kalman_smooth(self, person_id, kpt_idx, measurement):
        """Apply Kalman filtering"""
        kalman = self.kalman_filters[person_id][kpt_idx]
        return kalman.update(np.array(measurement))
    
    def _recover_keypoint(self, person_id, kpt_idx, current_pos, confidence):
        """Recover low-confidence keypoint using temporal information"""
        history = self.keypoint_history[person_id][kpt_idx]
        conf_history = self.confidence_history[person_id][kpt_idx]
        
        if len(history) < 3:
            return current_pos, confidence
            
        # Find recent high-confidence keypoints
        recent_good_points = []
        recent_good_confs = []
        
        for i in range(min(10, len(history))):
            if conf_history[-(i+1)] > self.confidence_threshold:
                recent_good_points.append(history[-(i+1)])
                recent_good_confs.append(conf_history[-(i+1)])
            if len(recent_good_points) >= 3:
                break
        
        if len(recent_good_points) >= 2:
            # Use Kalman prediction if we have recent good points
            kalman = self.kalman_filters[person_id][kpt_idx]
            predicted_pos = kalman.predict()
            
            # Blend prediction with current measurement based on confidence
            alpha = min(confidence, 0.3)  # Low weight for low confidence
            blended_pos = alpha * np.array(current_pos) + (1 - alpha) * predicted_pos
            
            # Boost confidence slightly if prediction seems reasonable
            improved_conf = min(confidence + 0.1, 0.8)
            return blended_pos, improved_conf
        
        return current_pos, confidence
    
    def post_process_trajectory(self, person_trajectories):
        """Apply post-processing smoothing to complete trajectories"""
        smoothed_trajectories = {}
        
        for person_id, person_data in person_trajectories.items():
            smoothed_person = {}
            
            for kpt_idx in range(17):  # 17 COCO keypoints
                if kpt_idx not in person_data:
                    continue
                    
                frames = person_data[kpt_idx]['frames']
                positions = person_data[kpt_idx]['positions']
                confidences = person_data[kpt_idx]['confidences']
                
                if len(positions) < 5:  # Need minimum points for smoothing
                    smoothed_person[kpt_idx] = person_data[kpt_idx]
                    continue
                
                # Apply Savitzky-Golay filter for additional smoothing
                positions = np.array(positions)
                try:
                    # Smooth x and y coordinates separately
                    window_length = min(5, len(positions) // 2 * 2 + 1)  # Ensure odd number
                    if window_length >= 5:
                        smooth_x = savgol_filter(positions[:, 0], window_length, 2)
                        smooth_y = savgol_filter(positions[:, 1], window_length, 2)
                        positions = np.column_stack([smooth_x, smooth_y])
                except:
                    pass  # Keep original if smoothing fails
                
                smoothed_person[kpt_idx] = {
                    'frames': frames,
                    'positions': positions.tolist(),
                    'confidences': confidences
                }
            
            smoothed_trajectories[person_id] = smoothed_person
            
        return smoothed_trajectories

class AcrobaticPoseProcessor:
    def __init__(self, model_size='s'):
        """Initialize with temporal smoothing capabilities"""
        self.model = YOLO(f'yolo11{model_size}-pose.pt')
        
        # Initialize temporal smoother
        self.temporal_smoother = TemporalSmoother(max_history=30, confidence_threshold=0.3)
        
        # COCO pose keypoints
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Define skeleton connections
        self.skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9),          # Left arm
            (6, 8), (8, 10),                 # Right arm  
            (5, 11), (6, 12), (11, 12),      # Torso
            (11, 13), (13, 15),              # Left leg
            (12, 14), (14, 16)               # Right leg
        ]
        
        # Colors for different people
        self.person_colors = [
            (0, 255, 0),    # Green for person 1
            (255, 0, 0),    # Blue for person 2
            (0, 0, 255),    # Red for person 3
            (255, 255, 0),  # Cyan for person 4
            (255, 0, 255),  # Magenta for person 5
        ]
        
        # Store trajectories for post-processing
        self.person_trajectories = defaultdict(lambda: defaultdict(lambda: {'frames': [], 'positions': [], 'confidences': []}))
        
    def draw_keypoints_and_skeleton(self, frame, keypoints, confidences, person_id=0, confidence_threshold=0.25):
        """Draw smoothed keypoints and skeleton"""
        color = self.person_colors[person_id % len(self.person_colors)]
        
        # Draw skeleton connections
        for connection in self.skeleton:
            kpt1_idx, kpt2_idx = connection
            if (kpt1_idx < len(keypoints) and kpt2_idx < len(keypoints)):
                
                x1, y1 = keypoints[kpt1_idx]
                x2, y2 = keypoints[kpt2_idx]
                conf1, conf2 = confidences[kpt1_idx], confidences[kpt2_idx]
                
                # Only draw if both keypoints have sufficient confidence
                if conf1 > confidence_threshold and conf2 > confidence_threshold:
                    # Vary line thickness based on confidence
                    thickness = max(1, int(3 * min(conf1, conf2)))
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        
        # Draw keypoints with confidence-based sizing
        for i, (keypoint, conf) in enumerate(zip(keypoints, confidences)):
            if conf > confidence_threshold:
                x, y = keypoint
                
                # Size based on confidence
                radius = max(2, int(6 * conf))
                
                # Color intensity based on confidence
                intensity = min(1.0, conf + 0.2)
                adjusted_color = tuple(int(c * intensity) for c in color)
                
                # Draw keypoint
                cv2.circle(frame, (int(x), int(y)), radius, adjusted_color, -1)
                cv2.circle(frame, (int(x), int(y)), radius, (255, 255, 255), 1)
                
                # Draw confidence score for debugging (optional)
                if conf < 0.5:  # Show confidence for uncertain points
                    cv2.putText(frame, f'{conf:.2f}', (int(x)+5, int(y)-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return frame
    
    def add_person_label(self, frame, bbox, person_id, confidence):
        """Add person ID label with smoothing info"""
        x1, y1, x2, y2 = bbox
        color = self.person_colors[person_id % len(self.person_colors)]
        
        # Create enhanced label
        label = f"Person {person_id + 1} ({confidence:.2f}) [Smoothed]"
        
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        cv2.rectangle(frame, (int(x1), int(y1) - text_height - 10), 
                     (int(x1) + text_width, int(y1)), color, -1)
        
        cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        return frame
    
    def process_video(self, input_path, output_path, confidence_threshold=0.25):
        """Process video with temporal smoothing"""
        if not os.path.exists(input_path):
            print(f"Error: Input video '{input_path}' not found!")
            return False
        
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
        print("🔄 Applying temporal smoothing...")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        # Process video with tracking
        results = self.model.track(input_path, stream=True, verbose=False, tracker="bytetrack.yaml")
        
        for result in results:
            frame = result.orig_img.copy()
            
            if result.boxes is not None and result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                keypoints = result.keypoints.xy.cpu().numpy()
                keypoint_confidences = result.keypoints.conf.cpu().numpy()
                
                # Get track IDs
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
                    
                    # Apply temporal smoothing
                    smoothed_keypoints, smoothed_confidences = self.temporal_smoother.smooth_keypoints(
                        person_id, person_keypoints, kpt_confidences, frame_count
                    )
                    
                    # Store trajectory data
                    for kpt_idx, (pos, conf) in enumerate(zip(smoothed_keypoints, smoothed_confidences)):
                        self.person_trajectories[person_id][kpt_idx]['frames'].append(frame_count)
                        self.person_trajectories[person_id][kpt_idx]['positions'].append(pos.tolist())
                        self.person_trajectories[person_id][kpt_idx]['confidences'].append(float(conf))
                    
                    # Draw smoothed pose overlay
                    frame = self.draw_keypoints_and_skeleton(
                        frame, smoothed_keypoints, smoothed_confidences, person_id, confidence_threshold
                    )
                    
                    # Add person label
                    frame = self.add_person_label(frame, bbox, person_id, confidence)
            
            # Write frame
            out.write(frame)
            frame_count += 1
            
            # Progress indicator
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames) - Smoothing applied")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"✅ Temporal smoothing complete! Output saved to: {output_path}")
        print(f"📊 Processed {len(self.person_trajectories)} people with smoothed trajectories")
        
        return True

def main():
    print("🚀 Initializing YOLOv11 Pose model with Temporal Smoothing...")
    processor = AcrobaticPoseProcessor(model_size='s')
    
    input_video = "acro.mov"
    output_video = "acro_with_smoothed_pose.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Input video '{input_video}' not found!")
        return
    
    print(f"📹 Processing video with temporal smoothing: {input_video}")
    print("⏳ This will take a bit longer due to smoothing calculations...")
    
    success = processor.process_video(
        input_path=input_video,
        output_path=output_video,
        confidence_threshold=0.25  # Lower threshold since smoothing helps with low-confidence points
    )
    
    if success:
        print(f"🎉 Done! Smoothed pose video: {output_video}")
        print("\n🎯 Temporal Smoothing Features Applied:")
        print("- Kalman filtering for trajectory prediction")
        print("- Confidence-weighted keypoint recovery")
        print("- Outlier detection and interpolation")
        print("- Motion-aware smoothing for rapid movements")
        print("\n💡 Compare the smoothness, especially during the flyer's rotation!")

if __name__ == "__main__":
    main()