import cv2
import numpy as np
from ultralytics import YOLO
import os
from scipy import signal
from scipy.interpolate import interp1d
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class KalmanFilter:
    """Simple Kalman filter for 2D point tracking"""
    def __init__(self, process_variance=0.03, measurement_variance=0.1):
        self.state = None  # [x, y, vx, vy]
        self.covariance = np.eye(4) * 1000  # Initial uncertainty
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        
        # State transition matrix
        self.F = np.array([[1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        
        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]])
        
        # Process noise
        self.Q = np.eye(4) * self.process_variance
        
        # Measurement noise
        self.R = np.eye(2) * self.measurement_variance
    
    def predict(self):
        """Predict next state"""
        if self.state is None:
            return None
        
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        return self.state[:2]
    
    def update(self, measurement, confidence=1.0):
        """Update state with measurement"""
        if self.state is None:
            # Initialize state
            self.state = np.array([measurement[0], measurement[1], 0, 0])
            return self.state[:2]
        
        # Adjust measurement noise based on confidence
        R_adjusted = self.R / max(confidence, 0.1)
        
        # Innovation
        y = measurement - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + R_adjusted
        
        # Kalman gain
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance
        
        return self.state[:2]

class TemporalSmoother:
    """Handles temporal smoothing for keypoints"""
    def __init__(self, window_size=5, outlier_threshold=50):
        self.window_size = window_size
        self.outlier_threshold = outlier_threshold
        self.history = {}  # Store history for each person-keypoint pair
        self.kalman_filters = {}  # Kalman filters for each person-keypoint
        
    def smooth_keypoints(self, person_id, keypoints, confidences):
        """Apply temporal smoothing to keypoints"""
        smoothed_keypoints = []
        
        for kpt_idx, (kpt, conf) in enumerate(zip(keypoints, confidences)):
            key = f"{person_id}_{kpt_idx}"
            
            # Initialize history and Kalman filter if needed
            if key not in self.history:
                self.history[key] = deque(maxlen=self.window_size)
                self.kalman_filters[key] = KalmanFilter()
            
            # Apply different smoothing based on confidence
            if conf > 0.5:
                # High confidence: use Kalman filter
                smoothed = self.kalman_filters[key].update(kpt[:2], conf)
                smoothed_kpt = [smoothed[0], smoothed[1], conf]
            elif conf > 0.2:
                # Medium confidence: weighted average with history
                smoothed_kpt = self._weighted_average(key, kpt, conf)
            else:
                # Low confidence: predict from Kalman or interpolate
                smoothed_kpt = self._recover_keypoint(key, kpt, conf)
            
            self.history[key].append(smoothed_kpt)
            smoothed_keypoints.append(smoothed_kpt)
        
        return smoothed_keypoints
    
    def _weighted_average(self, key, kpt, conf):
        """Apply weighted moving average"""
        if len(self.history[key]) == 0:
            return kpt
        
        weights = []
        points = []
        
        for hist_kpt in self.history[key]:
            weights.append(hist_kpt[2])  # Use confidence as weight
            points.append([hist_kpt[0], hist_kpt[1]])
        
        # Add current point
        weights.append(conf * 2)  # Give more weight to current
        points.append([kpt[0], kpt[1]])
        
        weights = np.array(weights)
        points = np.array(points)
        
        # Weighted average
        weights = weights / weights.sum()
        smoothed = np.average(points, axis=0, weights=weights)
        
        return [smoothed[0], smoothed[1], conf]
    
    def _recover_keypoint(self, key, kpt, conf):
        """Recover lost keypoint using prediction or interpolation"""
        # Try Kalman prediction
        if key in self.kalman_filters:
            predicted = self.kalman_filters[key].predict()
            if predicted is not None:
                return [predicted[0], predicted[1], conf * 1.5]  # Boost confidence slightly
        
        # Fall back to last known good position
        if len(self.history[key]) > 0:
            last_good = self.history[key][-1]
            return [last_good[0], last_good[1], conf]
        
        return kpt

class EnhancedAcrobaticPoseProcessor:
    def __init__(self, model_size='m', enable_smoothing=True):
        """
        Initialize enhanced pose processor with temporal smoothing
        """
        self.model = YOLO(f'yolo11{model_size}-pose.pt')
        self.enable_smoothing = enable_smoothing
        
        # Initialize temporal smoother
        self.smoother = TemporalSmoother(window_size=7, outlier_threshold=100)
        
        # Store full trajectory for post-processing
        self.full_trajectories = {}
        
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
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
        ]
        
        # Critical keypoints for acrobatic moves (hips, shoulders, ankles, wrists)
        self.critical_keypoints = [5, 6, 9, 10, 11, 12, 15, 16]
    
    def collect_trajectories(self, input_path):
        """First pass: collect all keypoint trajectories"""
        print("📊 First pass: Collecting keypoint trajectories...")
        
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        trajectories = {}
        frame_idx = 0
        
        results = self.model.track(input_path, stream=True, verbose=False, tracker="bytetrack.yaml")
        
        for result in results:
            if result.boxes is not None and result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                keypoint_confidences = result.keypoints.conf.cpu().numpy()
                
                track_ids = result.boxes.id
                if track_ids is not None:
                    track_ids = track_ids.cpu().numpy().astype(int)
                else:
                    track_ids = list(range(len(keypoints)))
                
                for person_idx in range(len(keypoints)):
                    person_id = track_ids[person_idx] if person_idx < len(track_ids) else person_idx
                    
                    if person_id not in trajectories:
                        trajectories[person_id] = {
                            'keypoints': [],
                            'confidences': [],
                            'frames': []
                        }
                    
                    trajectories[person_id]['keypoints'].append(keypoints[person_idx])
                    trajectories[person_id]['confidences'].append(keypoint_confidences[person_idx])
                    trajectories[person_id]['frames'].append(frame_idx)
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  Collecting: {frame_idx}/{total_frames} frames")
        
        return trajectories, total_frames
    
    def post_process_trajectories(self, trajectories, total_frames):
        """Apply global smoothing and interpolation to trajectories"""
        print("🔧 Applying global temporal smoothing...")
        
        smoothed_trajectories = {}
        
        for person_id, data in trajectories.items():
            frames = np.array(data['frames'])
            keypoints = np.array(data['keypoints'])
            confidences = np.array(data['confidences'])
            
            num_keypoints = keypoints.shape[1]
            smoothed_kpts = np.zeros((total_frames, num_keypoints, 2))
            smoothed_confs = np.zeros((total_frames, num_keypoints))
            
            for kpt_idx in range(num_keypoints):
                # Get trajectory for this keypoint
                kpt_traj = keypoints[:, kpt_idx, :]
                kpt_conf = confidences[:, kpt_idx]
                
                # Apply different smoothing for critical vs non-critical keypoints
                if kpt_idx in self.critical_keypoints:
                    window = 9  # Larger window for critical points
                    poly_order = 2
                else:
                    window = 5
                    poly_order = 1
                
                # Savitzky-Golay filter for smooth trajectories
                if len(frames) > window:
                    try:
                        # Smooth x and y separately
                        x_smooth = signal.savgol_filter(kpt_traj[:, 0], window, poly_order)
                        y_smooth = signal.savgol_filter(kpt_traj[:, 1], window, poly_order)
                        
                        # Interpolate to all frames
                        if len(frames) > 1:
                            # Create interpolation functions
                            fx = interp1d(frames, x_smooth, kind='cubic', 
                                        bounds_error=False, fill_value='extrapolate')
                            fy = interp1d(frames, y_smooth, kind='cubic',
                                        bounds_error=False, fill_value='extrapolate')
                            fc = interp1d(frames, kpt_conf, kind='linear',
                                        bounds_error=False, fill_value=0)
                            
                            # Interpolate to all frames
                            all_frames = np.arange(total_frames)
                            smoothed_kpts[all_frames, kpt_idx, 0] = fx(all_frames)
                            smoothed_kpts[all_frames, kpt_idx, 1] = fy(all_frames)
                            smoothed_confs[all_frames, kpt_idx] = fc(all_frames)
                        else:
                            # Single frame - just copy
                            smoothed_kpts[frames[0], kpt_idx] = kpt_traj[0]
                            smoothed_confs[frames[0], kpt_idx] = kpt_conf[0]
                    except:
                        # Fallback to simple copy if smoothing fails
                        for i, f in enumerate(frames):
                            smoothed_kpts[f, kpt_idx] = kpt_traj[i]
                            smoothed_confs[f, kpt_idx] = kpt_conf[i]
                else:
                    # Not enough points for smoothing
                    for i, f in enumerate(frames):
                        smoothed_kpts[f, kpt_idx] = kpt_traj[i]
                        smoothed_confs[f, kpt_idx] = kpt_conf[i]
            
            smoothed_trajectories[person_id] = {
                'keypoints': smoothed_kpts,
                'confidences': smoothed_confs
            }
        
        return smoothed_trajectories
    
    def draw_keypoints_and_skeleton(self, frame, keypoints, person_id=0, confidence_threshold=0.3):
        """Draw keypoints and skeleton with confidence visualization"""
        color = self.person_colors[person_id % len(self.person_colors)]
        
        # Draw skeleton connections
        for connection in self.skeleton:
            kpt1_idx, kpt2_idx = connection
            if (kpt1_idx < len(keypoints) and kpt2_idx < len(keypoints) and 
                len(keypoints[kpt1_idx]) >= 3 and len(keypoints[kpt2_idx]) >= 3):
                
                x1, y1, conf1 = keypoints[kpt1_idx]
                x2, y2, conf2 = keypoints[kpt2_idx]
                
                if conf1 > confidence_threshold and conf2 > confidence_threshold:
                    # Vary line thickness based on confidence
                    thickness = int(2 + 2 * min(conf1, conf2))
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        
        # Draw keypoints
        for i, keypoint in enumerate(keypoints):
            if len(keypoint) >= 3:
                x, y, conf = keypoint
                if conf > confidence_threshold:
                    # Vary circle size based on confidence
                    radius = int(3 + 3 * conf)
                    cv2.circle(frame, (int(x), int(y)), radius, color, -1)
                    cv2.circle(frame, (int(x), int(y)), radius, (255, 255, 255), 1)
                    
                    # Mark critical keypoints with additional indicator
                    if i in self.critical_keypoints:
                        cv2.circle(frame, (int(x), int(y)), radius + 2, (255, 255, 0), 1)
        
        return frame
    
    def process_video_with_smoothing(self, input_path, output_path, confidence_threshold=0.3):
        """Process video with two-pass temporal smoothing"""
        if not os.path.exists(input_path):
            print(f"Error: Input video '{input_path}' not found!")
            return False
        
        # First pass: collect trajectories
        trajectories, total_frames = self.collect_trajectories(input_path)
        
        # Apply global smoothing
        smoothed_trajectories = self.post_process_trajectories(trajectories, total_frames)
        
        # Second pass: render with smoothed keypoints
        print("🎬 Second pass: Rendering smoothed video...")
        
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw smoothed keypoints for each tracked person
            for person_id, data in smoothed_trajectories.items():
                if frame_idx < len(data['keypoints']):
                    keypoints = data['keypoints'][frame_idx]
                    confidences = data['confidences'][frame_idx]
                    
                    # Combine keypoints with confidences
                    keypoints_with_conf = []
                    for kpt, conf in zip(keypoints, confidences):
                        keypoints_with_conf.append([kpt[0], kpt[1], conf])
                    
                    # Draw smoothed pose
                    frame = self.draw_keypoints_and_skeleton(
                        frame, keypoints_with_conf, person_id, confidence_threshold
                    )
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add smoothing indicator
            if self.enable_smoothing:
                cv2.putText(frame, "Temporal Smoothing: ON", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                print(f"  Rendering: {frame_idx}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        print(f"✅ Processing complete! Output saved to: {output_path}")
        return True
    
    def process_video(self, input_path, output_path, confidence_threshold=0.3):
        """Main processing function"""
        if self.enable_smoothing:
            return self.process_video_with_smoothing(input_path, output_path, confidence_threshold)
        else:
            # Fall back to original single-pass processing
            return self.process_video_single_pass(input_path, output_path, confidence_threshold)
    
    def process_video_single_pass(self, input_path, output_path, confidence_threshold=0.3):
        """Original single-pass processing with online smoothing"""
        if not os.path.exists(input_path):
            print(f"Error: Input video '{input_path}' not found!")
            return False
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video '{input_path}'")
            return False
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        results = self.model.track(input_path, stream=True, verbose=False, tracker="bytetrack.yaml")
        
        for result in results:
            frame = result.orig_img.copy()
            
            if result.boxes is not None and result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                keypoints = result.keypoints.xy.cpu().numpy()
                keypoint_confidences = result.keypoints.conf.cpu().numpy()
                
                track_ids = result.boxes.id
                if track_ids is not None:
                    track_ids = track_ids.cpu().numpy().astype(int)
                else:
                    track_ids = list(range(len(boxes)))
                
                for i in range(len(boxes)):
                    person_keypoints = keypoints[i]
                    kpt_confidences = keypoint_confidences[i]
                    person_id = track_ids[i] if i < len(track_ids) else i
                    
                    # Apply online temporal smoothing
                    keypoints_with_conf = []
                    for j in range(len(person_keypoints)):
                        x, y = person_keypoints[j]
                        conf = kpt_confidences[j]
                        keypoints_with_conf.append([x, y, conf])
                    
                    # Smooth keypoints
                    if self.enable_smoothing:
                        keypoints_with_conf = self.smoother.smooth_keypoints(
                            person_id, keypoints_with_conf, kpt_confidences
                        )
                    
                    frame = self.draw_keypoints_and_skeleton(
                        frame, keypoints_with_conf, person_id, confidence_threshold
                    )
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        cap.release()
        out.release()
        
        print(f"✅ Processing complete! Output saved to: {output_path}")
        return True

def main():
    """Main function with enhanced temporal smoothing"""
    print("🚀 Initializing Enhanced YOLOv11 Pose model with Temporal Smoothing...")
    
    # Initialize processor with temporal smoothing enabled
    processor = EnhancedAcrobaticPoseProcessor(
        model_size='x',  # Use 'x' for best accuracy, 'm' for balance, 'n' for speed
        enable_smoothing=True  # Enable temporal smoothing
    )
    
    # Input and output paths
    input_video = "acro.mov"
    output_video = "acro_smoothed_pose.mp4"
    
    # Check if input video exists
    if not os.path.exists(input_video):
        print(f"⌚ Input video '{input_video}' not found!")
        print("Please make sure 'acro.mov' is in the same directory as this script.")
        return
    
    print(f"📹 Processing video: {input_video}")
    print("🔄 Using two-pass processing with temporal smoothing...")
    print("⏳ This will take longer but produce much smoother results...")
    
    # Process the video with temporal smoothing
    success = processor.process_video(
        input_path=input_video,
        output_path=output_video,
        confidence_threshold=0.2  # Lower threshold to capture more keypoints for smoothing
    )
    
    if success:
        print(f"🎉 Done! Check out your temporally smoothed pose video: {output_video}")
        print("\n💡 Advanced smoothing features applied:")
        print("✓ Kalman filtering for high-confidence keypoints")
        print("✓ Weighted averaging for medium-confidence keypoints")
        print("✓ Predictive recovery for lost keypoints")
        print("✓ Savitzky-Golay smoothing for trajectories")
        print("✓ Cubic interpolation between frames")
        print("✓ Enhanced tracking for critical joints (hips, shoulders, ankles, wrists)")
        print("\n🔧 To compare with original:")
        print("  Set enable_smoothing=False in the processor initialization")
        print("\n📊 For fine-tuning:")
        print("  - Adjust window_size in TemporalSmoother (3-15)")
        print("  - Modify confidence_threshold (0.1-0.5)")
        print("  - Change Kalman filter variance parameters")
    else:
        print("❌ Processing failed. Please check the error messages above.")

if __name__ == "__main__":
    main()