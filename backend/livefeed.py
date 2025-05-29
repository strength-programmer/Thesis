import argparse
import os
import time
import json
import datetime
import threading
import traceback
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

# ----------------------
# Configuration
# ----------------------
class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Human detection with action recognition")
        parser.add_argument("--REC", action='store_true', help="record stream")
        parser.add_argument("-weights", type=str, 
                           default=os.path.join(os.path.dirname(__file__), "weights", "avak_b16.pt"),
                           help="path to action recognition pretrained weights")
        # D:\JULIAN THE MAN\COLLEGE\FEU\THESIS\final\flask\backend\weights\avak_b16.pt
        # default=r"C:\\Users\\buanh\\Documents\\VSCODE\\thesis\\livefeed2\\webapp\\flask-template\\backend\\weights\\avak_b16.pt",
        parser.add_argument("-rate", type=int, default=8, help="sampling rate")
        parser.add_argument("-act_thresh", type=float, default=0.20, help="action confidence threshold")
        parser.add_argument("-det_thresh", type=float, default=0.6, help="person detection confidence threshold")
        parser.add_argument("-nms_thresh", type=float, default=0.75, help="NMS threshold for person detection")
        parser.add_argument("-act", type=lambda s: s.split(","), help="Comma-separated list of actions")
        parser.add_argument("-color", type=str, default='green', help="color to plot predictions")
        parser.add_argument("-font", type=float, default=0.7, help="font size")
        parser.add_argument("-line", type=int, default=2, help="line thickness")
        parser.add_argument("-delay", type=int, default=60, help="frame delay amount")
        parser.add_argument("-fps", type=int, default=10, help="target frames per second")
        parser.add_argument("-det_freq", type=int, default=6, help="detection frequency (every N frames)")
        parser.add_argument("-F", type=str, default="COFFEESHOP_1.mp4", help="file path for video input")
        parser.add_argument("-roi", type=str, default=None, help="Path to ROI coordinates JSON")
        self.args = parser.parse_args()

        # Load action classes
        if self.args.act is None:
            with open(os.path.join(os.path.dirname(__file__), 'ava_classes.json'), 'r') as f:
                self.args.act = json.load(f)
        else:
            self.args.act = [s.replace('_', ' ') for s in self.args.act]

        # Color mapping
        self.COLORS = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "cyan": (255, 255, 0),
            "magenta": (255, 0, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
        }
        self.color = self.COLORS.get(self.args.color, (0, 255, 0))
        self.font = self.args.font
        self.thickness = self.args.line

        # Fixed display dimensions
        self.display_width, self.display_height = 1280, 720

        # Fixed ROI line (hardcoded from previous user selection)
        self.roi_line_p1 = (7, 299)
        self.roi_line_p2 = (1253, 675)

        # Set video path
        if self.args.F and not os.path.isabs(self.args.F):
            self.args.F = os.path.join(os.path.dirname(__file__), self.args.F)

# ----------------------
# Model Setup
# ----------------------
class ModelManager:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Configure CUDA for performance
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        # Set up RT-DETR detector
        self.setup_detector()
        
        # Set up action recognition model
        self.setup_action_model()
        
        # Image processing setup
        self.imgsize = (240, 320)
        self.tfs = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # Action postprocessing
        self.postprocess = PostProcessActions()
        
        # Toggle states
        self.activity_recognition_on = True

    def setup_detector(self):
        """Initialize the RT-DETR person detector"""
        from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
        print("Loading RT-DETR model...")
        model_name = "PekingU/rtdetr_v2_r18vd"
        self.image_processor = RTDetrImageProcessor.from_pretrained(model_name)
        self.rt_detr_model = RTDetrV2ForObjectDetection.from_pretrained(model_name)
        
        # Use half precision if supported hardware is available
        self.use_half_precision = (self.device.type == "cuda" and 
                                  torch.cuda.is_available() and 
                                  torch.cuda.get_device_capability(0)[0] >= 7)
        if self.use_half_precision:
            self.rt_detr_model = self.rt_detr_model.half()
            print("Using half precision (FP16) for faster inference")
            
        self.rt_detr_model = self.rt_detr_model.to(self.device).eval()

    def setup_action_model(self):
        """Initialize the action recognition model"""
        from hb import get_hb
        print("Loading Action Recognition model...")
        self.action_model = get_hb(size='b', pretrain=None, det_token_num=20, 
                              text_lora=True, num_frames=9)['hb']
        self.action_model.load_state_dict(
            torch.load(self.config.args.weights, map_location='cpu', weights_only=True), 
            strict=False
        )
        self.action_model.to(self.device).eval()
        
        # Process action captions
        self.captions = self.config.args.act
        self.text_embeds = F.normalize(self.action_model.encode_text(self.captions), dim=-1)
    
    def detect_persons(self, frame):
        """Detect persons in a frame using RT-DETR model"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.use_half_precision:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.rt_detr_model(**inputs)
            
        target_sizes = torch.tensor([(pil_image.height, pil_image.width)]).to(self.device)
        results = self.image_processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=self.config.args.det_thresh
        )[0]
        
        results = {k: v.cpu() for k, v in results.items()}
        
        # Filter for persons only
        person_boxes_xyxy, person_scores = [], []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if self.rt_detr_model.config.id2label[label.item()].lower() == "person":
                person_scores.append(score.item())
                person_boxes_xyxy.append([int(i) for i in box.tolist()])
                
        # Apply NMS
        person_detections = []
        if person_boxes_xyxy:
            indices = cv2.dnn.NMSBoxes(
                person_boxes_xyxy, 
                person_scores, 
                self.config.args.det_thresh, 
                self.config.args.nms_thresh
            )
            for idx in indices:
                idx = idx[0] if isinstance(idx, np.ndarray) else idx
                person_detections.append({
                    "score": person_scores[idx], 
                    "box": person_boxes_xyxy[idx]
                })
                
        return person_detections
    
    def recognize_actions(self, buffer, persons_in_roi):
        """Recognize actions for people in ROI using the action model"""
        if not self.activity_recognition_on or len(buffer) < 9:
            return {}
        
        try:
            frames_to_use = buffer[-9:]  # Use last 9 frames
            clip_torch = torch.from_numpy(np.array(frames_to_use)).to(self.device) / 255
            clip_torch = self.tfs(clip_torch)
            
            person_actions = {}
            for i, person in enumerate(persons_in_roi):
                with torch.no_grad():
                    features = self.action_model.encode_vision(clip_torch.unsqueeze(0))
                    action_probs = F.normalize(features['pred_logits'], dim=-1) @ self.text_embeds.T
                    actions = self.postprocess(action_probs, threshold=self.config.args.act_thresh)
                    person_actions[i] = actions
            return person_actions
            
        except Exception as e:
            print(f"Error in action recognition: {e}")
            return {}
            
    def toggle_activity_recognition(self):
        """Toggle the activity recognition feature on/off"""
        self.activity_recognition_on = not self.activity_recognition_on
        print(f"Human activity recognition toggled to: {self.activity_recognition_on}")
        return self.activity_recognition_on
        
    def get_activity_recognition_state(self):
        """Get current state of activity recognition"""
        return self.activity_recognition_on

# ----------------------
# Utilities
# ----------------------
class PostProcessActions:
    """Process action recognition results"""
    def __call__(self, action_probs, threshold=0.25):
        # Keep everything on GPU until numpy conversion is needed
        if len(action_probs.shape) == 3:
            probs = action_probs[0, 0]
        elif len(action_probs.shape) == 2:
            probs = action_probs[0]
        else:
            probs = action_probs
            
        mask = probs > threshold
        indices = torch.nonzero(mask).flatten()
        actions = [(int(i), float(probs[i].item())) for i in indices]
        return sorted(actions, key=lambda x: x[1], reverse=True)

def is_on_active_side(pt, line_p1, line_p2):
    """Check if a point is on the 'active' side of the ROI line"""
    x, y = pt
    x1, y1 = line_p1
    x2, y2 = line_p2
    # Vector from p1 to p2
    dx, dy = x2 - x1, y2 - y1
    # Vector from p1 to pt
    dxp, dyp = x - x1, y - y1
    # Cross product
    cross = dx * dyp - dy * dxp
    return cross > 0  # Positive cross product defines the active side

# ----------------------
# Video and Recording Management
# ----------------------
class VideoManager:
    def __init__(self, config, model_manager):
        self.config = config
        self.model_manager = model_manager
        
        # Video buffers
        self.buffer = []         # Action recognition buffer (processed frames)
        self.delay_buffer = []   # Delay buffer (raw frames)
        
        # State variables
        self.fps_history = []
        self.last_person_detections = []
        self.frame_count = 0
        self.roi_id = 1
        
        # Threading
        self.latest_frame = None
        self.video_thread_started = False
        self.video_thread_lock = threading.Lock()
        
        # Recording setup - support for two types
        self.original_writer = None
        self.segmented_writer = None
        self.original_recording_active = False
        self.segmented_recording_active = False
        if self.config.args.REC:
            self.setup_recording()
            
    def setup_recording(self, recording_type="original"):
        """Setup video recording for either original or segmented recording"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create absolute path to recordings directory
        recordings_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "recordings")
        if not os.path.exists(recordings_dir):
            os.makedirs(recordings_dir)
        
        # Initially save both recordings in the main recordings folder
        # The segmented recording will be post-processed and moved later
        if recording_type == "segmented":
            output_path = os.path.join(recordings_dir, f"raw_segmented_recording_{timestamp}.mp4")
        else:
            output_path = os.path.join(recordings_dir, f"original_recording_{timestamp}.mp4")
        
        # Try different MP4-compatible codecs in order of performance and browser compatibility
        mp4_codecs = ['MJPG', 'mp4v', 'X264', 'avc1']  # MJPG is fastest for real-time recording
        
        success = False
        writer = None
        
        for codec in mp4_codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    max(5, min(self.config.args.fps, 30)),  # Limit FPS to 5-30 for better compatibility
                    (self.config.display_width, self.config.display_height)
                )
                
                # Test if the writer is working by writing a test frame
                if writer.isOpened():
                    test_frame = np.zeros((self.config.display_height, self.config.display_width, 3), dtype=np.uint8)
                    test_frame.fill(50)  # Gray test frame
                    writer.write(test_frame)
                    print(f"{recording_type.capitalize()} recording started with {codec} codec: {output_path}")
                    
                    # Assign to appropriate writer
                    if recording_type == "segmented":
                        self.segmented_writer = writer
                        self.current_segmented_recording_path = output_path
                        self.segmented_recording_active = True
                    else:
                        self.original_writer = writer
                        self.current_original_recording_path = output_path
                        self.original_recording_active = True
                    
                    success = True
                    break
                else:
                    if writer:
                        writer.release()
                    print(f"Failed to initialize {recording_type} recording with {codec} codec, trying next...")
                    
            except Exception as e:
                print(f"Error with {codec} codec for {recording_type} recording: {e}")
                if writer:
                    writer.release()
                continue
        
        if not success:
            print(f"ERROR: Could not initialize {recording_type} video writer with any MP4 codec!")
            if recording_type == "segmented":
                self.segmented_writer = None
            else:
                self.original_writer = None

    def stop_recording(self, recording_type="original"):
        """Stop video recording for the specified type"""
        if recording_type == "segmented":
            if self.segmented_writer is not None:
                self.segmented_writer.release()
                self.segmented_writer = None
                self.segmented_recording_active = False
                if hasattr(self, 'current_segmented_recording_path'):
                    print(f"Segmented recording stopped and saved: {self.current_segmented_recording_path}")
                    # Verify the file was created and has content
                    if os.path.exists(self.current_segmented_recording_path):
                        file_size = os.path.getsize(self.current_segmented_recording_path)
                        print(f"Segmented recording file size: {file_size} bytes")
                        
                        # Start post-processing in background thread
                        threading.Thread(
                            target=self._post_process_segmented_recording,
                            args=(self.current_segmented_recording_path,),
                            daemon=True
                        ).start()
                    else:
                        print("ERROR: Segmented recording file was not created!")
                else:
                    print("Segmented recording stopped")
        else:
            if self.original_writer is not None:
                self.original_writer.release()
                self.original_writer = None
                self.original_recording_active = False
                if hasattr(self, 'current_original_recording_path'):
                    print(f"Original recording stopped and saved: {self.current_original_recording_path}")
                    # Verify the file was created and has content
                    if os.path.exists(self.current_original_recording_path):
                        file_size = os.path.getsize(self.current_original_recording_path)
                        print(f"Original recording file size: {file_size} bytes")
                    else:
                        print("ERROR: Original recording file was not created!")
                else:
                    print("Original recording stopped")
    
    def _post_process_segmented_recording(self, raw_recording_path):
        """Post-process the raw segmented recording using auto.py"""
        try:
            print(f"Starting post-processing of: {raw_recording_path}")
            
            # Create segmented recordings directory
            recordings_dir = os.path.dirname(raw_recording_path)
            segmented_dir = os.path.join(recordings_dir, "segmented_recordings")
            if not os.path.exists(segmented_dir):
                os.makedirs(segmented_dir)
            
            # Generate output path for processed video
            filename = os.path.basename(raw_recording_path)
            processed_filename = filename.replace("raw_segmented_recording_", "segmented_recording_")
            processed_path = os.path.join(segmented_dir, processed_filename)
            
            # Import and run auto.py processing
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), 'postprocess'))
            from auto import auto_annotate_video_with_models
            
            # Run the auto annotation
            auto_annotate_video_with_models(
                video_path=raw_recording_path,
                output_video_path=processed_path,
                det_model="PekingU/rtdetr_v2_r18vd",
                sam_model_path="mobile_sam.pt",  # Changed from sam2.1_t.pt to mobile_sam.pt
                device="cuda" if torch.cuda.is_available() else "cpu",
                conf=0.55,
                visualize=True,
                use_sam=True,  # Will fallback to False if SAM model fails to load
                iou_threshold=0.3
            )
            
            # Remove the raw recording file after successful processing
            if os.path.exists(processed_path):
                os.remove(raw_recording_path)
                print(f"Post-processing complete! Processed video saved at: {processed_path}")
                print(f"Raw recording file removed: {raw_recording_path}")
            else:
                print(f"ERROR: Post-processing failed, processed file not found: {processed_path}")
                
        except Exception as e:
            print(f"ERROR in post-processing: {e}")
            traceback.print_exc()
            
    def process_frame(self, frame):
        """Process a single frame and return the annotated output"""
        self.frame_count += 1
        
        # Flip and resize the frame
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (self.config.display_width, self.config.display_height))
        
        # Store current frame for employee capture
        self.current_frame = frame.copy()
        
        # Add to delay buffer
        self.delay_buffer.append(frame.copy())
        
        # Process only if we have enough frames in the delay buffer
        if len(self.delay_buffer) <= self.config.args.delay + 20:
            return frame  # Return original frame if not enough delay buffer
            
        # Get delayed frame and process
        delayed_frame = self.delay_buffer.pop(0)
        
        # Person detection (only run every det_freq frames for better performance)
        person_detections = []
        if self.frame_count % self.config.args.det_freq == 0:
            person_detections = self.model_manager.detect_persons(delayed_frame)
            self.last_person_detections = person_detections
        else:
            person_detections = self.last_person_detections
            
        # Store current detections for employee capture
        # Convert detections to simple format for employee capture
        self.current_detections = []
        for person in person_detections:
            box, score = person["box"], person["score"]
            # Store as [x1, y1, x2, y2, confidence]
            self.current_detections.append([box[0], box[1], box[2], box[3], score])
            
        # Process frame for action recognition
        delayed_frame_resized = cv2.resize(
            delayed_frame, 
            (self.model_manager.imgsize[1], self.model_manager.imgsize[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        color_image = delayed_frame_resized.transpose(2, 0, 1)
        self.buffer.append(color_image)
        
        # Keep buffer size managed
        if len(self.buffer) > 2 * 9:  # 9 is required_frames
            self.buffer.pop(0)
            
        # Find persons in ROI and track their original indices
        out_frame = delayed_frame.copy()
        persons_in_roi = []
        roi_to_original_mapping = {}  # Maps ROI index to original detection index
        
        for i, p in enumerate(person_detections):
            box = p["box"]
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
            if is_on_active_side((cx, cy), self.config.roi_line_p1, self.config.roi_line_p2):
                roi_index = len(persons_in_roi)
                persons_in_roi.append(p)
                roi_to_original_mapping[roi_index] = i
                
        # Action recognition
        person_actions = {}
        if self.model_manager.activity_recognition_on and len(self.buffer) >= 9:
            person_actions = self.model_manager.recognize_actions(self.buffer, persons_in_roi)
            
        # Store current activities for employee capture using original detection indices
        self.current_activities = {}
        for roi_idx, actions in person_actions.items():
            original_idx = roi_to_original_mapping.get(roi_idx)
            if original_idx is not None:
                # Convert actions to readable format
                action_names = []
                for action_idx, confidence in actions:
                    if action_idx < len(self.model_manager.captions):
                        action_names.append(self.model_manager.captions[action_idx])
                self.current_activities[original_idx] = action_names
            
        # Draw ROI line
        cv2.line(out_frame, self.config.roi_line_p1, self.config.roi_line_p2, (0, 255, 255), 2)
        
        # Draw bounding boxes and annotations
        person_count = 0
        self.roi_id = 1
        
        for i, person in enumerate(person_detections):
            box, score = person["box"], person["score"]
            cv2.rectangle(
                out_frame, 
                (box[0], box[1]), 
                (box[2], box[3]), 
                self.config.color, 
                self.config.thickness
            )
            
            label_text = f"Person: {score:.2f}"
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
            
            if is_on_active_side((cx, cy), self.config.roi_line_p1, self.config.roi_line_p2):
                label_text += f" | ROI: employeeID:[{self.roi_id}]"
                self.roi_id += 1
                person_count += 1
                
            cv2.putText(
                out_frame, 
                label_text, 
                (box[0], box[1] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                self.config.font, 
                self.config.color, 
                self.config.thickness
            )
            
            # Display actions for this person
            if (self.model_manager.activity_recognition_on and 
                i in person_actions and 
                person_actions[i]):
                y_offset = 25
                for action_idx, confidence in person_actions[i]:
                    action_text = f"{self.model_manager.captions[action_idx]}: {confidence:.2f}"
                    cv2.putText(
                        out_frame, 
                        action_text, 
                        (box[0], box[1] + y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        self.config.font, 
                        self.config.color, 
                        self.config.thickness
                    )
                    y_offset += 25
                    
        # Draw stats
        cv2.putText(
            out_frame, 
            f"Total Persons: {len(person_detections)}", 
            (self.config.display_width - 320, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, 
            (255, 0, 0), 
            3
        )
        
        cv2.putText(
            out_frame, 
            f"Persons in ROI: {person_count}", 
            (10, self.config.display_height - 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (0, 255, 255), 
            2
        )
        
        # Calculate and display FPS
        current_time = time.time()
        if self.frame_count % 10 == 0:
            fps = 10 / (current_time - self.start_time) if hasattr(self, 'start_time') else 10
            self.fps_history.append(fps)
            if len(self.fps_history) > 10:
                self.fps_history.pop(0)
            self.start_time = current_time
            
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        cv2.putText(
            out_frame, 
            f"FPS: {avg_fps:.1f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            self.config.font, 
            (0, 0, 255), 
            self.config.thickness
        )
        
        # Write frame to video if recording (optimized for dual recording)
        recording_active = (self.original_writer is not None and self.original_recording_active) or \
                          (self.segmented_writer is not None and self.segmented_recording_active)
        
        if recording_active:
            # Only encode the frame once for both writers to save processing time
            try:
                if self.original_writer is not None and self.original_recording_active:
                    self.original_writer.write(out_frame)
                
                if self.segmented_writer is not None and self.segmented_recording_active:
                    # For segmented recording, you could apply additional processing here
                    # For now, we'll record the same frame but you can modify this later
                    self.segmented_writer.write(out_frame)
                    
                # Debug info every 30 frames when recording
                if self.frame_count % 30 == 0:
                    original_status = "ON" if self.original_recording_active else "OFF"
                    segmented_status = "ON (will post-process)" if self.segmented_recording_active else "OFF"
                    print(f"Recording status - Original: {original_status}, Segmented: {segmented_status}, Frame: {self.frame_count}")
                    
            except Exception as e:
                print(f"Error writing video frame: {e}")
        
        # Periodic CUDA cache clearing
        if self.frame_count % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return out_frame
        
    def start_background_video_thread(self):
        """Start video processing in a background thread"""
        if not self.video_thread_started:
            self.start_time = time.time()
            t = threading.Thread(target=self._video_background_worker, daemon=True)
            t.start()
            self.video_thread_started = True
        
    def _video_background_worker(self):
        """Worker function for background video processing"""
        cap = cv2.VideoCapture(self.config.args.F if self.config.args.F else 0)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.config.args.F if self.config.args.F else 0}")
            
        try:
            while True:
                frame_start_time = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                    
                processed_frame = self.process_frame(frame)
                if processed_frame is not None:
                    # Encode to JPEG in a try-catch to prevent blocking
                    try:
                        ret, jpeg = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ret:
                            with self.video_thread_lock:
                                self.latest_frame = jpeg.tobytes()
                    except Exception as e:
                        print(f"Error encoding frame: {e}")
                
                # Limit playback to target FPS
                elapsed = time.time() - frame_start_time
                target_frame_time = 1.0 / self.config.args.fps
                delay = max(0.001, target_frame_time - elapsed)  # Minimum 1ms delay
                time.sleep(delay)
                
        except Exception as e:
            print(f"Error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            print("Video thread cleanup complete.")
            
    def get_latest_frame(self):
        """Get the latest processed frame"""
        with self.video_thread_lock:
            return self.latest_frame
            
    def frame_generator(self):
        """Generator that yields processed frames as JPEG for Flask streaming"""
        cap = cv2.VideoCapture(self.config.args.F if self.config.args.F else 0)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.config.args.F if self.config.args.F else 0}")
            
        self.start_time = time.time()
        
        try:
            while True:
                frame_start_time = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                    
                processed_frame = self.process_frame(frame)
                if processed_frame is not None:
                    ret, jpeg = cv2.imencode('.jpg', processed_frame)
                    if ret:
                        yield jpeg.tobytes()
                
                # Limit playback to target FPS
                elapsed = time.time() - frame_start_time
                delay = max(1, int((1.0 / self.config.args.fps - elapsed) * 1000))
                time.sleep(delay / 1000.0)
                
        except Exception as e:
            print(f"Error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            print("Frame generator cleanup complete.")

# ----------------------
# Employee Activity Monitoring
# ----------------------
# Employee activity storage
employee_captures = []
employee_act_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'employee_act')

# Create employee_act directory if it doesn't exist
if not os.path.exists(employee_act_dir):
    os.makedirs(employee_act_dir)

def capture_employee_activity():
    """Capture current frame with employee detection and activity recognition"""
    try:
        # Get current frame and detection results
        current_frame = video_manager.current_frame
        if current_frame is None:
            return {"success": False, "error": "No current frame available"}
        
        # Get current detections and activities
        detections = getattr(video_manager, 'current_detections', [])
        activities = getattr(video_manager, 'current_activities', {})
        
        if not detections:
            return {"success": True, "employees_detected": 0, "message": "No employees detected in current frame"}
        
        # Get frame dimensions for quadrant calculation
        h, w = current_frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Get ROI line coordinates from config
        roi_line_p1 = (7, 299)
        roi_line_p2 = (1253, 675)
        
        captured_employees = []
        timestamp = datetime.datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Create timestamped folder only if we have valid captures
        timestamp_folder = None
        
        for i, detection in enumerate(detections):
            try:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, detection[:4])
                confidence = detection[4] if len(detection) > 4 else 0.0
                
                # Calculate center point of bounding box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Check if person is in ROI using the existing ROI line logic
                if not is_on_active_side((cx, cy), roi_line_p1, roi_line_p2):
                    continue  # Skip if not in ROI
                
                # Check if person is in quadrants 2 or 3 (left side of frame)
                if cx >= center_x:
                    continue  # Skip if in right quadrants (1 or 4)
                
                # Get activities for this detection
                detection_activities = activities.get(i, [])
                if isinstance(detection_activities, str):
                    detection_activities = [detection_activities]
                elif not isinstance(detection_activities, list):
                    detection_activities = []
                
                # Only capture if person has detected activities
                if not detection_activities:
                    continue  # Skip if no activities detected
                
                # Calculate the dimensions of the bounding box
                box_width = x2 - x1
                box_height = y2 - y1
                
                # Add larger margins to ensure we capture the full person
                # Use 15% margin for width and 25% for height to account for full body
                margin_x = int(box_width * 0.15)
                margin_y = int(box_height * 0.25)
                
                # Calculate crop coordinates with margins
                crop_x1 = max(0, x1 - margin_x)
                crop_y1 = max(0, y1 - margin_y)
                crop_x2 = min(w, x2 + margin_x)
                crop_y2 = min(h, y2 + margin_y)
                
                # Ensure minimum dimensions
                min_width = 200
                min_height = 400
                
                # Adjust crop dimensions if they're too small
                if crop_x2 - crop_x1 < min_width:
                    center_x_crop = (crop_x1 + crop_x2) // 2
                    crop_x1 = max(0, center_x_crop - min_width // 2)
                    crop_x2 = min(w, center_x_crop + min_width // 2)
                
                if crop_y2 - crop_y1 < min_height:
                    center_y_crop = (crop_y1 + crop_y2) // 2
                    crop_y1 = max(0, center_y_crop - min_height // 2)
                    crop_y2 = min(h, center_y_crop + min_height // 2)
                
                # Ensure we have a valid crop area
                if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                    print(f"Invalid crop dimensions for detection {i}: x1={crop_x1}, y1={crop_y1}, x2={crop_x2}, y2={crop_y2}")
                    continue
                
                # Crop the employee from the frame
                try:
                    employee_crop = current_frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                    
                    if employee_crop.size == 0:
                        print(f"Warning: Empty crop detected for employee {i+1}")
                        continue
                    
                    # Create timestamped folder if this is the first valid capture
                    if timestamp_folder is None:
                        timestamp_folder = os.path.join(employee_act_dir, timestamp_str)
                        if not os.path.exists(timestamp_folder):
                            os.makedirs(timestamp_folder)
                    
                    # Generate employee ID and filename
                    employee_id = f"EMP_{i+1:03d}"
                    filename = f"employee_{employee_id}_{timestamp_str}.jpg"
                    filepath = os.path.join(timestamp_folder, filename)
                    
                    # Save the cropped image with high quality
                    cv2.imwrite(filepath, employee_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    # Determine quadrant for ROI info
                    quadrant = 2 if cy < center_y else 3
                    roi_info = f"ROI Quadrant {quadrant}"
                    
                    # Create capture record
                    capture_data = {
                        "employee_id": employee_id,
                        "timestamp": timestamp.isoformat(),
                        "image_path": f"{timestamp_str}/{filename}",
                        "activities": detection_activities,
                        "confidence": confidence,
                        "bounding_box": [x1, y1, x2, y2],
                        "roi_info": roi_info,
                        "crop_dimensions": {
                            "x1": crop_x1,
                            "y1": crop_y1,
                            "x2": crop_x2,
                            "y2": crop_y2,
                            "width": crop_x2 - crop_x1,
                            "height": crop_y2 - crop_y1
                        }
                    }
                    
                    captured_employees.append(capture_data)
                    employee_captures.append(capture_data)
                    
                except Exception as crop_error:
                    print(f"Error cropping detection {i}: {crop_error}")
                    continue
                
            except Exception as e:
                print(f"Error processing detection {i}: {e}")
                continue
        
        # Keep only recent captures (last 100)
        if len(employee_captures) > 100:
            employee_captures[:] = employee_captures[-100:]
        
        return {
            "success": True,
            "employees_detected": len(captured_employees),
            "timestamp": timestamp.isoformat(),
            "captures": captured_employees
        }
        
    except Exception as e:
        print(f"Error in capture_employee_activity: {e}")
        return {"success": False, "error": str(e)}

def get_employee_captures():
    """Get list of employee captures with statistics"""
    try:
        # Sort captures by timestamp (newest first)
        sorted_captures = sorted(employee_captures, key=lambda x: x['timestamp'], reverse=True)
        
        # Calculate statistics
        total_captures = len(employee_captures)
        unique_employees = len(set(capture['employee_id'] for capture in employee_captures))
        
        statistics = {
            "total_captures": total_captures,
            "total_employees": unique_employees
        }
        
        return {
            "success": True,
            "captures": sorted_captures[:50],  # Return latest 50 captures
            "statistics": statistics
        }
        
    except Exception as e:
        print(f"Error in get_employee_captures: {e}")
        return {"success": False, "error": str(e)}

def clear_employee_captures():
    """Clear all employee captures and delete associated files"""
    try:
        global employee_captures
        
        # Delete all files in employee_act directory
        if os.path.exists(employee_act_dir):
            import shutil
            shutil.rmtree(employee_act_dir)
            os.makedirs(employee_act_dir)
        
        # Clear the captures list
        employee_captures.clear()
        
        return {"success": True, "message": "All employee captures cleared"}
        
    except Exception as e:
        print(f"Error in clear_employee_captures: {e}")
        return {"success": False, "error": str(e)}

# ----------------------
# Main Application
# ----------------------
# Create global instances for use in Flask application
config = Config()
model_manager = ModelManager(config)
video_manager = VideoManager(config, model_manager)

def start_background_video_thread():
    """Start the video processing in a background thread"""
    return video_manager.start_background_video_thread()

def get_latest_frame():
    """Get the latest processed frame"""
    return video_manager.get_latest_frame()

def toggle_activity_recognition():
    """Toggle the activity recognition feature on/off"""
    return model_manager.toggle_activity_recognition()

def get_activity_recognition_state():
    """Get current state of activity recognition"""
    return model_manager.get_activity_recognition_state()

def start_recording(recording_type="original"):
    """Start recording of the specified type"""
    video_manager.setup_recording(recording_type)
    return video_manager.original_recording_active if recording_type == "original" else video_manager.segmented_recording_active

def stop_recording(recording_type="original"):
    """Stop recording of the specified type"""
    video_manager.stop_recording(recording_type)
    return not (video_manager.original_recording_active if recording_type == "original" else video_manager.segmented_recording_active)

def get_recording_status():
    """Get the status of both recording types"""
    return {
        "original": video_manager.original_recording_active,
        "segmented": video_manager.segmented_recording_active
    }

def start_dual_recording():
    """Start both original and segmented recordings simultaneously"""
    original_success = start_recording('original')
    segmented_success = start_recording('segmented')
    return original_success and segmented_success

def stop_dual_recording():
    """Stop both original and segmented recordings simultaneously"""
    stop_recording('original')
    stop_recording('segmented')
    return True
