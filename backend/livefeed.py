import argparse
import os
import time
import json
import datetime
import threading
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
        
        # Recording setup
        self.writer = None
        if self.config.args.REC:
            self.setup_recording()
            
    def setup_recording(self):
        """Setup video recording"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("recordings", f"recording_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Changed from 'mp4v' to 'avc1' for better browser compatibility
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.config.args.fps,
            (self.config.display_width, self.config.display_height)
        )
        print(f"Recording started: {output_path}")
        
    def stop_recording(self):
        """Stop video recording"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print("Recording stopped")
            
    def process_frame(self, frame):
        """Process a single frame and return the annotated output"""
        self.frame_count += 1
        
        # Flip and resize the frame
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (self.config.display_width, self.config.display_height))
        
        # Add to delay buffer
        self.delay_buffer.append(frame.copy())
        
        # Process only if we have enough frames in the delay buffer
        if len(self.delay_buffer) <= self.config.args.delay + 20:
            return None
            
        # Get delayed frame and process
        delayed_frame = self.delay_buffer.pop(0)
        
        # Person detection (only run every det_freq frames for better performance)
        person_detections = []
        if self.frame_count % self.config.args.det_freq == 0:
            person_detections = self.model_manager.detect_persons(delayed_frame)
            self.last_person_detections = person_detections
        else:
            person_detections = self.last_person_detections
            
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
            
        # Find persons in ROI
        out_frame = delayed_frame.copy()
        persons_in_roi = []
        
        for p in person_detections:
            box = p["box"]
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
            if is_on_active_side((cx, cy), self.config.roi_line_p1, self.config.roi_line_p2):
                persons_in_roi.append(p)
                
        # Action recognition
        person_actions = {}
        if self.model_manager.activity_recognition_on and len(self.buffer) >= 9:
            person_actions = self.model_manager.recognize_actions(self.buffer, persons_in_roi)
            
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
        
        # Write frame to video if recording
        if self.writer is not None:
            self.writer.write(out_frame)
        
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
                    ret, jpeg = cv2.imencode('.jpg', processed_frame)
                    if ret:
                        with self.video_thread_lock:
                            self.latest_frame = jpeg.tobytes()
                
                # Limit playback to target FPS
                elapsed = time.time() - frame_start_time
                delay = max(1, int((1.0 / self.config.args.fps - elapsed) * 10000))
                time.sleep(delay / 10000.0)
                
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
