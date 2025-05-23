import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from ultralytics import SAM

def auto_annotate_video_with_models(
    video_path,
    output_video_path,
    det_model="PekingU/rtdetr_v2_r18vd",
    sam_model_path="mobile_sam.pt",
    device="cuda",
    conf=0.55,
    visualize=True,
    use_sam=True,
    iou_threshold=0.3,  # IoU threshold for person mask classification
    use_fp16=True      # New option: use fp16 (autocast) if True
):
    """
    1. Extract frames from video (1 frame increments)
    2. Detect persons in each frame using RT-DETR v2, get coordinates
    3. If use_sam=True, apply SAM for instance segmentation on detected persons
    4. Visualize bounding boxes and/or segmentation masks on the frames
    5. Write processed frames into a new video
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load RT-DETR v2
    print(f"Loading RT-DETR v2 model: {det_model}")
    image_processor = RTDetrImageProcessor.from_pretrained(det_model)
    rtdetr_model = RTDetrV2ForObjectDetection.from_pretrained(det_model).to(device).eval()
    
    # Load SAM model if enabled
    sam_model = None
    if use_sam:
        print(f"Loading SAM model: {sam_model_path}")
        try:
            # Check if model path is just a filename (not absolute path)
            if not os.path.isabs(sam_model_path):
                # Look for the model in the current directory first
                local_model_path = os.path.join(os.path.dirname(__file__), sam_model_path)
                if os.path.exists(local_model_path):
                    sam_model_path = local_model_path
                # If not found locally and it's mobile_sam.pt, let ultralytics download it
                elif sam_model_path == "mobile_sam.pt":
                    print("mobile_sam.pt not found locally, will be downloaded by ultralytics...")
            
            # Try to load SAM model
            sam_model = SAM(sam_model_path)
            sam_model.to(device)
            print("SAM model loaded successfully!")
            sam_model.info()
        except Exception as e:
            print(f"Warning: Failed to load SAM model ({e}). Continuing with detection only...")
            use_sam = False
            sam_model = None

    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video properties: {frame_width}x{frame_height} @ {frame_rate} FPS")
    
    # Create output video writer
    out = cv2.VideoWriter(
        output_video_path, 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        frame_rate, 
        (frame_width, frame_height)
    )

    # Helper function to calculate IoU between mask and box
    def calculate_iou(mask, box):
        x1, y1, x2, y2 = map(int, box[:4])
        # Create box mask
        box_mask = np.zeros_like(mask)
        box_mask[y1:y2, x1:x2] = 1
        
        # Calculate intersection and union
        intersection = np.logical_and(mask, box_mask).sum()
        union = np.logical_or(mask, box_mask).sum()
        
        # Calculate IoU and mask coverage
        iou = intersection / union if union > 0 else 0
        box_coverage = intersection / box_mask.sum() if box_mask.sum() > 0 else 0
        mask_coverage = intersection / mask.sum() if mask.sum() > 0 else 0
        
        return iou, box_coverage, mask_coverage

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame.")
            break
        
        print(f"Processing frame {frame_idx} ...", end=' ')
        
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Detect persons with RT-DETR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        inputs = image_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = rtdetr_model(**inputs)
            
        target_sizes = torch.tensor([(pil_image.height, pil_image.width)]).to(device)
        results = image_processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=conf
        )[0]
        
        results = {k: v.cpu() for k, v in results.items()}
        
        # Process person detections
        person_boxes = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if rtdetr_model.config.id2label[label.item()].lower() == "person":
                person_boxes.append([float(i) for i in box.tolist()] + [float(score)])
                
        print(f"Detected {len(person_boxes)} person(s).", end=' ')
        
        # Apply SAM for segmentation if enabled
        if use_sam and sam_model is not None and len(person_boxes) > 0:
            try:
                # Use autocast for fp16 if enabled
                autocast_ctx = torch.cuda.amp.autocast if (use_fp16 and device.type == "cuda") else torch.cpu.amp.autocast
                with torch.amp.autocast('cuda', enabled=use_fp16):
                    sam_results = sam_model.predict(source=rgb_frame, conf=conf, show=False, save=False)
                
                # Visualize segmentation masks
                if visualize and len(sam_results) > 0:
                    result = sam_results[0]
                    if hasattr(result, 'masks') and result.masks is not None:
                        overlay = vis_frame.copy()
                        
                        # Prepare masks for people and non-people
                        person_mask = np.zeros(vis_frame.shape[:2], dtype=bool)
                        non_person_mask = np.zeros(vis_frame.shape[:2], dtype=bool)
                        
                        # Track which masks are assigned to people
                        assigned_masks = set()
                        
                        # First, find which masks correspond to people
                        for i, mask in enumerate(result.masks.data):
                            mask_np = mask.cpu().numpy().astype(bool)
                            
                            # For each person box, check if this mask has significant overlap
                            best_iou = 0
                            for box_idx, box_score in enumerate(person_boxes):
                                iou, box_coverage, mask_coverage = calculate_iou(mask_np, box_score)
                                
                                # We consider a mask to be a person if either:
                                # 1. The IoU is above threshold OR
                                # 2. The mask covers a significant portion of the box
                                if iou > best_iou:
                                    best_iou = iou
                                
                            # If this mask has good IoU with any person box, mark it as a person
                            if best_iou >= iou_threshold:
                                person_mask = np.logical_or(person_mask, mask_np)
                                assigned_masks.add(i)
                        
                        # Then classify remaining masks as non-person
                        for i, mask in enumerate(result.masks.data):
                            if i not in assigned_masks:
                                mask_np = mask.cpu().numpy().astype(bool)
                                non_person_mask = np.logical_or(non_person_mask, mask_np)
                        
                        # Color the masks
                        colored_mask = np.zeros_like(vis_frame)
                        # Only show red mask for person, do not show green background
                        colored_mask[person_mask] = (0, 0, 255)  # Red
                        # If you want to show background when no person is detected, you can add logic here
                        vis_frame = cv2.addWeighted(colored_mask, 0.5, overlay, 1, 0)
            except Exception as e:
                print(f"SAM segmentation failed: {e}. Continuing with detection only...")
                use_sam = False
        
        # Draw bounding boxes from RT-DETR (invisible but functional)
        if visualize:
            for box_score in person_boxes:
                x1, y1, x2, y2, score = map(float, box_score)
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Make bounding boxes invisible by using completely transparent color
                # This keeps the detection functional but removes visual clutter
                color = (0, 0, 0, 0)  # Completely transparent
                thickness = 0  # No thickness = invisible
                
                # Optional: If you want to keep some minimal indication, you can use:
                # color = (0, 255, 255)  # Yellow for better visibility
                # cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Remove the bounding box drawing entirely
                # cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Remove confidence score label as well to keep it clean
                # label_text = f"Person: {score:.2f}"
                # cv2.putText(
                #     vis_frame, 
                #     label_text, 
                #     (x1, y1 - 10), 
                #     cv2.FONT_HERSHEY_SIMPLEX, 
                #     0.7, 
                #     color, 
                #     2
                # )
                
                # The detection is still functional for SAM segmentation
                # Only the visual representation is removed
        
        # Write the visualized frame
        out.write(vis_frame)
        print("Frame processed and written.")
        frame_idx += 1
    
    # Release resources
    cap.release()
    out.release()
    print(f"Video processing complete. Output saved at {output_video_path}")


if __name__ == "__main__":
    auto_annotate_video_with_models(
        video_path=r"C:\Users\buanh\Documents\VSCODE\thesis\postprocess\1sclipbike.mp4", #C:\Users\buanh\Documents\VSCODE\PYTHON\FINALTHESIS\COFFEESHOP_1.mp4 # C:\Users\buanh\Documents\VSCODE\thesis\postprocess
        output_video_path=r"C:\Users\buanh\Documents\VSCODE\thesis\postprocess\predicted_output.mp4", #F:\yt-dl\COFFEESHOP\Bicycle_clip_out_fail_detected_segmented.mp4
        det_model="PekingU/rtdetr_v2_r18vd",
        sam_model_path="mobile_sam.pt",  # Using mobile_sam.pt instead of sam2.1_t.pt
        device="cuda",
        conf=0.55,
        visualize=True,
        use_sam=True,
        iou_threshold=0.3  # Adjust this value based on your specific video
    )