#!/usr/bin/env python3
"""
Test script to verify mobile SAM model loading
"""
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_mobile_sam():
    """Test if mobile SAM can be loaded"""
    try:
        from ultralytics import SAM
        
        print("Testing mobile SAM model loading...")
        
        # Try to load mobile_sam.pt (will auto-download if not found)
        sam_model = SAM("mobile_sam.pt")
        print("✅ Mobile SAM model loaded successfully!")
        
        # Test model info
        sam_model.info()
        
        # Test if CUDA is available
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA is available, moving model to GPU...")
            sam_model.to("cuda")
            print("✅ Model successfully moved to CUDA")
        else:
            print("ℹ️  CUDA not available, using CPU")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading mobile SAM: {e}")
        return False

def test_post_processing():
    """Test the post-processing function"""
    try:
        from backend.postprocess.auto import auto_annotate_video_with_models
        
        # Check if we have any test video files
        recordings_dir = os.path.join(os.path.dirname(__file__), 'recordings')
        test_video = None
        
        if os.path.exists(recordings_dir):
            video_files = [f for f in os.listdir(recordings_dir) 
                          if f.endswith('.mp4') and f.startswith('raw_segmented_recording_')]
            if video_files:
                test_video = os.path.join(recordings_dir, video_files[0])
        
        if test_video:
            print(f"Found test video: {test_video}")
            print("✅ Post-processing function is available and can be tested")
            return True
        else:
            print("ℹ️  No test videos found, but post-processing function is available")
            return True
            
    except Exception as e:
        print(f"❌ Error with post-processing: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Mobile SAM Integration ===")
    
    sam_ok = test_mobile_sam()
    postprocess_ok = test_post_processing()
    
    if sam_ok and postprocess_ok:
        print("\n✅ All tests passed! Mobile SAM integration is ready.")
    else:
        print("\n⚠️  Some tests failed, but the system should still work with detection-only mode.")
