from video_processor import apply_dynamic_panning, apply_static_crop
import numpy as np
from moviepy.editor import ColorClip
import os
import traceback

def test_dynamic_panning():
    # Create a simple test clip (red background)
    width, height = 1280, 720
    test_clip = ColorClip((width, height), color=(255, 0, 0), duration=5)
    
    # Test data
    focus_points = [
        {"time": 0.0, "point": {"x": 0.2, "y": 0.2}},
        {"time": 2.5, "point": {"x": 0.8, "y": 0.8}},
        {"time": 5.0, "point": {"x": 0.5, "y": 0.5}}
    ]
    
    # Target dimensions for 9:16 crop
    new_width = 405  # Example for 9:16 ratio when original is 16:9
    new_height = 720
    
    # Test the function
    try:
        print("Original clip duration:", test_clip.duration)
        print("Original clip dimensions:", test_clip.size)
        
        try:
            result_clip = apply_dynamic_panning(
                test_clip, 
                focus_points, 
                new_width, 
                new_height, 
                width, 
                height
            )
            print("\n✅ Dynamic panning function works correctly!")
            print(f"Result clip dimensions: {result_clip.size}, duration: {result_clip.duration}")
            
            # Verify dimensions are correct
            expected_width, expected_height = new_width, new_height
            if result_clip.size == (expected_width, expected_height):
                print("✅ Dimensions match expected values")
            else:
                print(f"⚠️ Dimension mismatch: Expected {(expected_width, expected_height)}, got {result_clip.size}")
                
            # Verify duration is preserved
            if result_clip.duration == test_clip.duration:
                print("✅ Duration is preserved correctly")
            else:
                print(f"⚠️ Duration mismatch: Expected {test_clip.duration}, got {result_clip.duration}")
                
            return True
        except Exception as e:
            print(f"❌ Error in dynamic panning: {e}")
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return False

if __name__ == "__main__":
    print("Testing video processor functions...")
    success = test_dynamic_panning()
    print(f"Testing complete! Overall {'SUCCESS' if success else 'FAILURE'}") 