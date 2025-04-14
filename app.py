import os
import torch
import cv2
import numpy as np
import gradio as gr
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor
from plyfile import PlyData, PlyElement

# Create directories to save results
save_dir = "depth_results"
os.makedirs(save_dir, exist_ok=True)

# Initialize status for model downloads
model_status = {
    "yolo": False,
    "depth": False
}

def download_models():
    """Download all required models before processing any images"""
    # Download YOLO model first
    try:
        from ultralytics import YOLO
        print("Downloading YOLO model...")
        # Force download by trying to load it
        yolo_model = YOLO('yolov8n.pt')  # Using the smallest model for faster download
        model_status["yolo"] = True
        print("YOLO model downloaded successfully!")
        return yolo_model
    except Exception as e:
        print(f"Error downloading YOLO model: {str(e)}")
        return None

# Load the pre-trained depth estimation model
def download_depth_model():
    print("Downloading depth estimation model...")
    try:
        model_name = "Intel/dpt-large"
        processor = DPTImageProcessor.from_pretrained(model_name)
        model = DPTForDepthEstimation.from_pretrained(model_name)
        model_status["depth"] = True
        print("Depth model downloaded successfully!")
        return processor, model
    except Exception as e:
        print(f"Error downloading depth model: {str(e)}")
        return None, None

def depth_to_point_cloud(depth_map, rgb_image, output_file="depth_results/point_cloud.ply"):
    """
    Convert depth map and RGB image to a 3D point cloud and save as PLY file
    """
    # Define camera parameters (these are approximate and would need calibration)
    # For a typical camera with horizontal field of view of about 60 degrees
    height, width = depth_map.shape
    fx = width / (2 * np.tan(np.radians(60) / 2))  # focal length x
    fy = fx  # assume square pixels
    cx = width / 2   # principal point x
    cy = height / 2  # principal point y
    
    # Create meshgrid for pixel coordinates
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Calculate 3D coordinates
    Z = depth_map  # depth
    # Scale depth values to a reasonable range (e.g., 0.1 to 10 meters)
    depth_min, depth_max = np.min(Z), np.max(Z)
    Z_scaled = 0.1 + (Z - depth_min) * (10 - 0.1) / (depth_max - depth_min)
    
    X = (x_indices - cx) * Z_scaled / fx
    Y = (y_indices - cy) * Z_scaled / fy
    
    # Reshape to 1D arrays
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z_scaled.reshape(-1)
    
    # Get RGB values
    if len(rgb_image.shape) == 3:
        rgb = rgb_image.reshape(-1, 3)
    else:
        # Handle grayscale images
        rgb = np.stack([rgb_image] * 3, axis=-1).reshape(-1, 3)
    
    # Stack coordinates and colors
    points = np.column_stack([X, Y, Z, rgb])
    
    # Remove points with invalid depth
    valid_points = ~np.isnan(points[:,2])
    points = points[valid_points]
    
    # Create PLY data structure
    vertex = np.zeros(points.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    
    vertex['x'] = points[:, 0]
    vertex['y'] = points[:, 1]
    vertex['z'] = points[:, 2]
    vertex['red'] = points[:, 3]
    vertex['green'] = points[:, 4]
    vertex['blue'] = points[:, 5]
    
    # Create PLY element
    el = PlyElement.describe(vertex, 'vertex')
    
    # Create PLY data and write to file
    PlyData([el]).write(output_file)
    
    print(f"3D point cloud saved to: {output_file}")
    return output_file

def process_image(input_image, progress=gr.Progress()):
    """Process the input image to detect objects and display depth directly on the image."""
    if input_image is None:
        return None, None
    
    progress(0.1, desc="Loading models...")
    
    # Make sure we have the YOLO model
    if not model_status["yolo"]:
        progress(0.2, desc="Downloading YOLO model...")
        yolo_model = download_models()
    else:
        # Load it again if it was already downloaded
        from ultralytics import YOLO
        yolo_model = YOLO('yolov8n.pt')
    
    # Make sure we have the depth model
    if not model_status["depth"]:
        progress(0.3, desc="Downloading depth model...")
        processor, model = download_depth_model()
    else:
        # Load it again if it was already downloaded
        model_name = "Intel/dpt-large"
        processor = DPTImageProcessor.from_pretrained(model_name)
        model = DPTForDepthEstimation.from_pretrained(model_name)
    
    # Convert to PIL image if needed
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)
    
    # Save original image
    img_filename = os.path.join(save_dir, "original.jpg")
    input_image.save(img_filename)
    
    # Convert to numpy array for YOLO processing
    np_image = np.array(input_image)
    
    progress(0.4, desc="Detecting objects...")
    
    # Object detection with YOLO
    detected_objects = []
    if yolo_model is not None:
        results = yolo_model(np_image)
        
        # Process results
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                detected_objects.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name
                })
    
    progress(0.6, desc="Calculating depth...")
    
    # Depth estimation
    # Process the image for depth estimation
    inputs = processor(images=input_image, return_tensors="pt")
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Convert to numpy
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=input_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    depth_map = prediction.cpu().numpy()
    
    # Generate 3D point cloud file
    progress(0.7, desc="Generating 3D model...")
    point_cloud_file = depth_to_point_cloud(depth_map, np_image)
    
    # Calculate depth for each detected object
    output_image = np_image.copy()
    
    # Create a text file to store object depths
    depth_info_file = os.path.join(save_dir, "object_depths.txt")
    with open(depth_info_file, "w") as f:
        f.write("Object Detection and Depth Results:\n")
        f.write("---------------------------------\n")
        f.write(f"Point cloud 3D file: {point_cloud_file}\n\n")
        
        for i, obj in enumerate(detected_objects):
            x1, y1, x2, y2 = obj["box"]
            class_name = obj["class_name"]
            confidence = obj["confidence"]
            
            # Calculate the average depth in the object's bounding box
            object_depth_region = depth_map[y1:y2, x1:x2]
            if object_depth_region.size > 0:
                avg_depth = np.mean(object_depth_region)
                # Get actual depth value (not normalized)
                depth_value = avg_depth
            else:
                depth_value = 0
            
            # Draw green bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Create black background for depth label (similar to your example)
            # Place it above the bounding box
            label = f"Depth: {depth_value:.1f}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(output_image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 0, 0), -1)
            
            # Draw white text for depth label
            cv2.putText(output_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Create second label for object class below the bounding box
            class_label = f"{class_name}: {confidence:.2f}"
            class_text_size, _ = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output_image, (x1, y2), (x1 + class_text_size[0], y2 + class_text_size[1] + 5), (0, 0, 0), -1)
            cv2.putText(output_image, class_label, (x1, y2 + class_text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write to file
            f.write(f"Object {i+1}: {class_name}\n")
            f.write(f"  Confidence: {confidence:.2f}\n")
            f.write(f"  Position: ({x1}, {y1}) to ({x2}, {y2})\n")
            f.write(f"  Depth: {depth_value:.1f}\n\n")
    
    # Save the detection result
    result_filename = os.path.join(save_dir, "result_with_depth.jpg")
    cv2.imwrite(result_filename, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    
    # Create a heatmap visualization of the depth
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_normalized = (depth_map - depth_min) / (depth_max - depth_min) * 255
    depth_heatmap = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
    depth_heatmap = cv2.cvtColor(depth_heatmap, cv2.COLOR_BGR2RGB)
    
    # Save the depth heatmap
    depth_filename = os.path.join(save_dir, "depth_heatmap.jpg")
    cv2.imwrite(depth_filename, cv2.cvtColor(depth_heatmap, cv2.COLOR_RGB2BGR))
    
    progress(1.0, desc="Processing complete!")
    return output_image, depth_heatmap

# Create the Gradio interface
with gr.Blocks(title="3D Depth Analysis System") as demo:
    gr.Markdown("# 3D Depth Analysis System")
    gr.Markdown("Upload an image to detect objects, estimate depth, and generate a 3D model.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image")
            submit_btn = gr.Button("Process Image", variant="primary")
        
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Detection Result"):
                    output_image = gr.Image(label="Objects with Depth Values")
                with gr.TabItem("Depth Visualization"):
                    depth_heatmap = gr.Image(label="Depth Heatmap")
    
    # Status message to show download progress
    status_msg = gr.Markdown("""
    **Note**: On first run, the application will download YOLO and depth estimation models.
    
    **Saved Files**:
    - **3D Point Cloud**: `depth_results/point_cloud.ply` (open with MeshLab, Blender, etc.)
    - **Depth Information**: `depth_results/object_depths.txt`
    - **Detection Image**: `depth_results/result_with_depth.jpg`
    - **Depth Heatmap**: `depth_results/depth_heatmap.jpg`
    """)
    
    submit_btn.click(
        fn=process_image,
        inputs=input_image,
        outputs=[output_image, depth_heatmap]
    )
    
    gr.Markdown("""
    ## Applications
    
    This tool can be used for:
    
    1. **Flood Damage Assessment**: Measure water depth in flooded areas
    2. **Wall Damage Analysis**: Detect and measure depth of cracks or holes
    3. **Structural Assessment**: Calculate dimensions of structural damage
    4. **Interior Mapping**: Create 3D models of room interiors
    
    ## Interpretation Guide
    
    - **Depth Values**: Relative measurements (not absolute meters/feet without calibration)
    - **Color Map**: Darker colors are closer, brighter colors are farther away
    - **3D Point Cloud**: Open the PLY file in 3D software for further analysis
    """)

# Download models immediately when script starts
if __name__ == "__main__":
    print("Initializing application and downloading models...")
    # Pre-download models before launching the interface
    try:
        yolo_model = download_models()
        processor, model = download_depth_model()
        print("All models downloaded successfully! Launching interface...")
    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        print("Will attempt to download models when processing images.")
    
    # Launch the interface
    demo.launch()