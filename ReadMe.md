# 3D Depth Analysis System

A powerful application that combines object detection with depth estimation to analyze images and generate 3D models. This system can be used for damage assessment, interior mapping, structural analysis, and other applications requiring depth perception.

## Features

- **Object Detection**: Uses YOLOv8 to identify and classify objects in images
- **Depth Estimation**: Calculates relative depth for all pixels in the image
- **3D Point Cloud Generation**: Creates a 3D model that can be opened in any 3D software
- **Depth Visualization**: Provides a color-coded heat map of depth values
- **Detailed Analytics**: Generates a text report with depth measurements for each detected object

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU recommended (but not required)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/depth-analysis-system.git
   cd depth-analysis-system
   ```

2. Install the required dependencies:
   ```
   pip install torch torchvision transformers ultralytics gradio opencv-python pillow numpy plyfile
   ```

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. The Gradio interface will launch in your browser (typically at http://127.0.0.1:7860)

3. Upload an image and click "Process Image"

4. View the results in the tabs:
   - "Detection Result" shows objects with depth values
   - "Depth Visualization" shows the depth heatmap

5. Check the `depth_results` folder for output files:
   - `point_cloud.ply`: 3D model (open with MeshLab, Blender, etc.)
   - `object_depths.txt`: Detailed depth information for each object
   - `result_with_depth.jpg`: Image with object detection and depth values
   - `depth_heatmap.jpg`: Color-coded visualization of depth

## Applications

### Flood Damage Assessment
Measure water depth in flooded areas by analyzing the relative depth values of water surfaces compared to reference objects.

### Structural Analysis
Detect and measure the depth of cracks, holes, or other damage in walls and structures. The 3D model provides a detailed representation for damage assessment.

### Interior Mapping
Generate 3D models of room interiors that can be used for renovation planning, virtual tours, or space assessment.

### Object Distance Measurement
Calculate relative distances between objects in a scene, which is useful for spatial planning and organization.

## Understanding Results

- **Depth Values**: The numbers represent relative depth values, not absolute measurements in feet or meters. For real-world units, calibration is required.

- **Color Map**: In the depth visualization, darker colors represent closer objects, while brighter colors represent objects farther away.

- **3D Point Cloud**: The PLY file contains XYZ coordinates and RGB color information for each point. It can be opened in 3D software for further analysis.

## Customization

### Calibration for Real-World Measurements

To convert relative depth values to actual distances (meters/feet):

1. Include objects of known size and distance in your image
2. Calculate a scaling factor based on these references
3. Adjust the camera parameters in the `depth_to_point_cloud` function

### Using Different Models

The system uses YOLOv8n by default, but you can use larger models for better accuracy:

```python
yolo_model = YOLO('yolov8m.pt')  # medium model
yolo_model = YOLO('yolov8l.pt')  # large model
yolo_model = YOLO('yolov8x.pt')  # extra large model
```

For the depth model, you can also try other variants:
```python
model_name = "Intel/dpt-hybrid-midas"  # Faster but less accurate
```

## Limitations

- Depth estimation is based on monocular vision and provides relative values, not absolute measurements
- Accuracy depends on image quality, lighting conditions, and scene complexity
- The 3D model is a point cloud representation, not a fully textured mesh
- Processing large images may require significant computational resources

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Intel's DPT (Dense Prediction Transformer) for depth estimation
- Ultralytics for the YOLOv8 object detection model
- Gradio for the web interface
