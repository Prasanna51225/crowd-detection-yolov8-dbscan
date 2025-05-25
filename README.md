# Crowd-detection-yolov8-dbscan
A high-precision crowd detection pipeline that combines YOLOv8 (object detection), DeepSORT (multi-object tracking), DBSCAN (density-based clustering), and temporal filtering to identify and track persistent groups of 3+ people across video frames. Includes annotated output videos and heatmap visualizations for crowd analysis.
This project implements a temporal crowd detection pipeline that identifies groups of people in a video who stay close together for a significant duration. It uses state-of-the-art computer vision techniques to detect, track, and analyze crowd behavior in real-time.

## Project Objective

Detect and annotate groups of three or more people who remain clustered for at least ten consecutive frames. This is useful for applications in surveillance, crowd monitoring, and public safety.

## Components Used

- YOLOv8 for real-time person detection
- DeepSORT for multi-object tracking with ID consistency
- DBSCAN for density-based spatial clustering
- Temporal filtering logic to ensure group persistence over time
- Matplotlib and Seaborn for heatmap visualization
- OpenCV for video frame manipulation and annotation

## Key Features

- Accurate detection of individuals in each video frame
- Consistent tracking of individuals across frames using DeepSORT
- Identification of crowded groups using DBSCAN
- Temporal filtering to exclude short-term or accidental gatherings
- Annotated video output with bounding boxes and group IDs
- Heatmap generation to visualize crowd density and movement

## How It Works

1. Detect people in each frame using YOLOv8
2. Track them over time with DeepSORT
3. Cluster tracked individuals using DBSCAN based on spatial proximity
4. Apply temporal filtering to retain only stable groups
5. Save and visualize the output with bounding boxes, group IDs, and heatmaps

## Output

- Video with annotated bounding boxes and group cluster IDs
- Optional heatmap showing crowd intensity across the frame
- Outputs saved to local directory or Google Drive (for Colab users)

## Use Cases

- Public space monitoring
- Event crowd analysis
- Security surveillance
- Smart city analytics

## Installation and Setup

1. Clone this repository
2. Install required dependencies from `Requirements.txt`
3. Run the pipeline using your input video

```bash
Requirements.txt
code.py 
