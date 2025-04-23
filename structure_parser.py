import cv2
import numpy as np
from typing import Dict, List, Any

def analyze_layout(image: np.ndarray, config: Dict) -> List[Dict[str, Any]]:
    """
    Analyze the layout of a dashboard image to identify different regions.
    
    Args:
        image: Preprocessed image
        config: Configuration for layout analysis
        
    Returns:
        List of regions with their bounding boxes and types
    """
    regions = []
    
    # Basic contour-based layout analysis
    # This is a simplified version - in a real-world scenario, 
    # you might use a more sophisticated approach like LayoutParser
    
    # Find contours
    _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process each contour
    for contour in contours:
        # Skip small contours
        if cv2.contourArea(contour) < config.get("min_region_area", 100):
            continue
            
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Determine region type based on aspect ratio and position
        region_type = get_region_type(x, y, w, h, image.shape, config)
        
        # Add to regions list
        regions.append({
            "type": region_type,
            "bbox": {
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h
            }
        })
    
    return regions

def get_region_type(x: int, y: int, w: int, h: int, image_shape: tuple, config: Dict) -> str:
    """
    Determine the region type based on its position and dimensions.
    
    This is a simplified heuristic approach. In a real application, 
    you might use machine learning to classify regions.
    """
    aspect_ratio = w / h
    img_height, img_width = image_shape[0], image_shape[1]
    
    # Check if it's a header (typically at the top)
    if y < img_height * config.get("header_threshold", 0.15):
        return "title"
    
    # Check if it's likely a table (usually has a certain aspect ratio)
    if 0.8 < aspect_ratio < 3.0 and w > img_width * config.get("table_width_threshold", 0.3):
        return "table"
    
    # Check if it's likely a chart (usually square-ish)
    if 0.75 < aspect_ratio < 1.5 and w > img_width * config.get("chart_width_threshold", 0.2):
        return "chart"
    
    # Default to text
    return "text"