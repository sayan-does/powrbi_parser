import cv2
import numpy as np
import pytesseract
from typing import Dict, List, Any


def extract_visual_data(image: np.ndarray, regions: List[Dict[str, Any]], config: Dict) -> Dict[str, Any]:
    """
    Extract visual data from the image, including tables and charts.
    
    Args:
        image: Preprocessed image
        regions: Layout regions identified in the image
        config: Configuration for visual data extraction
        
    Returns:
        Dictionary with tables and charts
    """
    tables = []
    charts = []

    # Process each region based on its type
    for region in regions:
        region_type = region.get("type", "").lower()
        bbox = region.get("bbox", {})

        # Skip non-visual regions
        if region_type not in ["table", "chart"]:
            continue

        # Extract the region from the image
        x1, y1, x2, y2 = bbox.get("x1", 0), bbox.get(
            "y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)
        if x1 >= x2 or y1 >= y2:
            continue

        region_image = image[y1:y2, x1:x2]

        # Process based on region type
        if region_type == "table":
            table_data = extract_table(region_image, config.get("table", {}))
            if table_data:
                tables.append(table_data)
        elif region_type == "chart":
            chart_data = extract_chart(region_image, config.get("chart", {}))
            if chart_data:
                charts.append(chart_data)

    return {
        "tables": tables,
        "charts": charts
    }


def extract_table(image: np.ndarray, config: Dict) -> Dict[str, Any]:
    """
    Extract data from a table image.
    """
    # Enhance the table image
    _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)

    # Use Tesseract to extract table data
    custom_config = '--oem 3 --psm 6 -c preserve_interword_spaces=1'
    data = pytesseract.image_to_data(
        binary, config=custom_config, output_type=pytesseract.Output.DICT)

    # Process the extracted data into a structured table
    rows = []
    current_row = []
    last_line_num = -1

    for i, text in enumerate(data['text']):
        if not text.strip():
            continue

        line_num = data['line_num'][i]

        # If we're on a new line, start a new row
        if line_num != last_line_num:
            if current_row:
                rows.append(current_row)
                current_row = []
            last_line_num = line_num

        # Add cell to current row
        current_row.append(text)

    # Add the last row if it exists
    if current_row:
        rows.append(current_row)

    # Structure the table
    table = {
        "position": {"x1": 0, "y1": 0, "x2": image.shape[1], "y2": image.shape[0]},
        "rows": rows
    }

    # Try to extract headers from the first row
    if rows and len(rows) > 1:
        table["headers"] = rows[0]
        table["data"] = rows[1:]
    else:
        table["data"] = rows

    return table


def extract_chart(image: np.ndarray, config: Dict) -> Dict[str, Any]:
    """
    Extract data from a chart image.
    
    Note: This is a simplified implementation. In a real-world scenario,
    you would likely use specialized libraries for chart extraction.
    """
    # For this example, we'll just return a placeholder
    # In a real implementation, you would integrate with a chart extraction library
    # like WebPlotDigitizer or chartocr

    # Extract the chart type (simplified)
    chart_type = detect_chart_type(image)

    # Return basic chart information
    chart = {
        "type": chart_type,
        "position": {"x1": 0, "y1": 0, "x2": image.shape[1], "y2": image.shape[0]},
        "text": pytesseract.image_to_string(image).strip(),
        "estimated_values": []  # Would contain actual values in a real implementation
    }

    return chart


def detect_chart_type(image: np.ndarray) -> str:
    """
    Detect the type of chart in the image.
    This is a simplified implementation.
    """
    # Very basic chart type detection based on image features
    # In a real implementation, you would use a more sophisticated approach

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(
        image.shape) == 3 else image

    # Count the number of horizontal and vertical lines
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    if lines is None:
        return "unknown"

    horizontal_lines = 0
    vertical_lines = 0

    for line in lines:
        rho, theta = line[0]
        # Check if line is horizontal or vertical
        if abs(theta) < 0.1 or abs(theta - np.pi) < 0.1:
            vertical_lines += 1
        elif abs(theta - np.pi/2) < 0.1:
            horizontal_lines += 1

    # Make a simple determination based on line counts
    if vertical_lines > horizontal_lines * 2:
        return "bar_chart"
    elif horizontal_lines > vertical_lines * 2:
        return "line_chart"
    elif vertical_lines > 10 and horizontal_lines > 10:
        return "grid_chart"  # Could be a scatter plot or heatmap
    else:
        return "pie_chart"  # Default to pie chart if not many lines
