import cv2
import numpy as np

def extract_polygons(prob_map, threshold=0.5, hazard_type="unknown"):
    """
    Convert a probability map to GeoJSON-like polygons.
    """
    mask = (prob_map >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = []
    for cnt in contours:
        # Simplify contour
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) >= 3:
            # Convert to [lon, lat] - assuming dummy coordinates for now
            # In a real app, map grid (x,y) to (lon, lat) using bounding box
            coords = []
            for point in approx:
                x, y = point[0]
                lon = 75.0 + (x / 256.0) * 2.0
                lat = 20.0 - (y / 256.0) * 2.0
                coords.append([lon, lat])
                
            # Close the polygon
            coords.append(coords[0])
            
            # Find peak prob in this contour
            # Create a mask for just this contour
            c_mask = np.zeros_like(prob_map, dtype=np.uint8)
            cv2.drawContours(c_mask, [cnt], 0, 255, -1)
            peak_prob = float(np.max(prob_map[c_mask == 255]))
            
            severity = "High" if peak_prob > 0.8 else "Moderate"
            if hazard_type == "Cloudburst":
                severity = "Extreme"
                
            features.append({
                "type": "Feature",
                "properties": {
                    "hazard": hazard_type,
                    "probability": peak_prob,
                    "severity": severity
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                }
            })
            
    return features
