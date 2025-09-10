from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# Load model and run inference
model = YOLO('res/ModelAI/best.pt')
results = model('res/ImageCrop/A_20250717_094831_000.png')

# Load the original image
img_path = 'res/ImageCrop/A_20250717_094831_000.png'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Define colors for each class
class_colors = {
    'OK': (0, 255, 0),      # Green
    'NG': (255, 0, 0),      # Red
    '0': (128, 0, 128),     # Purple
    0: (128, 0, 128),       # Purple (for numeric class)
    1: (0, 255, 0),         # Green (assuming class 1 is OK)
    2: (255, 0, 0),         # Red (assuming class 2 is NG)
}

# Create a copy of the image for drawing
img_with_boxes = img_rgb.copy()

# Get image dimensions
height, width = img_rgb.shape[:2]

# Process results
result = results[0]

# Check if we have detection results (boxes) or classification results (probs)
if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
    # Detection mode - draw bounding boxes
    print("Detection mode: Drawing bounding boxes")
    
    boxes = result.boxes
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        # Get confidence and class
        conf = box.conf[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        
        # Get class name
        class_name = result.names[cls]
        
        # Get color for this class
        color = class_colors.get(class_name, class_colors.get(cls, (255, 255, 255)))
        
        # Draw bounding box
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 3)
        
        # Draw label with background
        label = f"{class_name}: {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Draw label background
        cv2.rectangle(img_with_boxes, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0] + 10, y1), 
                     color, -1)
        
        # Draw label text
        cv2.putText(img_with_boxes, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

elif hasattr(result, 'probs') and result.probs is not None:
    # Classification mode - draw border around entire image
    print("Classification mode: Drawing border around entire image")
    
    probs = result.probs
    names = result.names
    
    # Get the predicted class
    predicted_class_idx = probs.top1
    predicted_class_name = names[predicted_class_idx]
    confidence = probs.top1conf.cpu().numpy()
    
    # Get color for predicted class
    color = class_colors.get(predicted_class_name, class_colors.get(predicted_class_idx, (255, 255, 255)))
    
    # Draw border around entire image (multiple rectangles for thick border)
    border_thickness = 10
    for i in range(border_thickness):
        cv2.rectangle(img_with_boxes, 
                     (i, i), 
                     (width - 1 - i, height - 1 - i), 
                     color, 1)
    
    # Add label at top-left corner
    label = f"{predicted_class_name}: {confidence:.3f}"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    
    # Draw label background
    cv2.rectangle(img_with_boxes, 
                 (border_thickness, border_thickness), 
                 (border_thickness + label_size[0] + 20, border_thickness + label_size[1] + 20), 
                 color, -1)
    
    # Draw label text
    cv2.putText(img_with_boxes, label, 
               (border_thickness + 10, border_thickness + label_size[1] + 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Original image
axes[0].imshow(img_rgb)
axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
axes[0].axis('off')

# Image with bounding boxes/border
axes[1].imshow(img_with_boxes)
axes[1].set_title('YOLO Results with Colored Boxes/Border', fontsize=14, fontweight='bold')
axes[1].axis('off')

# Add color legend
legend_elements = []
for class_name, color in class_colors.items():
    if isinstance(class_name, str):  # Only show string class names in legend
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=np.array(color)/255, 
                                           label=class_name))

if legend_elements:
    axes[1].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()

# Print detailed results
print("="*50)
print("YOLO RESULTS WITH COLORED VISUALIZATION")
print("="*50)
print(f"Image: {img_path}")
print(f"Image shape: {img_rgb.shape}")
print()

# Color mapping info
print("Color Mapping:")
print("-" * 20)
print("OK  -> Green")
print("NG  -> Red") 
print("0   -> Purple")
print()

if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
    print("Detection Results:")
    print("-" * 20)
    boxes = result.boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        class_name = result.names[cls]
        
        print(f"Box {i+1}: {class_name} ({conf:.3f}) at [{x1}, {y1}, {x2}, {y2}]")

elif hasattr(result, 'probs') and result.probs is not None:
    print("Classification Results:")
    print("-" * 20)
    probs = result.probs
    names = result.names
    
    # Get all probabilities
    all_probs = probs.data.cpu().numpy()
    
    # Sort by probability
    sorted_indices = np.argsort(all_probs)[::-1]
    
    for i, idx in enumerate(sorted_indices):
        class_name = names[idx]
        prob = all_probs[idx]
        print(f"{i+1:2d}. {class_name:10s}: {prob:.4f} ({prob*100:.1f}%)")
    
    print(f"\nBorder color: {names[probs.top1]} (confidence: {probs.top1conf:.3f})")

else:
    print("No detection boxes or classification probabilities found")

# Save the result image (optional)
result_img_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
output_path = 'res/result_with_boxes.jpg'
cv2.imwrite(output_path, result_img_bgr)
print(f"\nResult image saved to: {output_path}")