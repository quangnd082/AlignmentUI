import cv2
import numpy as np
from ultralytics import YOLO

def detect_and_draw_centers(image, model_seg):
    
    results = model_seg.predict(image, verbose=False)
    output_image = image.copy()

    for r in results:
        masks = r.masks
        boxes = r.boxes

        # Trường hợp có mask
        if masks is not None:
            for mask in masks.data.cpu().numpy():
                mask_img = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 2)

                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(output_image, (cx, cy), 5, (0, 0, 255), -1)

        # Trường hợp không có mask → vẽ bbox
        elif boxes is not None:
            for box in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Tính tâm bbox
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.circle(output_image, (cx, cy), 5, (0, 0, 255), -1)

    return output_image

if __name__ == "__main__":
    # Load model segmentation (YOLOv8)
    model = YOLO("res/ModelAI/A366_nho_2.pt")
    
    # Đọc ảnh
    img = cv2.imread(r"C:\Users\DTC\Desktop\ImageOK\PASS_2025_08_01_09_47_44.jpg")

    # Gọi hàm detect
    result_img = detect_and_draw_centers(img, model)

    # Hiển thị kết quả
    
    cv2.namedWindow("Detected Image", cv2.WINDOW_FREERATIO)
    cv2.imshow("Detected Image", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
