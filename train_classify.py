from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

model.train(data=r'C:\Users\DTC\Desktop\ImageFBCP\SubFPCP-Almus-Classify.v5i.folder', epochs=15)