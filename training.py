from ultralytics import YOLO
model=YOLO("yolo11m.pt")
model.train(task="detect", mode='train', epochs=15, data='data_custom.yaml', model='yolo11m.pt', imgsz=640, batch=8)