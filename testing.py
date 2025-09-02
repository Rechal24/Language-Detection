from ultralytics import YOLO
model = YOLO("best.pt")
model.predict(mode="predict", model="best.pt", show=True, conf=0.1, save=True, source="english(20).jpg")