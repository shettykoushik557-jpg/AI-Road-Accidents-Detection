from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

os.makedirs("accidents", exist_ok=True)

video = cv2.VideoCapture("videos/4686100-uhd_3840_2160_24fps.mp4")
vehicle_classes = [2, 3, 5, 7]   # Car, Motorcycle, Bus, Truck

image_number = 1

while True:

    success, frame = video.read()

    if not success:
        break

    results = model(frame)

    display_frame = frame.copy()

    vehicles = []
    
    for box in results[0].boxes:

        class_id = int(box.cls[0])

        if class_id in vehicle_classes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            vehicles.append((x1, y1, x2, y2))

            label = model.names[class_id]

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                display_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    accident = False

    for i in range(len(vehicles)):

        for j in range(i + 1, len(vehicles)):

            x1, y1, x2, y2 = vehicles[i]
            a1, b1, a2, b2 = vehicles[j]

            if x2 > a1 and a2 > x1 and y2 > b1 and b2 > y1:
                accident = True

    if accident:

        cv2.putText(
            display_frame,
            "ACCIDENT DETECTED",
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        filename = f"accidents/accident_{image_number}.jpg"

        cv2.imwrite(filename, display_frame)

        image_number += 1

    display_frame = cv2.resize(display_frame, (900, 500))

    cv2.imshow("AI Road Accident Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
