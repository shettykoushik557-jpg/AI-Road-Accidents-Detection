from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
video = cv2.VideoCapture("videos/4686100-uhd_3840_2160_24fps.mp4")

while True:
    success, frame = video.read()
    if not success:
        break

    output = model(frame)

    result_frame = output[0].plot()
    small_frame=cv2.resize(result_frame, (800,450))
    accident = True

    if accident:
        cv2.putText(
            small_frame,
            "ACCIDENT DETECTED",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("Vehicle Detection", small_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()