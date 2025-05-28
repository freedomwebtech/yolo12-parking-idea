import cv2
from ultralytics import YOLO
import numpy as np
# Load YOLOv8 model
model = YOLO('best.pt')
names = model.names


# Open video
cap = cv2.VideoCapture("test2.mp4")


# Debug mouse position
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)


frame_count = 0
area1=[(202,317),(195,387),(1000,402),(972,310)]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020,600))
    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        listarea1=[]
        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            cx=int(x1+x2)//2
            cy=int(y1+y2)//2
            label = names[class_id]
            result=cv2.pointPolygonTest(np.array(area1,np.int32),(cx,cy),False)
            if result>=0:
               cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
               listarea1.append(cx)
           
            cv2.putText(frame, "Area1", (204, 304),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
          

    area1counter=len(listarea1)
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),2)
     
    cv2.putText(frame, f"Area1:-{area1counter}", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            

    cv2.imshow("RGB", frame)
    if cv2.waitKey(0) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
