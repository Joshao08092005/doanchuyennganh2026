
import cv2
from insightface.app import FaceAnalysis

# 1. Khoi tao mo hinh AI
print("Dang khoi tao mo hinh InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("Da tai mo hinh thanh cong!")

# 2. Mo camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Khong mo duoc camera!")
    exit()

print("Camera da mo!")
print("Nhan Q de thoat.")

# 3. Nhan dien khuon mat
while True:
    ret, frame = cap.read()

    if not ret:
        print("Khong doc duoc hinh anh!")
        break

    # Phat hien khuon mat
    faces = app.get(frame)

    for face in faces:
        # Lay toa do khuon mat
        x1, y1, x2, y2 = face.bbox.astype(int)

        # Ve khung khuon mat
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Lay vector dac trung
        embedding = face.embedding

        cv2.putText(
            frame,
            f"Face - {len(embedding)} dimensions",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Hien thi so khuon mat
    cv2.putText(
        frame,
        f"So khuon mat: {len(faces)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("HE THONG NHAN DIEN KHUON MAT", frame)

    # Nhan Q de thoat
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Da dong camera!")
