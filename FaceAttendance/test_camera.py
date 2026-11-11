
import cv2

# Bước 1: Kết nối với webcam
camera = cv2.VideoCapture(0)

# Bước 2: Kiểm tra camera
if not camera.isOpened():
    print("Khong the mo camera!")
    exit()

print("Camera da ket noi thanh cong!")

# Bước 3: Hiển thị hình ảnh camera
while True:

    ret, frame = camera.read()

    if not ret:
        print("Khong doc duoc hinh anh!")
        break

    # Hiển thị camera
    cv2.imshow("HE THONG DIEM DANH KHUON MAT", frame)

    # Nhấn Q để thoát chương trình
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bước 4: Đóng camera
camera.release()
cv2.destroyAllWindows()

print("Da dong camera!")
