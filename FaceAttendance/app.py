import cv2
import numpy as np
import time
import threading
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, send_file
import base64
import pandas as pd
import io
from insightface.app import FaceAnalysis
from database import connect_db

# Khởi tạo ứng dụng Web Flask
app = Flask(__name__)

# Cấu hình AI
MODEL_NAME = "buffalo_s"
MIN_SIMILARITY = 0.55  # Khuôn mặt phải giống ít nhất 55% mới được tính là thành công
COOLDOWN_TIME = 60     # Phải cách nhau 60 giây giữa 2 lần điểm danh để chống spam

print("Dang khoi tao model AI (Che do sieu nhe cho CPU)...")
# CHỈ nạp module phát hiện (detection) và nhận diện (recognition) để giảm 50% tải cho CPU
face_app = FaceAnalysis(
    name=MODEL_NAME,
    allowed_modules=['detection', 'recognition'],
    providers=['CPUExecutionProvider']
)
face_app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.5)

# Từ điển lưu thời gian để tính toán chống spam
last_checkin_dict = {}

# Các biến toàn cục phục vụ tách luồng Camera và AI chống giật lag
latest_frame = None
cached_results = []
ai_lock = threading.Lock()
is_camera_running = False

def load_registered_faces():
    """Tải toàn bộ vector khuôn mặt từ CSDL lên RAM"""
    conn = connect_db()
    faces_db = []
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT f.StudentID, s.StudentCode, u.FullName, f.FaceEmbedding 
            FROM dbo.FaceData f
            JOIN dbo.Students s ON f.StudentID = s.StudentID
            JOIN dbo.Users u ON s.UserID = u.UserID
            WHERE f.IsActive = 1
        """)
        for row in cursor.fetchall():
            faces_db.append({
                "student_id": row[0],
                "student_code": row[1],
                "full_name": row[2],
                "embedding": np.frombuffer(row[3], dtype=np.float32)
            })
        return faces_db
    finally:
        conn.close()

# Tải sẵn dữ liệu khuôn mặt lên RAM khi bật server
registered_faces = load_registered_faces()

def mark_attendance(student_id):
    """Ghi nhận điểm danh vào bảng dbo.Attendance (Tự động xử lý SessionID chống lỗi Khóa Ngoại)"""
    conn = connect_db()
    try:
        cursor = conn.cursor()
        # Kiểm tra xem hôm nay sinh viên này đã được điểm danh chưa
        cursor.execute("""
            SELECT COUNT(*) FROM dbo.Attendance 
            WHERE StudentID = ? AND CAST(CheckInTime AS DATE) = CAST(GETDATE() AS DATE)
        """, student_id)
        count = cursor.fetchone()[0]

        # Nếu hôm nay chưa điểm danh thì mới INSERT vào CSDL
        if count == 0:
            cursor.execute("SELECT TOP 1 SessionID FROM dbo.AttendanceSessions")
            session_row = cursor.fetchone()

            if session_row:
                session_id = session_row[0]
                cursor.execute("""
                    INSERT INTO dbo.Attendance (StudentID, SessionID, Status, CheckInTime)
                    VALUES (?, ?, ?, ?)
                """, student_id, session_id, 'Present', datetime.now())
            else:
                cursor.execute("""
                    INSERT INTO dbo.Attendance (StudentID, Status, CheckInTime)
                    VALUES (?, ?, ?)
                """, student_id, 'Present', datetime.now())

            conn.commit()
            print(f"✅ Đã điểm danh thành công cho StudentID: {student_id}")
    except Exception as e:
        print(f"⚠️ Lỗi ghi Database: {e}")
    finally:
        conn.close()

def ai_worker():
    """Luồng AI chạy ngầm độc lập - Không bao giờ làm giật lag luồng Video"""
    global latest_frame, cached_results, is_camera_running

    while is_camera_running:
        frame_to_process = None
        with ai_lock:
            if latest_frame is not None:
                frame_to_process = latest_frame.copy()

        if frame_to_process is None:
            time.sleep(0.05)
            continue

        try:
            # Thu nhỏ ảnh 50% trước khi đưa vào AI để tăng tốc tối đa
            small_frame = cv2.resize(frame_to_process, (0, 0), fx=0.5, fy=0.5)
            faces = face_app.get(small_frame)
            new_results = []

            for face in faces:
                # Nhân đôi lại tọa độ bbox vì ảnh đã thu nhỏ 0.5
                bbox = (face.bbox * 2).astype(int)
                x1, y1, x2, y2 = bbox
                embedding_cam = np.asarray(face.embedding, dtype=np.float32)

                best_match = None
                max_similarity = -1

                for db_face in registered_faces:
                    dot_product = np.dot(embedding_cam, db_face["embedding"])
                    norm1 = np.linalg.norm(embedding_cam)
                    norm2 = np.linalg.norm(db_face["embedding"])
                    sim = dot_product / (norm1 * norm2)

                    if sim > max_similarity:
                        max_similarity = sim
                        best_match = db_face

                if max_similarity >= MIN_SIMILARITY:
                    student_id = best_match["student_id"]
                    student_code = best_match["student_code"]
                    name = best_match["full_name"]

                    label_text = f"{student_code} - {name}"
                    new_results.append((x1, y1, x2, y2, label_text, (0, 255, 0)))

                    current_time = time.time()
                    if current_time - last_checkin_dict.get(student_id, 0) > COOLDOWN_TIME:
                        last_checkin_dict[student_id] = current_time
                        threading.Thread(target=mark_attendance, args=(student_id,), daemon=True).start()
                else:
                    new_results.append((x1, y1, x2, y2, "Unknown", (0, 0, 255)))

            cached_results = new_results
        except Exception as ai_err:
            print(f"⚠️ Lỗi xử lý AI: {ai_err}")

        # Nghỉ 0.1 giây giữa các lần quét để giảm tải CPU
        time.sleep(0.1)

def generate_frames():
    """Luồng đọc Camera riêng biệt - Đẩy hình lên Web mượt 30 FPS"""
    global latest_frame, cached_results, is_camera_running

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Không thể mở camera từ Flask!")
            return

    print("✅ Đã mở camera thành công (Chế độ Tách Luồng Đa Nhiệm - Không Lag)!")
    is_camera_running = True
    cached_results = []

    # Bật luồng AI chạy song song ở chế độ nền
    threading.Thread(target=ai_worker, daemon=True).start()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                time.sleep(0.02)
                continue

            # Cập nhật khung hình mới nhất cho luồng AI xử lý
            with ai_lock:
                latest_frame = frame

            # Vẽ khung lên hình dựa trên kết quả mới nhất của luồng AI
            for (x1, y1, x2, y2, label, color) in cached_results:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Nén JPG chất lượng 70% để truyền qua trình duyệt siêu nhanh
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    except GeneratorExit:
        print("Đã ngắt kết nối luồng Camera từ web.")
    except Exception as e:
        print(f"❌ Lỗi vòng lặp camera: {e}")
    finally:
        is_camera_running = False
        cap.release()
        print("🔒 Đã giải phóng Camera và dừng luồng AI an toàn.")

# ==========================================
# CÁC API ROUTING CHO WEB
# ==========================================

# 1. Trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# 2. Luồng Video cho thẻ <img> trên Web
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 3. API Lấy danh sách toàn bộ sinh viên kèm trạng thái (Đã có mặt / Chưa có mặt)
@app.route('/api/attendance_status')
def attendance_status():
    conn = connect_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.StudentID, s.StudentCode, u.FullName,
                   (SELECT TOP 1 a.CheckInTime 
                    FROM dbo.Attendance a 
                    WHERE a.StudentID = s.StudentID 
                      AND CAST(a.CheckInTime AS DATE) = CAST(GETDATE() AS DATE)
                    ORDER BY a.CheckInTime ASC) as CheckInTime
            FROM dbo.Students s
            JOIN dbo.Users u ON s.UserID = u.UserID
            ORDER BY s.StudentCode ASC
        """)
        
        students_list = []
        for row in cursor.fetchall():
            student_id, student_code, full_name, check_in_time = row
            is_present = False
            status = "Chưa có mặt"
            time_str = "--:--:--"
            
            if check_in_time is not None:
                is_present = True
                status = "Đã có mặt"
                time_str = check_in_time.strftime("%H:%M:%S")
                
            students_list.append({
                "student_id": student_id,
                "student_code": student_code,
                "full_name": full_name,
                "is_present": is_present,
                "status": status,
                "check_in_time": time_str
            })
            
        return jsonify(students_list)
    finally:
        conn.close()

# 4. API lịch sử gần nhất
@app.route('/api/recent_attendance')
def recent_attendance():
    conn = connect_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT TOP 10 s.StudentCode, u.FullName, a.CheckInTime 
            FROM dbo.Attendance a
            JOIN dbo.Students s ON a.StudentID = s.StudentID
            JOIN dbo.Users u ON s.UserID = u.UserID
            ORDER BY a.CheckInTime DESC
        """)
        records = []
        for row in cursor.fetchall():
            if row[2] is not None:
                records.append({
                    "student_code": row[0],
                    "full_name": row[1],
                    "check_in": row[2].strftime("%H:%M:%S") 
                })
        return jsonify(records)
    finally:
        conn.close()

# 5. API XỬ LÝ ĐIỂM DANH BẰNG ẢNH TẬP THỂ
@app.route('/api/attendance_by_image', methods=['POST'])
def attendance_by_image():
    if 'image' not in request.files:
        return jsonify({"error": "Không có ảnh"}), 400
    
    file = request.files['image']
    in_memory_file = file.read()
    nparr = np.frombuffer(in_memory_file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    faces = face_app.get(img)
    recognized_students = []

    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        embedding_cam = np.asarray(face.embedding, dtype=np.float32)

        best_match = None
        max_similarity = -1

        for db_face in registered_faces:
            dot_product = np.dot(embedding_cam, db_face["embedding"])
            norm1 = np.linalg.norm(embedding_cam)
            norm2 = np.linalg.norm(db_face["embedding"])
            sim = dot_product / (norm1 * norm2)

            if sim > max_similarity:
                max_similarity = sim
                best_match = db_face

        if max_similarity >= MIN_SIMILARITY:
            student_id = best_match["student_id"]
            student_code = best_match["student_code"]
            name = best_match["full_name"]
            
            mark_attendance(student_id)
            recognized_students.append(f"{student_code} - {name}")

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, f"{student_code} - {name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "image_base64": f"data:image/jpeg;base64,{img_base64}",
        "face_count": len(faces),
        "recognized": recognized_students
    })

# 6. API Điểm danh thủ công
@app.route('/api/manual_checkin', methods=['POST'])
def manual_checkin():
    data = request.json
    student_id = data.get('student_id')
    
    if not student_id:
        return jsonify({"error": "Thiếu mã sinh viên"}), 400
        
    mark_attendance(student_id)
    return jsonify({"success": True, "message": "Điểm danh thủ công thành công!"})

# 7. API Làm mới danh sách khuôn mặt trên RAM
@app.route('/api/reload_faces', methods=['POST'])
def reload_faces():
    global registered_faces
    registered_faces = load_registered_faces()
    return jsonify({"success": True, "total_faces": len(registered_faces)})

# 8. API Xuất file Excel danh sách điểm danh trong ngày
@app.route('/api/export_excel')
def export_excel():
    conn = connect_db()
    try:
        query = """
            SELECT s.StudentCode AS [Mã Sinh Viên], 
                   u.FullName AS [Họ và Tên],
                   ISNULL(FORMAT(a.CheckInTime, 'HH:mm:ss'), '--:--:--') AS [Thời Gian Điểm Danh],
                   CASE WHEN a.CheckInTime IS NOT NULL THEN N'Đã có mặt' ELSE N'Chưa có mặt' END AS [Trạng Thái]
            FROM dbo.Students s
            JOIN dbo.Users u ON s.UserID = u.UserID
            OUTER APPLY (
                SELECT TOP 1 CheckInTime, Status 
                FROM dbo.Attendance att 
                WHERE att.StudentID = s.StudentID 
                  AND CAST(att.CheckInTime AS DATE) = CAST(GETDATE() AS DATE)
                ORDER BY att.CheckInTime ASC
            ) a
            ORDER BY s.StudentCode ASC
        """
        df = pd.read_sql(query, conn)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='BaoCaoDiemDanh')
        output.seek(0)
        
        filename = f"BaoCaoDiemDanh_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
        return send_file(output, as_attachment=True, download_name=filename, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    finally:
        conn.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)