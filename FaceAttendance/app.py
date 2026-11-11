import cv2
import numpy as np
import time
import threading
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, send_file, session, redirect, url_for
import base64
import pandas as pd
import io
from insightface.app import FaceAnalysis
from database import connect_db

app = Flask(__name__)
app.secret_key = "face_attendance_vinhuni_secret_key"

# Cấu hình AI
MODEL_NAME = "buffalo_s"
MIN_SIMILARITY = 0.55
COOLDOWN_TIME = 15

print("Dang khoi tao model AI (Che do sieu nhe cho CPU)...")
face_app = FaceAnalysis(
    name=MODEL_NAME,
    allowed_modules=['detection', 'recognition'],
    providers=['CPUExecutionProvider']
)
face_app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.5)

last_checkin_dict = {}

# Biến toàn cục điều khiển Camera, AI và Lớp học phần (ClassSubjectID) đang chọn
latest_frame = None
cached_results = []
ai_lock = threading.Lock()
is_camera_running = False
active_class_subject_id = None
enrolled_student_ids = set()

def load_registered_faces():
    """Tải toàn bộ vector khuôn mặt từ dbo.FaceData lên RAM"""
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

registered_faces = load_registered_faces()

def load_enrolled_students(class_subject_id):
    """Lấy danh sách StudentID từ bảng chuẩn dbo.Enrollments"""
    conn = connect_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT StudentID FROM dbo.Enrollments 
            WHERE ClassSubjectID = ?
        """, class_subject_id)
        return set(row[0] for row in cursor.fetchall())
    finally:
        conn.close()

def get_or_create_today_session(cursor, class_subject_id):
    """Tự động lấy hoặc tạo buổi điểm danh hôm nay trong bảng dbo.AttendanceSessions"""
    cursor.execute("""
        SELECT TOP 1 SessionID FROM dbo.AttendanceSessions
        WHERE ClassSubjectID = ? AND SessionDate = CAST(GETDATE() AS DATE)
        ORDER BY SessionID DESC
    """, class_subject_id)
    row = cursor.fetchone()
    if row:
        return row[0]

    # Tìm TeacherID phụ trách ClassSubjectID này
    cursor.execute("SELECT TeacherID FROM dbo.ClassSubjects WHERE ClassSubjectID = ?", class_subject_id)
    t_row = cursor.fetchone()
    teacher_id = t_row[0] if t_row else 1

    now = datetime.now()
    session_name = f"Điểm danh ngày {now.strftime('%d/%m/%Y')}"
    cursor.execute("""
        INSERT INTO dbo.AttendanceSessions 
            (ClassSubjectID, TeacherID, SessionName, SessionDate, StartTime, Status, CreatedAt)
        VALUES (?, ?, ?, CAST(GETDATE() AS DATE), CAST(GETDATE() AS TIME), 'Open', SYSDATETIME())
    """, class_subject_id, teacher_id, session_name)
    cursor.execute("SELECT @@IDENTITY")
    return int(cursor.fetchone()[0])

def mark_attendance(student_id, class_subject_id, similarity=0.85):
    """Ghi nhận điểm danh chuẩn vào dbo.Attendance thông qua dbo.AttendanceSessions"""
    if not class_subject_id:
        return
    conn = connect_db()
    try:
        cursor = conn.cursor()
        session_id = get_or_create_today_session(cursor, class_subject_id)

        # Kiểm tra xem sinh viên đã điểm danh trong buổi học (SessionID) này chưa
        cursor.execute("""
            SELECT COUNT(*) FROM dbo.Attendance 
            WHERE SessionID = ? AND StudentID = ?
        """, session_id, student_id)
        count = cursor.fetchone()[0]

        if count == 0:
            cursor.execute("""
                INSERT INTO dbo.Attendance (SessionID, StudentID, CheckInTime, Status, Method, Similarity)
                VALUES (?, ?, SYSDATETIME(), 'Present', 'FaceAI', ?)
            """, session_id, student_id, float(similarity))
            conn.commit()
            print(f"✅ Đã lưu điểm danh vào SQL: StudentID={student_id}, SessionID={session_id} (ClassSubjectID={class_subject_id})")
        else:
            conn.commit()
    except Exception as e:
        print(f"⚠️ Lỗi ghi Database: {e}")
    finally:
        conn.close()

def ai_worker():
    """Luồng AI chạy ngầm độc lập"""
    global latest_frame, cached_results, is_camera_running, active_class_subject_id, enrolled_student_ids

    while is_camera_running:
        frame_to_process = None
        with ai_lock:
            if latest_frame is not None:
                frame_to_process = latest_frame.copy()

        if frame_to_process is None:
            time.sleep(0.05)
            continue

        try:
            small_frame = cv2.resize(frame_to_process, (0, 0), fx=0.5, fy=0.5)
            faces = face_app.get(small_frame)
            new_results = []

            for face in faces:
                bbox = (face.bbox * 2).astype(int)
                x1, y1, x2, y2 = bbox
                embedding_cam = np.asarray(face.embedding, dtype=np.float32)

                best_match = None
                max_similarity = -1

                for db_face in registered_faces:
                    sim = np.dot(embedding_cam, db_face["embedding"]) / (np.linalg.norm(embedding_cam) * np.linalg.norm(db_face["embedding"]))
                    if sim > max_similarity:
                        max_similarity = sim
                        best_match = db_face

                if max_similarity >= MIN_SIMILARITY:
                    student_id = best_match["student_id"]
                    student_code = best_match["student_code"]
                    name = best_match["full_name"]

                    # Kiểm tra sinh viên có nằm trong danh sách dbo.Enrollments của lớp đang chọn không
                    if active_class_subject_id and student_id in enrolled_student_ids:
                        label_text = f"{student_code} - {name}"
                        new_results.append((x1, y1, x2, y2, label_text, (0, 255, 0)))

                        current_time = time.time()
                        checkin_key = (active_class_subject_id, student_id)
                        if current_time - last_checkin_dict.get(checkin_key, 0) > COOLDOWN_TIME:
                            last_checkin_dict[checkin_key] = current_time
                            threading.Thread(
                                target=mark_attendance,
                                args=(student_id, active_class_subject_id, max_similarity),
                                daemon=True
                            ).start()
                    else:
                        label_text = f"{student_code} (Khong thuoc lop)"
                        new_results.append((x1, y1, x2, y2, label_text, (0, 165, 255)))
                else:
                    new_results.append((x1, y1, x2, y2, "Unknown", (0, 0, 255)))

            cached_results = new_results
        except Exception as ai_err:
            print(f"⚠️ Lỗi xử lý AI: {ai_err}")

        time.sleep(0.1)

def generate_frames():
    """Luồng đọc Camera mượt 30 FPS"""
    global latest_frame, cached_results, is_camera_running

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return

    is_camera_running = True
    cached_results = []
    threading.Thread(target=ai_worker, daemon=True).start()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                time.sleep(0.02)
                continue

            with ai_lock:
                latest_frame = frame

            for (x1, y1, x2, y2, label, color) in cached_results:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except GeneratorExit:
        pass
    finally:
        is_camera_running = False
        cap.release()

# ==========================================
# CÁC API ĐĂNG NHẬP & QUẢN LÝ LỚP HỌC PHẦN (DÙNG BẢNG GỐC SQL)
# ==========================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        conn = connect_db()
        try:
            cursor = conn.cursor()
            # Kiểm tra đăng nhập liên kết giữa bảng Users và bảng Teachers
            cursor.execute("""
                SELECT u.UserID, u.FullName, t.TeacherID, t.Department
                FROM dbo.Users u
                JOIN dbo.Teachers t ON u.UserID = t.UserID
                WHERE u.Username = ? AND u.PasswordHash = ? AND u.IsActive = 1
            """, username, password)
            teacher = cursor.fetchone()
            if teacher:
                session['user_id'] = teacher[0]
                session['full_name'] = teacher[1]
                session['teacher_id'] = teacher[2]
                session['department'] = teacher[3]
                return redirect(url_for('index'))
            else:
                return render_template('login.html', error="Sai tài khoản hoặc mật khẩu giảng viên!")
        finally:
            conn.close()

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'teacher_id' not in session:
        return redirect(url_for('login'))

    conn = connect_db()
    try:
        cursor = conn.cursor()
        # Lấy danh sách Lớp học phần của Giảng viên đang đăng nhập từ dbo.ClassSubjects
        cursor.execute("""
            SELECT cs.ClassSubjectID, sub.SubjectName, cls.ClassName, cs.Semester, cs.AcademicYear
            FROM dbo.ClassSubjects cs
            JOIN dbo.Subjects sub ON cs.SubjectID = sub.SubjectID
            JOIN dbo.Classes cls ON cs.ClassID = cls.ClassID
            WHERE cs.TeacherID = ?
            ORDER BY cs.ClassSubjectID ASC
        """, session['teacher_id'])
        courses = [
            {
                "course_id": r[0],
                "subject_name": r[1],
                "class_name": f"{r[2]} ({r[3]} - {r[4]})" if r[3] else r[2]
            }
            for r in cursor.fetchall()
        ]
    finally:
        conn.close()

    return render_template('index.html', teacher_name=session.get('full_name'), courses=courses)

@app.route('/api/select_course', methods=['POST'])
def select_course():
    """Khi giảng viên đổi lớp trên dropdown -> Cập nhật ClassSubjectID đang chọn"""
    global active_class_subject_id, enrolled_student_ids
    data = request.json
    course_id = data.get('course_id')
    if course_id:
        active_class_subject_id = int(course_id)
        enrolled_student_ids = load_enrolled_students(active_class_subject_id)
        return jsonify({"success": True, "active_course_id": active_class_subject_id})
    return jsonify({"error": "Chưa chọn lớp"}), 400

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/attendance_status')
def attendance_status():
    """Lấy danh sách SV thuộc lớp học phần từ dbo.Enrollments + trạng thái từ dbo.Attendance"""
    global active_class_subject_id, enrolled_student_ids
    class_subject_id = request.args.get('course_id', type=int)
    if not class_subject_id:
        return jsonify([])

    if active_class_subject_id != class_subject_id:
        active_class_subject_id = class_subject_id
        enrolled_student_ids = load_enrolled_students(class_subject_id)

    conn = connect_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.StudentID, s.StudentCode, u.FullName,
                   (SELECT TOP 1 a.CheckInTime 
                    FROM dbo.Attendance a
                    JOIN dbo.AttendanceSessions ses ON a.SessionID = ses.SessionID
                    WHERE a.StudentID = s.StudentID 
                      AND ses.ClassSubjectID = e.ClassSubjectID
                      AND ses.SessionDate = CAST(GETDATE() AS DATE)
                    ORDER BY a.CheckInTime ASC) as CheckInTime
            FROM dbo.Enrollments e
            JOIN dbo.Students s ON e.StudentID = s.StudentID
            JOIN dbo.Users u ON s.UserID = u.UserID
            WHERE e.ClassSubjectID = ?
            ORDER BY s.StudentCode ASC
        """, class_subject_id)
        
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

@app.route('/api/attendance_by_image', methods=['POST'])
def attendance_by_image():
    global active_class_subject_id, enrolled_student_ids
    if 'image' not in request.files:
        return jsonify({"error": "Không có ảnh"}), 400
    
    class_subject_id = request.form.get('course_id', type=int) or active_class_subject_id
    if class_subject_id:
        active_class_subject_id = class_subject_id
        enrolled_student_ids = load_enrolled_students(class_subject_id)

    file = request.files['image']
    nparr = np.frombuffer(file.read(), np.uint8)
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
            sim = np.dot(embedding_cam, db_face["embedding"]) / (np.linalg.norm(embedding_cam) * np.linalg.norm(db_face["embedding"]))
            if sim > max_similarity:
                max_similarity = sim
                best_match = db_face

        if max_similarity >= MIN_SIMILARITY:
            student_id = best_match["student_id"]
            student_code = best_match["student_code"]
            name = best_match["full_name"]
            
            if active_class_subject_id and student_id in enrolled_student_ids:
                mark_attendance(student_id, active_class_subject_id, max_similarity)
                recognized_students.append(f"{student_code} - {name}")
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, f"{student_code} - {name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 3)
                cv2.putText(img, f"{student_code} (Khac lop)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
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

@app.route('/api/export_excel')
def export_excel():
    class_subject_id = request.args.get('course_id', type=int) or active_class_subject_id
    conn = connect_db()
    try:
        query = """
            SELECT s.StudentCode AS [Mã Sinh Viên], 
                   u.FullName AS [Họ và Tên],
                   ISNULL(FORMAT(a.CheckInTime, 'HH:mm:ss'), '--:--:--') AS [Thời Gian Điểm Danh],
                   CASE WHEN a.CheckInTime IS NOT NULL THEN N'Đã có mặt' ELSE N'Chưa có mặt' END AS [Trạng Thái]
            FROM dbo.Enrollments e
            JOIN dbo.Students s ON e.StudentID = s.StudentID
            JOIN dbo.Users u ON s.UserID = u.UserID
            OUTER APPLY (
                SELECT TOP 1 att.CheckInTime 
                FROM dbo.Attendance att
                JOIN dbo.AttendanceSessions ses ON att.SessionID = ses.SessionID
                WHERE att.StudentID = s.StudentID 
                  AND ses.ClassSubjectID = e.ClassSubjectID
                  AND ses.SessionDate = CAST(GETDATE() AS DATE)
                ORDER BY att.CheckInTime ASC
            ) a
            WHERE e.ClassSubjectID = ?
            ORDER BY s.StudentCode ASC
        """
        df = pd.read_sql(query, conn, params=(class_subject_id,))
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='BaoCaoDiemDanh')
        output.seek(0)
        filename = f"DiemDanh_LopHocPhan_{class_subject_id}_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
        return send_file(output, as_attachment=True, download_name=filename, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    finally:
        conn.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)