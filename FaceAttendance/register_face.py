import os
import cv2
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis
from database import connect_db

# Thư mục chứa ảnh của các sinh viên trong lớp
IMAGE_FOLDER = "student_images"
MODEL_NAME = "buffalo_s"

# Danh sách 10 sinh viên mặc định của lớp: Môn An toàn thông tin - Lớp 03
CLASS_STUDENTS = [
    ("SV001", "Nguyen Van An"),
    ("SV002", "Tran Thi Binh"),
    ("SV003", "Le Van Cuong"),
    ("SV004", "Pham Thi Dung"),
    ("SV005", "Hoang Van Em"),
    ("SV006", "Vu Thi Phuong"),
    ("SV007", "Dang Van Giang"),
    ("SV008", "Bui Thi Hoa"),
    ("SV009", "Do Van Khoa"),
    ("SV789", "Nguyen Van Hao")
]

print("Đang khởi tạo AI để trích xuất khuôn mặt hàng loạt...")
face_app = FaceAnalysis(
    name=MODEL_NAME,
    allowed_modules=['detection', 'recognition'],
    providers=['CPUExecutionProvider']
)
face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

def insert_face_data_smart(cursor, student_id, embedding_bytes, embedding_dim, img_path):
    """Hàm tự động kiểm tra mọi cột bắt buộc trong dbo.FaceData để không bao giờ lỗi NULL"""
    cursor.execute("""
        SELECT c.name, t.name AS type_name, c.is_nullable, c.is_identity,
               OBJECT_DEFINITION(c.default_object_id) AS default_val
        FROM sys.columns c
        JOIN sys.types t ON c.user_type_id = t.user_type_id
        WHERE c.object_id = OBJECT_ID('dbo.FaceData')
    """)
    cols_info = cursor.fetchall()

    insert_cols = []
    insert_vals = []
    now = datetime.now()

    for col_name, type_name, is_nullable, is_identity, default_val in cols_info:
        if is_identity:
            continue  # Bỏ qua cột khóa chính tự tăng (FaceID / FaceDataID)

        col_lower = col_name.lower()
        if col_lower == 'studentid':
            insert_cols.append(col_name)
            insert_vals.append(student_id)
        elif col_lower == 'faceembedding':
            insert_cols.append(col_name)
            insert_vals.append(embedding_bytes)
        elif col_lower == 'embeddingmodel':
            insert_cols.append(col_name)
            insert_vals.append(MODEL_NAME)
        elif col_lower == 'embeddingdimension':
            insert_cols.append(col_name)
            insert_vals.append(embedding_dim)
        elif col_lower == 'isactive':
            insert_cols.append(col_name)
            insert_vals.append(1)
        elif not is_nullable and default_val is None:
            # Tự động lấp đầy tất cả các cột NOT NULL khác (như ConsentAt, CreatedAt, ImagePath...)
            insert_cols.append(col_name)
            if 'date' in type_name or 'time' in type_name:
                insert_vals.append(now)
            elif 'int' in type_name or 'bit' in type_name or 'float' in type_name or 'decimal' in type_name:
                insert_vals.append(1)
            elif 'bin' in type_name:
                insert_vals.append(embedding_bytes)
            else:
                insert_vals.append(img_path)

    placeholders = ", ".join(["?"] * len(insert_cols))
    col_names_str = ", ".join(insert_cols)
    sql = f"INSERT INTO dbo.FaceData ({col_names_str}) VALUES ({placeholders})"
    cursor.execute(sql, insert_vals)

def sync_class_and_images():
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        print(f"📁 Đã tạo thư mục '{IMAGE_FOLDER}'. Hãy bỏ ảnh sinh viên vào đây rồi chạy lại!")
        return

    conn = connect_db()
    cursor = conn.cursor()

    try:
        # 1. Làm sạch dữ liệu điểm danh và khuôn mặt cũ
        cursor.execute("DELETE FROM dbo.Attendance")
        cursor.execute("DELETE FROM dbo.FaceData")

        # Lấy RoleID của Student
        cursor.execute("SELECT TOP 1 RoleID FROM dbo.Roles WHERE RoleName = N'Student'")
        role_row = cursor.fetchone()
        role_id = role_row[0] if role_row else 1

        # 2. Đảm bảo toàn bộ 10 sinh viên của lớp đều có trong bảng Users & Students
        student_map = {}
        for code, name in CLASS_STUDENTS:
            cursor.execute("SELECT StudentID FROM dbo.Students WHERE StudentCode = ?", code)
            row = cursor.fetchone()
            if row:
                student_id = row[0]
                cursor.execute("""
                    UPDATE u SET u.FullName = ?
                    FROM dbo.Users u
                    JOIN dbo.Students s ON u.UserID = s.UserID
                    WHERE s.StudentID = ?
                """, name, student_id)
            else:
                username = code.lower()
                email = f"{username}@vinhuni.edu.vn"
                cursor.execute("""
                    INSERT INTO dbo.Users (Username, PasswordHash, FullName, Email, RoleID)
                    VALUES (?, '123456', ?, ?, ?)
                """, username, name, email, role_id)
                cursor.execute("SELECT @@IDENTITY")
                user_id = int(cursor.fetchone()[0])

                cursor.execute("""
                    INSERT INTO dbo.Students (UserID, StudentCode)
                    VALUES (?, ?)
                """, user_id, code)
                cursor.execute("SELECT @@IDENTITY")
                student_id = int(cursor.fetchone()[0])

            student_map[code] = student_id

        conn.commit()
        print(f"✅ Đã khởi tạo danh sách {len(CLASS_STUDENTS)} sinh viên (Môn: An toàn thông tin - Lớp 03) trong SQL!")

        # 3. Quét toàn bộ file ảnh trong thư mục student_images để nạp khuôn mặt vào SQL
        files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if not files:
            print(f"⚠️ Thư mục '{IMAGE_FOLDER}' hiện đang trống! Hãy copy ảnh (VD: SV789_Nguyen Van Hao.jpg) vào thư mục '{IMAGE_FOLDER}' rồi chạy lại file này.")
            return

        registered_count = 0
        for filename in files:
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split("_")
            student_code = parts[0].strip().upper()

            if student_code not in student_map:
                full_name = parts[1].strip() if len(parts) > 1 else student_code
                cursor.execute("""
                    INSERT INTO dbo.Users (Username, PasswordHash, FullName, Email, RoleID)
                    VALUES (?, '123456', ?, ?, ?)
                """, student_code.lower(), full_name, f"{student_code.lower()}@vinhuni.edu.vn", role_id)
                cursor.execute("SELECT @@IDENTITY")
                uid = int(cursor.fetchone()[0])
                cursor.execute("INSERT INTO dbo.Students (UserID, StudentCode) VALUES (?, ?)", uid, student_code)
                cursor.execute("SELECT @@IDENTITY")
                student_map[student_code] = int(cursor.fetchone()[0])
                conn.commit()

            student_id = student_map[student_code]
            img_path = os.path.join(IMAGE_FOLDER, filename)

            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                print(f"❌ Không đọc được ảnh: {filename}")
                continue

            faces = face_app.get(img)
            if len(faces) == 0:
                print(f"⚠️ Không tìm thấy khuôn mặt nào trong ảnh: {filename}")
                continue

            faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
            embedding_vec = faces[0].embedding.astype(np.float32)
            embedding_bytes = embedding_vec.tobytes()
            embedding_dim = len(embedding_vec)

            # Gọi hàm INSERT thông minh tự động điền đủ mọi cột NOT NULL
            insert_face_data_smart(cursor, student_id, embedding_bytes, embedding_dim, img_path)
            
            registered_count += 1
            print(f"📸 Đã nạp khuôn mặt thành công từ ảnh: {filename} -> Mã SV: {student_code}")

        conn.commit()
        print("==================================================================")
        print(f"🎉 HOÀN TẤT! Đã nạp {registered_count} khuôn mặt từ thư mục '{IMAGE_FOLDER}' vào SQL.")
        print("Bây giờ hãy chạy 'python app.py' để bắt đầu điểm danh!")
        print("==================================================================")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    sync_class_and_images()