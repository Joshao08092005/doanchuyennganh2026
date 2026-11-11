from database import connect_db

conn = connect_db()
cur = conn.cursor()
cur.execute("""
    SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME IN ('Roles','Users','Students','Teachers','Subjects','Classes','ClassSubjects','Enrollments','AttendanceSessions','Attendance')
    ORDER BY TABLE_NAME, ORDINAL_POSITION
""")

current_table = ""
for t_name, c_name, d_type, is_null in cur.fetchall():
    if t_name != current_table:
        print(f"\n=== BẢNG: dbo.{t_name} ===")
        current_table = t_name
    print(f"  - {c_name} ({d_type}, Cho phép NULL: {is_null})")

conn.close()