
import pyodbc

# =====================================
# 1. THONG TIN KET NOI SQL SERVER
# =====================================

SERVER = r".\SQLEXPRESS"
DATABASE = "FaceAttendanceDB"


# =====================================
# 2. HAM KET NOI DATABASE
# =====================================

def connect_db():

    drivers = pyodbc.drivers()

    driver = None

    for item in [
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server"
    ]:
        if item in drivers:
            driver = item
            break

    if driver is None:
        raise Exception("Chua cai ODBC Driver 17 hoac 18!")

    connection_string = (
        f"DRIVER={{{driver}}};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )

    return pyodbc.connect(connection_string)


# =====================================
# 3. KIEM TRA KET NOI
# =====================================

def test_connection():

    conn = None

    try:
        conn = connect_db()

        cursor = conn.cursor()
        cursor.execute("SELECT DB_NAME()")

        database_name = cursor.fetchone()[0]

        print("KET NOI THANH CONG!")
        print("Database:", database_name)

        return True

    except Exception as error:
        print("KET NOI THAT BAI!")
        print("Chi tiet:", error)

        return False

    finally:
        if conn is not None:
            conn.close()


# =====================================
# 4. LAY DANH SACH SINH VIEN
# =====================================

def show_students():

    conn = None

    try:
        conn = connect_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT TOP (10) *
            FROM dbo.Students
        """)

        columns = [column[0] for column in cursor.description]

        print("\n===== DANH SACH SINH VIEN =====")
        print("Cac cot:", columns)

        students = cursor.fetchall()

        if not students:
            print("Chua co sinh vien nao!")

        for student in students:

            print("\n--------------------------")

            for column, value in zip(columns, student):
                print(f"{column}: {value}")

        print("\nTong so sinh vien hien thi:", len(students))

    except Exception as error:
        print("LOI KHI DOC SINH VIEN:", error)

    finally:
        if conn is not None:
            conn.close()


# =====================================
# 5. CHAY CHUONG TRINH
# =====================================

if __name__ == "__main__":

    if test_connection():
        show_students()
