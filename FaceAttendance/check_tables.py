
from database import connect_db

def check_face_data():
    conn = None
    try:
        conn = connect_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'FaceData'
            ORDER BY ORDINAL_POSITION
        """)

        print("===== CAU TRUC BANG FaceData =====")

        columns = cursor.fetchall()

        for column in columns:
            print(
                "Ten cot:", column[0],
                "| Kieu du lieu:", column[1],
                "| Do dai:", column[2]
            )

        if not columns:
            print("Khong tim thay bang FaceData!")

    except Exception as error:
        print("Loi:", error)

    finally:
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    check_face_data()
