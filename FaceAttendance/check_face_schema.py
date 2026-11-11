
from database import connect_db

conn = None

try:
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMNPROPERTY(
                OBJECT_ID(TABLE_SCHEMA + '.' + TABLE_NAME),
                COLUMN_NAME,
                'IsIdentity'
            ) AS IsIdentity,
            COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'FaceData'
        ORDER BY ORDINAL_POSITION
    """)

    print("===== CHI TIET BANG FaceData =====")

    for row in cursor.fetchall():
        print(
            f"Ten cot: {row[0]} | "
            f"Kieu: {row[1]} | "
            f"Cho phep NULL: {row[2]} | "
            f"Tu tang: {row[3]} | "
            f"Mac dinh: {row[4]}"
        )

    cursor.execute("""
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'Students'
        ORDER BY ORDINAL_POSITION
    """)

    print("\n===== CHI TIET BANG Students =====")

    for row in cursor.fetchall():
        print(row)

except Exception as error:
    print("LOI:", error)

finally:
    if conn is not None:
        conn.close()
