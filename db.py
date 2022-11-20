import sqlite3

database = "database.db"
def create_books_table():
    # データベースへの接続
    con = sqlite3.connect(database)
    con.execute("CREATE TABLE IF NOT EXISTS data (sec, rx, ry, box_x, box_y)")
    # 接続を閉じる
    con.close()