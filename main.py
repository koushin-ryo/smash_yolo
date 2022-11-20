from flask import url_for, Flask, render_template
from flask import request, redirect, session
import cv2
import torch
import os
import glob
import sqlite3
import numpy as np
import pandas as pd
import base64
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
import shutil

database = "database.db"

#Flaskオブジェクト作成
app = Flask(__name__)

# image_folder
image_folder = os.path.join("static", "images")
app.config["UPLOAD_FOLDER"] = image_folder

# save
save_path = "save/"

# ライブラリの重複を許す
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def create_df_table():
    # データベースへの接続
    con = sqlite3.connect(database)
    con.execute("CREATE TABLE IF NOT EXISTS data (sec, rx, ry, box_x, box_y)")
    # 接続を閉じる
    con.close()

# 動画の画像化
def save_frame_range(video_file):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    #start_frame = round(start_sec*fps)
    #stop_frame = round(stop_sec*fps)

    # 各種設定
    start_frame = 1
    stop_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_frame = 10
    base_path = image_folder + "/"
    images_name = "test"

    if not cap.isOpened():
        return

    digit = 5
    output_list = []

    # 画像を特定のファイルに保存する
    for n in range(start_frame, stop_frame, step_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path+images_name, str(n).zfill(digit), "jpg"), frame)
        else:
            return

# 画像の推論
def output_Coordinate(img):
  # ボックスデータの取得
  results = model(img)
  objects = results.pandas().xyxy[0]
  
  if len(objects) == 2:
    # 絶対距離の計算
    xa = (objects["xmin"][0] + objects["xmax"][0])/2
    ya = (objects["ymin"][0] + objects["ymax"][0])/2
    xb = (objects["xmin"][1] + objects["xmax"][1])/2
    yb = (objects["ymin"][1] + objects["ymax"][1])/2
    rx = abs(xa - xb)
    ry = abs(ya - yb)
    # ボックスの長さ
    abox_x = abs(objects["xmax"][0] - objects["xmin"][0])
    abox_y = abs(objects["ymax"][0] - objects["ymin"][0])
    bbox_x = abs(objects["xmax"][1] - objects["xmin"][1])
    bbox_y = abs(objects["ymax"][1] - objects["ymin"][1])
    box_x = (abox_x + bbox_x)/2
    box_y = (abox_y + bbox_y)/2
  else:
    rx = None
    ry = None
    box_x = None
    box_y = None
  return rx, ry, box_x, box_y

# グラフの変換
def img_format(byte):
    byte = base64.b64encode(byte.getvalue()).decode("utf-8")
    img = "data:image/png;base64,{}".format(byte)
    return img

# ホームページ
@app.route("/")
def index():
    return render_template("index.html")

# uploadページ
@app.route("/uploads", methods=["POST", "GET"])
def upload_file():
    if request.method == "POST":
        # video取得する
        getfile = request.files["the_file"]
        # video保存
        filename = getfile.filename
        video_file = "/Users/ryounosukekouriki/Desktop/test_app/saves/" + filename
        getfile.save(video_file)
        # 動画を画像化して保存
        save_frame_range(video_file)
        # 画像表示ページを出力
        return render_template("index.html")
    else:
        return render_template("uploads.html")


# 推論
@app.route("/predict", methods=["GET","POST"])
def predict():
    images_dir = os.path.join("static", "images")
    images = sorted(glob.glob(images_dir+"/*"))

    # predict
    con = sqlite3.connect(database)
    for image in images:
        rx, ry, box_x, box_y = output_Coordinate(image)
        basename = os.path.splitext(os.path.basename(image))[0]
        sec = basename[len(basename)-5:]
        con.execute("INSERT INTO data VALUES(?, ?, ?, ?, ?)",
                [sec, rx, ry, box_x, box_y]
        )
    con.commit()
    con.close()

    return render_template(
        "predict.html"
    )
    
# 結果の表示
@app.route("/show", methods=["GET","POST"])
def show():
    max_num = 4

    # データ読み込み
    con = sqlite3.connect(database)
    # dataをpythonのタプルのリスト化する：fetchall
    data = con.execute("SELECT * FROM data").fetchall()
    con.close()

    df_dic = []
    for row in data:
        df_dic.append(
            {"sec":row[0],
            "rx":row[1],
            "ry":row[2],
            "box_x":row[3],
            "box_y":row[4]}
        )
    
    df = pd.DataFrame(df_dic)
    df["rx"] = df["rx"]/df["box_x"]
    df["ry"] = df["ry"]/df["box_y"]

    # x軸度数分布表
    x_num = len(df[df["rx"]<max_num]["rx"])
    xfreq = df[df["rx"]<max_num]["rx"].value_counts(bins=np.linspace(0,max_num,11), sort=False)
    cum_xfreq = xfreq.cumsum()
    table_x = pd.DataFrame(
        {
            "%":xfreq/x_num,
            "累積%":cum_xfreq/x_num
        },
        index = xfreq.index
    ).round(2)

    # x_mean
    x_mean = np.mean(df[df["rx"]<max_num]["rx"])

    # x軸間合い分布
    byte_x = BytesIO()
    sns.displot(df[(df["ry"]<2)&(df["rx"]<max_num)]["rx"],kind="kde")
    plt.xlim(0,max_num)
    plt.savefig(byte_x)
    graph_x = img_format(byte_x)

    # y軸度数分布表
    y_num = len(df[df["ry"]<max_num]["ry"])
    freq_y = df[df["ry"]<max_num]["ry"].value_counts(bins=np.linspace(0,max_num,11), sort=False)
    cum_yfreq = freq_y.cumsum()
    table_y = pd.DataFrame(
        {
            "%":freq_y/y_num,
            "累積%":cum_yfreq/y_num
        },
        index = freq_y.index
    ).round(2)

    # y_mean
    y_mean = np.mean(df[df["ry"]<max_num]["ry"])

    # y軸間合い分布
    byte_y = BytesIO()
    sns.displot(df[df["ry"]<max_num]["ry"],kind="kde")
    plt.xlim(0,max_num)
    plt.savefig(byte_y)
    graph_y = img_format(byte_y)

    return render_template(
        "show.html",
        table_x = table_x.to_html(header='true'),
        table_y = table_y.to_html(header='true'),
        x_mean = x_mean,
        y_mean = y_mean,
        graph_x = graph_x,
        graph_y = graph_y
    )

@app.route("/delete")
def delete():
    con = sqlite3.connect(database)
    con.execute("DELETE FROM data")
    con.commit()
    con.close()

    # 画像ファイルの削除
    shutil.rmtree(image_folder)
    os.mkdir(image_folder)

    # 確認
    con = sqlite3.connect(database)
    cheack = con.execute("SELECT * FROM data").fetchall()
    con.close()
    if cheack:
        cheack_data = "undeleted"
    else:
        cheack_data = "ok"

    return render_template(
        "delete.html",
        cheack_data = cheack_data
    )


if __name__=='__main__':
    # 初期化処理
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    app.run(debug=True)
    # 最初に空のテーブルを作成する
    create_df_table()