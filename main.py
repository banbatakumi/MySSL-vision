import cv2
import numpy as np
import time
import json

# --- 設定と定数をインポート ---
from config import (UDP_IP, UDP_PORT, REAL_CALIBRATION_DISTANCE_CM,
                    ROBOT_MARKER_MIN_AREA, ORANGE_BALL_MIN_AREA, BALL_COMPENSATION_TIME,
                    MIN_ARROW_LENGTH_PIXELS, WINDOW_NAME, MASK_TOGGLE_KEY)

# --- 各モジュールをインポート ---
from udp_sender import UDPSender
import image_processing as imgproc  # 画像処理関数
import calibration                # キャリブレーション関連 (グローバル変数とコールバック)
from detection import detect_robot_from_reds  # ロボット検出関数

# --- 変数初期化 ---
last_detected_orange_balls = []  # 最後に検出されたオレンジボールの座標
last_detection_time = 0  # 最後にボールが検出された時刻


def main():
    # --- 初期化 ---
    global last_detection_time, last_detected_orange_balls  # グローバル変数を参照
    # UDP送信クラスのインスタンス化
    udp_sender = UDPSender(UDP_IP, UDP_PORT)

    # カメラを開く
    cap = cv2.VideoCapture(0)  # 0はデフォルトカメラ
    if not cap.isOpened():
        print("Error: Could not open camera.")
        udp_sender.close()  # ソケットを閉じる
        return

    # ウィンドウを作成し、マウスコールバックを設定
    cv2.namedWindow(WINDOW_NAME)
    # calibrationモジュールの関数を設定
    cv2.setMouseCallback(WINDOW_NAME, calibration.mouse_hsv_callback)

    # --- 変数初期化 ---
    prev_time = time.time()
    mask_view_enabled = False  # マスク表示モードのフラグ

    # --- メインループ ---
    while True:
        # --- フレーム取得と前処理 ---
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # フレームの反転 (カメラの設置状況に合わせて調整)
        frame = cv2.flip(frame, 1)  # 水平反転
        frame = cv2.flip(frame, 0)  # 垂直反転

        # FPS計算
        current_time = time.time()
        delta_time = current_time - prev_time
        prev_time = current_time
        fps = 1 / delta_time if delta_time > 0 else 0

        # 画像サイズと中心座標
        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2

        # BGR画像をHSV色空間に変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- 現在のキャリブレーション情報を取得 ---
        calib_points, cm_scale = calibration.get_calibration_info()

        # --- 重心検出 ---
        red_centers = imgproc.get_centroids(
            hsv, "red", min_area=ROBOT_MARKER_MIN_AREA)
        green_centers = imgproc.get_centroids(
            hsv, "green", min_area=ROBOT_MARKER_MIN_AREA)
        yellow_centers = imgproc.get_centroids(
            hsv, "yellow", min_area=ROBOT_MARKER_MIN_AREA)
        blue_centers = imgproc.get_centroids(
            hsv, "blue", min_area=ROBOT_MARKER_MIN_AREA)
        orange_centers = imgproc.get_centroids(
            hsv, "orange", min_area=ORANGE_BALL_MIN_AREA)

        # --- 表示フレームの準備 ---
        display_frame = frame.copy()  # 元のフレームに描画する

        # マスク表示モードが有効な場合、マスクを適用
        if mask_view_enabled:
            combined_mask = imgproc.generate_combined_mask(hsv)
            display_frame = cv2.bitwise_and(
                display_frame, display_frame, mask=combined_mask)
            mask_status_text = "Masked View"
            mask_status_color = (0, 255, 255)  # 黄色
        else:
            mask_status_text = "Normal View"
            mask_status_color = (0, 255, 0)  # 緑色

        # --- ロボット検出と描画 ---
        # detect_robot_from_reds が display_frame に直接描画し、検出情報を返す
        yellow_robots = detect_robot_from_reds(
            red_centers, green_centers, yellow_centers, "yellow",
            display_frame, cm_scale, MIN_ARROW_LENGTH_PIXELS)
        blue_robots = detect_robot_from_reds(
            red_centers, green_centers, blue_centers, "blue",
            display_frame, cm_scale, MIN_ARROW_LENGTH_PIXELS)

        detected_orange_balls = []
        current_time = time.time()
        if orange_centers:
            # ボールが検出された場合
            for (cx, cy) in orange_centers:
                cx_centered_cm, cy_centered_cm = None, None
                if cm_scale is not None:
                    # 中心からの相対座標 (X右正, Y上正)
                    cx_centered_cm = (cx - center_x) * cm_scale
                    cy_centered_cm = (center_y - cy) * cm_scale

                # display_frame に円を描画
                cv2.circle(display_frame, (cx, cy), 15,
                           (0, 165, 255), 3)  # オレンジ色 BGR(Orange)

                # CMスケールがあれば情報をリストに追加
                if cm_scale is not None:
                    ball_info = {
                        "pos": (round(cx_centered_cm, 2), round(cy_centered_cm, 2))
                    }
                    detected_orange_balls.append(ball_info)

            # 検出されたボールを記録
            last_detected_orange_balls = detected_orange_balls
            last_detection_time = current_time
        else:
            # ボールが検出されなかった場合、一定時間内なら最後の座標を維持
            if current_time - last_detection_time <= BALL_COMPENSATION_TIME:
                detected_orange_balls = last_detected_orange_balls
            else:
                # 時間が経過したら座標をリセット
                detected_orange_balls = []

        # --- UDP送信データの準備と送信 ---
        vision_data = {
            # "timestamp": time.time(),
            "fps": round(fps, 2),
            "orange_balls": detected_orange_balls,
            "yellow_robots": yellow_robots,
            "blue_robots": blue_robots,
            "is_calibrated": cm_scale is not None  # キャリブレーション状態も送信
        }
        udp_sender.send_data(vision_data)

        # --- 画面表示情報の描画 ---

        # 1. キャリブレーション状態表示 (左上)
        calib_status_text = ""
        calib_color = (255, 255, 255)  # デフォルトは白
        if cm_scale is None:
            points_count = len(calib_points)
            if points_count == 0:
                calib_status_text = f"Calibrate: Click 1st point ({REAL_CALIBRATION_DISTANCE_CM}cm)"
                calib_color = (0, 255, 255)  # 黄色
            elif points_count == 1:
                calib_status_text = f"Calibrate: Click 2nd point ({REAL_CALIBRATION_DISTANCE_CM}cm)"
                calib_color = (0, 165, 255)  # オレンジ
            # points_countが2でもcm_scaleがNoneは計算失敗時 -> リセットされるのでここでは考慮不要
        else:
            calib_status_text = f"Scale: 1 pixel = {cm_scale:.4f} cm (Click to recalibrate)"
            calib_color = (0, 255, 0)  # 緑色

        cv2.putText(display_frame, calib_status_text, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame, calib_status_text, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, calib_color, 1, cv2.LINE_AA)

        # 2. クリックされたキャリブレーション点の描画 (スケール計算前のみ)
        if cm_scale is None:
            for i, pt in enumerate(calib_points):
                point_color = (0, 255, 255) if i == 0 else (
                    0, 165, 255)  # 1点目:黄, 2点目:オレンジ
                cv2.circle(display_frame, pt, 5, point_color, -1)
                pt_text = f"P{i+1}:({pt[0]},{pt[1]})"
                cv2.putText(display_frame, pt_text, (
                    pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(display_frame, pt_text, (
                    pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. FPS表示 (左上)
        fps_text = f"FPS: {fps:.1f}"  # FPS表示は小数点以下1桁に
        cv2.putText(display_frame, fps_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame, fps_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # ボールの座標の表示
        text = "Orange Ball"
        if cm_scale is not None and detected_orange_balls:
            text += f" | CM:({vision_data['orange_balls'][0]['pos'][0]:.1f},{vision_data['orange_balls'][0]['pos'][1]:.1f})"

        cv2.putText(display_frame, text, (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame, text, (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. マスクビューステータス表示 (右上)
        status_text_size, _ = cv2.getTextSize(
            mask_status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        status_text_pos = (width - status_text_size[0] - 10, 25)  # 右寄せで表示
        cv2.putText(display_frame, mask_status_text, status_text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame, mask_status_text, status_text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mask_status_color, 1, cv2.LINE_AA)

        # 5. マウスカーソル位置のHSV表示 (右下)
        mouse_x, mouse_y = calibration.get_mouse_position()  # calibrationモジュールから取得
        if 0 <= mouse_x < width and 0 <= mouse_y < height:
            hsv_value = hsv[mouse_y, mouse_x]
            mouse_hsv_text = f"HSV @({mouse_x},{mouse_y}): {hsv_value}"
            text_size, _ = cv2.getTextSize(
                mouse_hsv_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            text_x = width - text_size[0] - 10
            text_y = height - 10  # 右下
            cv2.putText(display_frame, mouse_hsv_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, mouse_hsv_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # --- フレーム表示 ---
        cv2.imshow(WINDOW_NAME, display_frame)

        # --- キー入力処理 ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 'q' キーで終了
            break
        elif key == MASK_TOGGLE_KEY:  # マスク表示切り替え
            mask_view_enabled = not mask_view_enabled
            print(
                f"Mask view {'enabled' if mask_view_enabled else 'disabled'}")

    # --- 終了処理 ---
    cap.release()          # カメラを解放
    cv2.destroyAllWindows()  # ウィンドウを閉じる
    udp_sender.close()     # UDPソケットを閉じる
    print("Application finished.")


if __name__ == "__main__":
    main()
