# --- START OF FILE main.py ---

import cv2
import numpy as np
import math
import time
import socket  # socketモジュールを追加
import json   # jsonモジュールを追加

COLOR_RANGES = {
    "red": ((0, 90, 80), (10, 255, 255)),  # V max を 255 に変更
    # 赤色の後半範囲を追加 (HSVのH値は0-180で表現されるため、赤は0付近と180付近の両方に現れる可能性がある)
    "red2": ((170, 90, 80), (180, 255, 255)),  # V max を 255 に変更
    "yellow": ((20, 100, 150), (40, 255, 255)),  # V min を少し上げる
    "green": ((40, 40, 100), (80, 255, 255)),  # S min, V min を調整
    "blue": ((90, 80, 100), (130, 255, 255)),  # H範囲広げ、S min, V min を調整
    "orange": ((5, 150, 150), (20, 255, 255)),  # H min を調整
}

# --- UDP 通信設定 ---
UDP_IP = "192.168.50.86"  # << ここを制御プログラムのIPアドレスに変更 >>
UDP_PORT = 50007     # << 任意だが、制御プログラムと合わせる >>
# -------------------

# UDPソケットを作成
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"UDP socket created for sending to {UDP_IP}:{UDP_PORT}")
except socket.error as e:
    print(f"Failed to create UDP socket: {e}")
    sock = None

# --- スケールキャリブレーション用変数 ---
calibration_points = []  # クリックされた座標を格納 [point1, point2]
cm_per_pixel_scale = None  # 1ピクセルあたりのCM数 (計算結果)
REAL_CALIBRATION_DISTANCE_CM = 50  # キャリブレーションで使用する実際の距離 (cm)
# ------------------------------------

# マウスイベントの座標を保持するグローバル変数
mouse_x, mouse_y = -1, -1
# マウスイベントコールバック関数


def mouse_hsv_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, calibration_points, cm_per_pixel_scale

    # マウス移動時のHSV表示用
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

    # 左クリック時のキャリブレーション点記録またはリセット
    if event == cv2.EVENT_LBUTTONDOWN:
        if cm_per_pixel_scale is not None:
            # スケールが計算済みの場合、左クリックでリセット
            calibration_points = []
            cm_per_pixel_scale = None
            print("Calibration reset by left click.")

        # リセットされたか、まだスケール計算前の状態であれば点を追加
        if cm_per_pixel_scale is None:
            if len(calibration_points) < 2:
                calibration_points.append((x, y))
                print(
                    f"Calibration point {len(calibration_points)} recorded: ({x}, {y})")

                if len(calibration_points) == 2:
                    # 2点目がクリックされたら尺度を計算
                    p1 = calibration_points[0]
                    p2 = calibration_points[1]
                    pixel_distance = distance(p1, p2)
                    if pixel_distance > 0:
                        cm_per_pixel_scale = REAL_CALIBRATION_DISTANCE_CM / pixel_distance
                        print(
                            f"Scale calculated: 1 pixel = {cm_per_pixel_scale:.4f} cm")
                        # 尺度計算後、ポイントの表示は描画ループ側で制御するのでクリアしない
                        # calibration_points = [] # Keep points for potential drawing until reset
                    else:
                        print("Pixel distance is zero. Cannot calculate scale.")
                        cm_per_pixel_scale = None  # スケール無効化
                        calibration_points = []  # 失敗時はリセット


# 面積閾値を引数として受け取るように修正
# この関数は重心計算のみを行う
def get_centroids(hsv, color_name, min_area=1000):
    masks = []
    if color_name == "red":
        masks.append(cv2.inRange(hsv, *COLOR_RANGES["red"]))
        if "red2" in COLOR_RANGES:
            masks.append(cv2.inRange(hsv, *COLOR_RANGES["red2"]))
        # Handle case where only one red range might exist or masks list could be empty
        mask = cv2.bitwise_or(*masks) if len(masks) > 1 else (
            masks[0] if masks else np.zeros(hsv.shape[:2], dtype=np.uint8))
    else:
        if color_name not in COLOR_RANGES:
            # print(f"Warning: Color '{color_name}' not found in COLOR_RANGES.") # Suppress frequent warnings
            return []
        mask = cv2.inRange(hsv, *COLOR_RANGES[color_name])

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                result.append((cx, cy))
    return result

# マスク表示用に、指定されたHSV画像から全色の合成マスクを生成する関数


def generate_combined_mask(hsv):
    all_masks = []
    # get_centroidsと同じカーネルを使用
    kernel = np.ones((5, 5), np.uint8)

    # 各色についてマスクを生成
    for color_name, (lower, upper) in COLOR_RANGES.items():
        if color_name == "red2":  # redでまとめて処理するのでスキップ
            continue
        if color_name == "red":
            mask1 = cv2.inRange(hsv, *COLOR_RANGES["red"])
            # red2 がなければ red を使う
            mask2 = cv2.inRange(
                hsv, *COLOR_RANGES.get("red2", COLOR_RANGES["red"]))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, lower, upper)

        # マスクのノイズ除去 (get_centroids と同様の処理)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        all_masks.append(mask)

    if not all_masks:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)  # マスクがない場合は黒画像

    # 全てのマスクを合成
    combined_mask = all_masks[0]
    for i in range(1, len(all_masks)):
        combined_mask = cv2.bitwise_or(combined_mask, all_masks[i])

    return combined_mask


def distance(p1, p2):
    p1_arr = np.array(p1)
    p2_arr = np.array(p2)
    return np.linalg.norm(p1_arr - p2_arr)


def find_closest(p, points):
    if not points:
        return None
    return min(points, key=lambda g: distance(p, g))


def find_middle(p1, p2, middle_candidates):
    if not middle_candidates:
        return None

    v_p1_p2 = np.array(p2) - np.array(p1)
    dist_p1_p2_sq = np.sum(v_p1_p2**2)

    if dist_p1_p2_sq == 0:  # p1とp2が同じ点の場合
        return None

    best_m = None
    min_dist_from_line = float('inf')
    # 線分長に対する許容距離の割合 (30%以内なら中間点候補とする)
    allowed_dist_ratio = 0.3

    for m in middle_candidates:
        v_p1_m = np.array(m) - np.array(p1)
        # ベクトルv_p1_mのv_p1_p2への射影を計算
        dot_product = np.dot(v_p1_m, v_p1_p2)
        # 射影の位置が線分p1-p2の間にあるかチェック (0 <= t <= 1)
        projection_ratio = dot_product / dist_p1_p2_sq

        if 0 <= projection_ratio <= 1:
            p1_arr = np.array(p1)
            # 線分p1-p2上で点mに最も近い点(垂線の足)p_primeを計算
            p_prime = p1_arr + projection_ratio * v_p1_p2
            # 点mと垂線の足p_primeとの距離を計算
            distance_from_line = np.linalg.norm(np.array(m) - p_prime)

            # 線分からの距離が許容範囲内かチェック
            if distance_from_line < allowed_dist_ratio * math.sqrt(dist_p1_p2_sq):
                # 最も線分に近い点を中間点として採用
                if distance_from_line < min_dist_from_line:
                    min_dist_from_line = distance_from_line
                    best_m = m

    return best_m


# 描画対象のフレーム(display_frame)を引数に追加
def detect_robot_from_reds(red_list, green_list, middle_list, middle_name, frame_to_draw, cm_scale, min_arrow_length=50):
    height, width = frame_to_draw.shape[:2]
    center_x = width // 2
    center_y = height // 2

    detected_robots = []

    # 各マーカーが既に使用されたかを追跡するためのセット
    used_red_indices = set()
    used_green_indices = set()
    used_middle_indices = set()

    # 赤マーカーを基準にロボットを探す
    for r_idx, r in enumerate(red_list):
        if r_idx in used_red_indices:
            continue  # この赤マーカーは既に使用済み

        # 利用可能な緑マーカーの中から、現在の赤マーカーに最も近いものを探す
        closest_g_info = None
        min_dist_g = float('inf')
        for g_idx, g in enumerate(green_list):
            if g_idx in used_green_indices:
                continue  # この緑マーカーは既に使用済み
            d = distance(r, g)
            if d < min_dist_g:
                min_dist_g = d
                closest_g_info = (g, g_idx)

        # 近い緑マーカーが見つからないか、近すぎる場合はスキップ
        if closest_g_info is None or min_dist_g < min_arrow_length:
            continue

        g, g_idx = closest_g_info

        # 利用可能な中間マーカー（黄色または青）の中から、線分r-gの間にあり最も近いものを探す
        potential_middles = []
        for m_idx, m in enumerate(middle_list):
            if m_idx not in used_middle_indices:
                potential_middles.append((m, m_idx))  # (座標, 元のインデックス) のタプル

        middle_coords_only = [item[0]
                              for item in potential_middles]  # find_middleには座標リストを渡す
        best_m_coord = find_middle(r, g, middle_coords_only)

        if best_m_coord is None:  # 適切な中間マーカーが見つからなかった
            continue

        # 見つかった中間マーカーの元のインデックスを探す
        m_idx = -1
        for m_coord, original_idx in potential_middles:
            if np.array_equal(m_coord, best_m_coord):
                m_idx = original_idx
                break

        if m_idx == -1:  # 基本的に起こらないはずだが念のため
            continue

        # この組み合わせで見つかったロボットに使用したマーカーを記録
        used_red_indices.add(r_idx)
        used_green_indices.add(g_idx)
        used_middle_indices.add(m_idx)

        # 中間点の座標をmとする
        m = best_m_coord

        # ロボットの中心座標 (3点の平均)
        cx, cy = int((r[0] + m[0] + g[0]) / 3), int((r[1] + m[1] + g[1]) / 3)

        # 向きの計算 (赤 -> 緑のベクトル)
        angle_rad = math.atan2(g[1] - r[1], g[0] - r[0])
        angle_deg = math.degrees(angle_rad)
        # 角度を -180 から 180 の範囲に正規化
        if angle_deg > 180:
            angle_deg -= 360
        elif angle_deg <= -180:
            angle_deg += 360

        label = f"{middle_name.upper()} robot"

        # --- CM単位の座標計算 ---
        cx_centered_cm, cy_centered_cm = None, None
        if cm_scale is not None:
            # 中心からの相対座標を計算 (Y軸は上向きを正とする)
            cx_centered_cm = (cx - center_x) * cm_scale
            cy_centered_cm = (cy - center_y) * -cm_scale

        # --- デバッグ/表示用の描画 (frame_to_draw に対して行う) ---
        # 各マーカーに円を描画
        cv2.circle(frame_to_draw, r, 8, (0, 0, 255), -1)  # 赤
        if middle_name == "yellow":
            cv2.circle(frame_to_draw, m, 8, (0, 255, 255), -1)  # 黄
        elif middle_name == "blue":
            cv2.circle(frame_to_draw, m, 8, (255, 0, 0), -1)  # 青
        cv2.circle(frame_to_draw, g, 8, (0, 255, 0), -1)  # 緑

        # ロボットの中心から向きを示す矢印を描画
        arrow_length = 40
        v_x = g[0] - r[0]
        v_y = g[1] - r[1]
        vec_len = math.sqrt(v_x**2 + v_y**2)
        if vec_len > 0:
            unit_v_x = v_x / vec_len
            unit_v_y = v_y / vec_len
            arrow_end_x = int(cx + unit_v_x * arrow_length)
            arrow_end_y = int(cy + unit_v_y * arrow_length)
            cv2.arrowedLine(frame_to_draw, (cx, cy), (arrow_end_x, arrow_end_y),
                            (255, 0, 255), 2)  # マゼンタ色の矢印

        # ロボットの中心に小さい黒丸を描画
        cv2.circle(frame_to_draw, (cx, cy), 5, (0, 0, 0), -1)

        # テキストで情報表示 (黒縁取り付き)
        text = f"{label} {angle_deg:.1f} deg"
        if cm_scale is not None:
            text += f" | CM:({cx_centered_cm:.1f},{cy_centered_cm:.1f})"

        # テキスト描画 (黒縁取り)
        cv2.putText(frame_to_draw, text, (cx + 15, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        # テキスト描画 (白文字)
        cv2.putText(frame_to_draw, text, (cx + 15, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        # --- 描画終わり ---

        # CMスケールが計算済みの場合、検出情報をリストに追加
        if cm_scale is not None:
            robot_info = {
                "type": label,
                "orientation_deg": round(angle_deg, 2),
                "center_relative_cm": (
                    round(cx_centered_cm, 2), round(cy_centered_cm, 2))
            }
            detected_robots.append(robot_info)

    # 検出されたロボットの情報リストを返す (描画は frame_to_draw に対して完了済み)
    return detected_robots


def main():
    global sock, mouse_x, mouse_y, calibration_points, cm_per_pixel_scale

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        if sock:
            sock.close()
        return

    window_name = "Detection Results"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_hsv_callback)

    prev_time = time.time()
    mask_view_enabled = False  # マスク表示モードのフラグ (初期値はFalse)
    mask_toggle_key = ord('m')  # マスク表示を切り替えるキー (ここでは 'm')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # フレームの反転処理 (カメラの設置向きに合わせる)
        frame = cv2.flip(frame, 1)  # 水平反転
        frame = cv2.flip(frame, 0)  # 垂直反転

        current_time = time.time()
        delta_time = current_time - prev_time
        prev_time = current_time
        fps = 1 / delta_time if delta_time > 0 else 0

        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2

        # BGR画像をHSV色空間に変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- マウスカーソル位置のHSV表示用テキスト生成 ---
        mouse_hsv_text = ""
        if 0 <= mouse_x < width and 0 <= mouse_y < height:
            hsv_value = hsv[mouse_y, mouse_x]
            mouse_hsv_text = f"Mouse @({mouse_x},{mouse_y}) HSV: {hsv_value}"

        # --- 各色の重心検出 ---
        robot_marker_min_area = 800  # ロボットマーカーの最小面積閾値
        red_centers = get_centroids(hsv, "red", min_area=robot_marker_min_area)
        green_centers = get_centroids(
            hsv, "green", min_area=robot_marker_min_area)
        yellow_centers = get_centroids(
            hsv, "yellow", min_area=robot_marker_min_area)
        blue_centers = get_centroids(
            hsv, "blue", min_area=robot_marker_min_area)

        orange_ball_min_area = 100   # オレンジボールの最小面積閾値
        orange_centers = get_centroids(
            hsv, "orange", min_area=orange_ball_min_area)

        # --- マスクビュー用の合成マスク生成 ---
        combined_mask = generate_combined_mask(hsv)

        # --- 表示フレームの決定 ---
        display_frame = frame.copy()  # 描画用に元のフレームをコピー
        status_text = "Normal View"
        status_color = (0, 255, 0)  # 通常ビューは緑
        if mask_view_enabled:
            # マスクビューが有効な場合、合成マスクを適用
            display_frame = cv2.bitwise_and(
                display_frame, display_frame, mask=combined_mask)
            status_text = "Masked View"
            status_color = (0, 255, 255)  # マスクビューは黄色

        # --- ロボットとボールの検出・描画 ---
        # 注意: 検出ロジック自体は元のhsv画像で行い、描画は display_frame に対して行う
        # detect_robot_from_reds 関数が描画対象フレーム(display_frame)を受け取るように変更済み
        yellow_robots = detect_robot_from_reds(
            red_centers, green_centers, yellow_centers, "yellow", display_frame, cm_per_pixel_scale)
        blue_robots = detect_robot_from_reds(
            red_centers, green_centers, blue_centers, "blue", display_frame, cm_per_pixel_scale)

        # オレンジボールの検出位置に円とテキストを描画
        detected_orange_balls = []
        for (cx, cy) in orange_centers:
            # CM座標計算
            cx_centered_cm, cy_centered_cm = None, None
            if cm_per_pixel_scale is not None:
                cx_centered_cm = (cx - center_x) * cm_per_pixel_scale
                cy_centered_cm = (cy - center_y) * -cm_per_pixel_scale

            # display_frame に描画
            cv2.circle(display_frame, (cx, cy), 15,
                       (0, 165, 255), 3)  # オレンジ色の円
            text = "Orange Ball"
            if cm_per_pixel_scale is not None:
                text += f" | CM:({cx_centered_cm:.1f},{cy_centered_cm:.1f})"

            # テキスト描画 (黒縁取り + 白文字)
            cv2.putText(display_frame, text, (cx + 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, text, (cx + 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # CMスケールがあれば情報をリストに追加
            if cm_per_pixel_scale is not None:
                ball_info = {
                    "center_relative_cm": (
                        round(cx_centered_cm, 2), round(cy_centered_cm, 2))
                }
                detected_orange_balls.append(ball_info)

        # --- UDP送信データの準備 ---
        vision_data = {
            "timestamp": time.time(),
            "fps": round(fps, 2),
            "orange_balls": detected_orange_balls,
            "yellow_robots": yellow_robots,
            "blue_robots": blue_robots
        }

        json_data = json.dumps(vision_data)
        byte_data = json_data.encode('utf-8')

        # --- UDP送信 ---
        if sock:
            try:
                sock.sendto(byte_data, (UDP_IP, UDP_PORT))
            except socket.error as e:
                # エラーが頻発する場合にコンソールが溢れないようにコメントアウト推奨
                # print(f"Failed to send UDP data: {e}")
                pass  # 接続エラー時などは無視して継続

        # --- 画面表示情報の描画 (display_frame に対して行う) ---

        # キャリブレーション状態表示 (Y座標を調整して他のテキストと重ならないように)
        calibration_status_text = ""
        calib_color = (255, 255, 255)
        if cm_per_pixel_scale is None:
            if len(calibration_points) == 0:
                calibration_status_text = f"Calibrate: Click 1st point ({REAL_CALIBRATION_DISTANCE_CM}cm)"
                calib_color = (0, 255, 255)  # 黄色
            elif len(calibration_points) == 1:
                calibration_status_text = f"Calibrate: Click 2nd point ({REAL_CALIBRATION_DISTANCE_CM}cm)"
                calib_color = (0, 165, 255)  # オレンジ
            # 2点クリック済みだが計算失敗時は calibration_points がクリアされるので、この分岐は不要
        else:
            calibration_status_text = f"Scale: 1 pixel = {cm_per_pixel_scale:.4f} cm (Click to recalibrate)"
            calib_color = (0, 255, 0)  # 緑色

        cv2.putText(display_frame, calibration_status_text, (10, 55),  # Y=55
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame, calibration_status_text, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, calib_color, 1, cv2.LINE_AA)

        # クリックされたキャリブレーション点を描画 (スケール計算前のみ)
        if cm_per_pixel_scale is None and len(calibration_points) > 0:
            for i, pt in enumerate(calibration_points):
                point_color = (0, 255, 255) if i == 0 else (
                    0, 165, 255)  # 1点目:黄, 2点目:オレンジ
                cv2.circle(display_frame, pt, 5, point_color, -1)
                pt_text = f"({pt[0]},{pt[1]})"
                cv2.putText(display_frame, pt_text, (pt[0]+10, pt[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(display_frame, pt_text, (pt[0]+10, pt[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # --- FPS表示 ---
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(display_frame, fps_text, (10, 25),  # Y=25
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 1, cv2.LINE_AA)

        # --- マスクビュー ステータス表示 (右上) ---
        status_text_pos = (width - 180, 25)  # 右上の座標を調整
        cv2.putText(display_frame, status_text, status_text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame, status_text, status_text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 1, cv2.LINE_AA)

        # --- マウスカーソル位置のHSV表示 (右下) ---
        if mouse_hsv_text:
            text_size, _ = cv2.getTextSize(
                mouse_hsv_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            text_x = width - text_size[0] - 10
            text_y = height - 10  # 右下の座標

            cv2.putText(display_frame, mouse_hsv_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, mouse_hsv_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 結果を表示
        cv2.imshow(window_name, display_frame)

        # キー入力処理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 'q' キーでループを抜ける
            break
        elif key == mask_toggle_key:  # 指定したキー (デフォルト 'm') でマスク表示を切り替え
            mask_view_enabled = not mask_view_enabled
            print(
                f"Mask view {'enabled' if mask_view_enabled else 'disabled'} by pressing '{chr(mask_toggle_key)}'")

    # リソースの解放
    cap.release()
    cv2.destroyAllWindows()
    if sock:
        sock.close()
        print("UDP socket closed.")


if __name__ == "__main__":
    main()

# --- END OF FILE ---
