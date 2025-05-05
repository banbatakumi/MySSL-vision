import cv2
import numpy as np
import math
import time
import socket  # socketモジュールを追加
import json   # jsonモジュールを追加

COLOR_RANGES = {
    "red": ((0, 80, 80), (10, 130, 255)),
    # 赤色の後半範囲を追加 (HSVのH値は0-180で表現されるため、赤は0付近と180付近の両方に現れる可能性がある)
    "red2": ((170, 80, 80), (180, 130, 255)),
    "yellow": ((20, 150, 200), (40, 200, 255)),
    "green": ((30, 60, 150), (60, 110, 255)),
    "blue": ((90, 100, 150), (120, 150, 255)),
    "orange": ((10, 150, 150), (20, 255, 255)),
}

# --- UDP 通信設定 ---
UDP_IP = "127.0.0.1"  # << ここを制御プログラムのIPアドレスに変更 >>
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
                        # 尺度計算後、ポイントの表示を消すためにリストをクリア
                        # ただし、描画ループ側で calibration_points が空になったら描画しないようにする
                    else:
                        print("Pixel distance is zero. Cannot calculate scale.")
                        cm_per_pixel_scale = None  # スケール無効化
                        # 計算失敗時もポイントはクリアしないでおく (描画ループで消える)


# 面積閾値を引数として受け取るように修正
def get_centroids(hsv, color_name, min_area=1000):
    masks = []
    if color_name == "red":
        masks.append(cv2.inRange(hsv, *COLOR_RANGES["red"]))
        if "red2" in COLOR_RANGES:
            masks.append(cv2.inRange(hsv, *COLOR_RANGES["red2"]))
        mask = cv2.bitwise_or(
            *masks) if masks else np.zeros(hsv.shape[:2], dtype=np.uint8)
    else:
        if color_name not in COLOR_RANGES:
            print(f"Warning: Color '{color_name}' not found in COLOR_RANGES.")
            return []
        mask = cv2.inRange(hsv, *COLOR_RANGES[color_name])

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

    if dist_p1_p2_sq == 0:
        return None

    best_m = None
    min_dist_from_line = float('inf')
    allowed_dist_ratio = 0.2  # 線分長の20%以内なら中間点として許容

    for m in middle_candidates:
        v_p1_m = np.array(m) - np.array(p1)
        dot_product = np.dot(v_p1_m, v_p1_p2)
        projection_ratio = dot_product / dist_p1_p2_sq

        if 0 <= projection_ratio <= 1:
            p1_arr = np.array(p1)
            p_prime = p1_arr + projection_ratio * v_p1_p2
            distance_from_line = np.linalg.norm(np.array(m) - p_prime)

            if distance_from_line < allowed_dist_ratio * distance(p1, p2):
                if distance_from_line < min_dist_from_line:
                    min_dist_from_line = distance_from_line
                    best_m = m

    return best_m


def detect_robot_from_reds(red_list, green_list, middle_list, middle_name, frame, cm_scale, min_arrow_length=50):
    height, width = frame.shape[:2]
    center_x = width // 2
    center_y = height // 2

    detected_robots = []

    used_green_indices = set()
    used_middle_indices = set()

    for r_idx, r in enumerate(red_list):
        available_greens = [(g, i) for i, g in enumerate(
            green_list) if i not in used_green_indices]
        closest_g_info = min(available_greens, key=lambda item: distance(
            r, item[0])) if available_greens else None

        if closest_g_info is None:
            continue

        g, g_idx = closest_g_info

        if distance(r, g) < min_arrow_length:
            continue

        available_middles = [(m, i) for i, m in enumerate(
            middle_list) if i not in used_middle_indices]
        m_coords_only = [item[0] for item in available_middles]
        m = find_middle(r, g, m_coords_only)

        if m is None:
            continue

        m_idx = -1
        for i, candidate_m in enumerate(middle_list):
            if np.array_equal(candidate_m, m) and i not in used_middle_indices:
                m_idx = i
                break

        if m_idx == -1:
            continue

        used_green_indices.add(g_idx)
        used_middle_indices.add(m_idx)

        # ロボットの中心座標 (ピクセル)
        cx, cy = int((r[0] + m[0] + g[0]) / 3), int((r[1] + m[1] + g[1]) / 3)

        # 中心基準のピクセル座標
        cx_centered = cx - center_x
        cy_centered = cy - center_y

        # 向きの計算 (赤 -> 緑)
        angle_rad = math.atan2(g[1] - r[1], g[0] - r[0])
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360

        label = f"{middle_name.upper()} robot"

        # --- CM単位の座標計算 ---
        cx_cm, cy_cm, cx_centered_cm, cy_centered_cm = None, None, None, None
        if cm_scale is not None:
            cx_cm = cx * cm_scale
            cy_cm = cy * cm_scale
            cx_centered_cm = (cx - center_x) * cm_scale
            cy_centered_cm = (cy - center_y) * cm_scale  # Y下向き正のままCMに変換

        # --- デバッグ/表示用の描画 ---
        cv2.circle(frame, r, 8, (0, 0, 255), -1)
        if middle_name == "yellow":
            cv2.circle(frame, m, 8, (0, 255, 255), -1)
        elif middle_name == "blue":
            cv2.circle(frame, m, 8, (255, 0, 0), -1)
        cv2.circle(frame, g, 8, (0, 255, 0), -1)

        arrow_length = 40
        v_x = g[0] - r[0]
        v_y = g[1] - r[1]
        vec_len = math.sqrt(v_x**2 + v_y**2)
        if vec_len > 0:
            unit_v_x = v_x / vec_len
            unit_v_y = v_y / vec_len
            arrow_end_x = int(cx + unit_v_x * arrow_length)
            arrow_end_y = int(cy + unit_v_y * arrow_length)
            cv2.arrowedLine(frame, (cx, cy), (arrow_end_x,
                            arrow_end_y), (255, 0, 255), 2)

        cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)

        # テキストで情報表示
        text = f"{label} @({cx_centered},{cy_centered}) {angle_deg:.1f} deg"
        if cm_scale is not None:
            text += f" | CM:({cx_centered_cm:.1f},{cy_centered_cm:.1f})"

        cv2.putText(
            frame,
            text,
            (cx + 15, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
        # --- 描画終わり ---

        # 検出結果をリストに追加 (CM座標も含む)
        robot_info = {
            "type": label,
            "center_pixel": (cx, cy),
            "center_relative_pixel": (cx_centered, cy_centered),
            "orientation_deg": angle_deg,
        }
        if cm_scale is not None:
            robot_info["center_cm"] = (round(cx_cm, 2), round(cy_cm, 2))
            robot_info["center_relative_cm"] = (
                round(cx_centered_cm, 2), round(cy_centered_cm, 2))

        detected_robots.append(robot_info)

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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.flip(frame, 0)

        current_time = time.time()
        delta_time = current_time - prev_time
        prev_time = current_time
        fps = 1 / delta_time if delta_time > 0 else 0

        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- マウスカーソル位置のHSV表示の更新 ---
        mouse_hsv_text = ""
        if 0 <= mouse_x < width and 0 <= mouse_y < height:
            hsv_value = hsv[mouse_y, mouse_x]
            mouse_hsv_text = f"Mouse @({mouse_x},{mouse_y}) HSV: {hsv_value}"

        # --- 検出処理 ---
        robot_marker_min_area = 800  # 調整してください
        red_centers = get_centroids(hsv, "red", min_area=robot_marker_min_area)
        green_centers = get_centroids(
            hsv, "green", min_area=robot_marker_min_area)
        yellow_centers = get_centroids(
            hsv, "yellow", min_area=robot_marker_min_area)
        blue_centers = get_centroids(
            hsv, "blue", min_area=robot_marker_min_area)

        orange_ball_min_area = 100  # 調整してください
        orange_centers = get_centroids(
            hsv, "orange", min_area=orange_ball_min_area)

        # ロボットの検出と描画 (CMスケールを渡す)
        yellow_robots = detect_robot_from_reds(
            red_centers, green_centers, yellow_centers, "yellow", frame, cm_per_pixel_scale)
        blue_robots = detect_robot_from_reds(
            red_centers, green_centers, blue_centers, "blue", frame, cm_per_pixel_scale)

        # オレンジボールの検出位置に円とテキストを描画 (CM座標も計算)
        detected_orange_balls = []
        for (cx, cy) in orange_centers:
            cx_centered = cx - center_x
            cy_centered = cy - center_y

            # --- CM単位の座標計算 ---
            cx_cm, cy_cm, cx_centered_cm, cy_centered_cm = None, None, None, None
            if cm_per_pixel_scale is not None:
                cx_cm = cx * cm_per_pixel_scale
                cy_cm = cy * cm_per_pixel_scale
                cx_centered_cm = (cx - center_x) * cm_per_pixel_scale
                cy_centered_cm = (cy - center_y) * \
                    cm_per_pixel_scale  # Y下向き正のままCMに変換

            cv2.circle(frame, (cx, cy), 15, (0, 165, 255), 3)
            text = f"Orange Ball @({cx_centered},{cy_centered})"
            if cm_per_pixel_scale is not None:
                text += f" | CM:({cx_centered_cm:.1f},{cy_centered_cm:.1f})"

            cv2.putText(frame, text, (cx + 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            # ボール情報をリストに追加 (CM座標も含む)
            ball_info = {
                "center_pixel": (cx, cy),
                "center_relative_pixel": (cx_centered, cy_centered),
            }
            if cm_per_pixel_scale is not None:
                ball_info["center_cm"] = (round(cx_cm, 2), round(cy_cm, 2))
                ball_info["center_relative_cm"] = (
                    round(cx_centered_cm, 2), round(cy_centered_cm, 2))

            detected_orange_balls.append(ball_info)

        # --- UDP送信データの準備 ---
        vision_data = {
            "timestamp": time.time(),
            "fps": fps,
            "scale_cm_per_pixel": cm_per_pixel_scale,  # スケール値自体も送信
            "orange_balls": detected_orange_balls,
            "yellow_robots": yellow_robots,
            "blue_robots": blue_robots
        }

        # データをJSON文字列に変換し、バイト列にする
        json_data = json.dumps(vision_data)
        byte_data = json_data.encode('utf-8')

        # --- UDP送信 ---
        if sock:
            try:
                sock.sendto(byte_data, (UDP_IP, UDP_PORT))
            except socket.error as e:
                # print(f"Failed to send UDP data: {e}") # 頻繁に出る場合はコメントアウト
                pass

        # --- 画面表示情報の描画 ---

        # キャリブレーション状態表示
        calibration_status_text = ""
        color = (255, 255, 255)  # Default white color

        if cm_per_pixel_scale is None:
            # スケール未設定の場合の表示
            if len(calibration_points) == 0:
                calibration_status_text = f"Calibrate: Click 1st point ({REAL_CALIBRATION_DISTANCE_CM}cm)"
                color = (0, 255, 255)  # Yellow
            elif len(calibration_points) == 1:
                calibration_status_text = f"Calibrate: Click 2nd point ({REAL_CALIBRATION_DISTANCE_CM}cm)"
                color = (0, 165, 255)  # Orange
            elif len(calibration_points) == 2:  # 2点クリックしたが計算失敗した場合など
                calibration_status_text = "Calibration failed. Click to retry."
                color = (0, 0, 255)  # Red
        else:
            # スケール設定済みの場合の表示
            calibration_status_text = f"Scale: 1 pixel = {cm_per_pixel_scale:.4f} cm (Click to recalibrate)"
            color = (0, 255, 0)  # Green

        cv2.putText(frame, calibration_status_text, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, calibration_status_text, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)

        # クリックされた点に円を描画 (スケール計算前のみ表示)
        if cm_per_pixel_scale is None:
            for i, pt in enumerate(calibration_points):
                if i == 0:
                    cv2.circle(frame, pt, 5, (0, 255, 255), -1)  # 黄色
                elif i == 1:
                    cv2.circle(frame, pt, 5, (0, 165, 255), -1)  # オレンジ
                # 点のそばに座標も表示
                cv2.putText(frame, f"({pt[0]},{pt[1]})", (pt[0]+10, pt[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # --- FPS表示 ---
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 1, cv2.LINE_AA)

        # --- マウスカーソル位置のHSV表示 ---
        if mouse_hsv_text:
            text_size, _ = cv2.getTextSize(
                mouse_hsv_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            text_x = width - text_size[0] - 10
            text_y = height - 10

            cv2.putText(frame, mouse_hsv_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, mouse_hsv_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 結果を表示
        cv2.imshow(window_name, frame)

        # 'q'キーで終了
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # 'r'キーでのリセット機能は左クリックでのリセットに置き換えたため削除

    # リソースの解放
    cap.release()
    cv2.destroyAllWindows()
    if sock:
        sock.close()
        print("UDP socket closed.")


if __name__ == "__main__":
    main()
