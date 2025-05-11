import cv2
import numpy as np
import time
import json

# --- 設定と定数をインポート ---
from config import (UDP_IP, UDP_PORT, REAL_CALIBRATION_DISTANCE_M,
                    ROBOT_MARKER_MIN_AREA, ORANGE_BALL_MIN_AREA, BALL_COMPENSATION_TIME,
                    WINDOW_NAME, MASK_TOGGLE_KEY, DEBUG_TOGGLE_KEY,
                    MAX_DIST_MIDDLE_TO_OUTER_MARKER, MIN_PAIR_DISTANCE_DIFFERENCE,
                    MIN_MARKERS_SEPARATION_IN_PAIR)

# --- 各モジュールをインポート ---
from udp_sender import UDPSender
import image_processing as imgproc  # 画像処理関数
import calibration                # キャリブレーション関連
from detection import detect_fivel_marker_robots  # 新しい検出関数

# --- 変数初期化 ---
last_detected_orange_balls_info = []  # 最後に検出されたオレンジボールの情報リスト
last_detection_time_orange = 0  # 最後にボールが検出された時刻


def main():
    global last_detection_time_orange, last_detected_orange_balls_info
    udp_sender = UDPSender(UDP_IP, UDP_PORT)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        udp_sender.close()
        return

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, calibration.mouse_hsv_callback)

    prev_time = time.time()
    mask_view_enabled = False
    debug_view_enabled = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.flip(frame, 0)

        current_time = time.time()  # For timestamp
        delta_time = current_time - prev_time
        prev_time = current_time
        fps = 1 / delta_time if delta_time > 0 else 0

        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        calib_points, m_scale = calibration.get_calibration_info()

        # --- 重心検出 ---
        magenta_centers = imgproc.get_centroids(
            hsv, "magenta", min_area=ROBOT_MARKER_MIN_AREA)
        green_centers = imgproc.get_centroids(
            hsv, "green", min_area=ROBOT_MARKER_MIN_AREA)
        yellow_centers = imgproc.get_centroids(
            hsv, "yellow", min_area=ROBOT_MARKER_MIN_AREA)
        blue_centers = imgproc.get_centroids(
            hsv, "blue", min_area=ROBOT_MARKER_MIN_AREA)
        orange_centers_coords = imgproc.get_centroids(
            hsv, "orange", min_area=ORANGE_BALL_MIN_AREA)

        display_frame = frame.copy()

        if mask_view_enabled:
            combined_mask = imgproc.generate_combined_mask(hsv)
            display_frame = cv2.bitwise_and(
                display_frame, display_frame, mask=combined_mask)
            mask_status_text = "Masked View"
            mask_status_color = (0, 255, 255)
        else:
            mask_status_text = "Normal View"
            mask_status_color = (0, 255, 0)

        # --- ロボット検出 ---
        all_detected_robots_info = []
        globally_used_magenta_coords = set()
        globally_used_green_coords = set()
        globally_used_yellow_coords = set()
        globally_used_blue_coords = set()

        yellow_team_robots, used_m_y, used_g_y, used_mid_y = detect_fivel_marker_robots(
            magenta_centers, green_centers, yellow_centers, "yellow",
            display_frame, m_scale,
            globally_used_magenta_coords, globally_used_green_coords, globally_used_yellow_coords, debug_view_enabled,
            MAX_DIST_MIDDLE_TO_OUTER_MARKER, MIN_PAIR_DISTANCE_DIFFERENCE, MIN_MARKERS_SEPARATION_IN_PAIR
        )
        all_detected_robots_info.extend(yellow_team_robots)
        globally_used_magenta_coords.update(used_m_y)
        globally_used_green_coords.update(used_g_y)
        globally_used_yellow_coords.update(used_mid_y)

        blue_team_robots, used_m_b, used_g_b, used_mid_b = detect_fivel_marker_robots(
            magenta_centers, green_centers, blue_centers, "blue",
            display_frame, m_scale,
            globally_used_magenta_coords, globally_used_green_coords, globally_used_blue_coords, debug_view_enabled,
            MAX_DIST_MIDDLE_TO_OUTER_MARKER, MIN_PAIR_DISTANCE_DIFFERENCE, MIN_MARKERS_SEPARATION_IN_PAIR
        )
        all_detected_robots_info.extend(blue_team_robots)
        globally_used_magenta_coords.update(used_m_b)
        globally_used_green_coords.update(used_g_b)
        globally_used_blue_coords.update(used_mid_b)

        # --- オレンジボール検出と描画 ---
        current_detected_orange_balls_info = []
        # Renamed to avoid conflict with loop's current_time
        current_time_orange_detection = time.time()
        if orange_centers_coords:
            for (cx, cy) in orange_centers_coords:
                if debug_view_enabled:
                    cv2.circle(display_frame, (cx, cy), 15, (0, 165, 255), 3)
                if m_scale is not None:
                    cx_m = (cx - center_x) * m_scale
                    cy_m = (center_y - cy) * m_scale
                    current_detected_orange_balls_info.append(
                        {"pos": (round(cx_m, 2), round(cy_m, 2))})
                else:
                    pass
            last_detected_orange_balls_info = current_detected_orange_balls_info
            last_detection_time_orange = current_time_orange_detection
        else:
            if current_time_orange_detection - last_detection_time_orange <= BALL_COMPENSATION_TIME:
                current_detected_orange_balls_info = last_detected_orange_balls_info
            else:
                last_detected_orange_balls_info = []
                current_detected_orange_balls_info = []

        # --- UDP送信データの準備と送信 ---
        yellow_robots_payload = {}
        blue_robots_payload = {}

        for robot_info in all_detected_robots_info:
            # 'pos' が存在し、かつ (None, None) でないことを確認 (キャリブレーション済みの場合のみ)
            if 'pos' in robot_info and robot_info['pos'] != (None, None) and robot_info['pos'][0] is not None:
                robot_id_str = str(robot_info['id'])
                # ペイロードには角度と位置情報のみを含める
                robot_data_for_payload = {
                    "angle": robot_info['angle'],
                    "pos": robot_info['pos']
                }

                if robot_info['team'] == 'yellow':
                    yellow_robots_payload[robot_id_str] = robot_data_for_payload
                elif robot_info['team'] == 'blue':
                    blue_robots_payload[robot_id_str] = robot_data_for_payload

        vision_data = {
            "timestamp": current_time,  # シミュレーションの例に合わせてタイムスタンプを追加
            "fps": round(fps, 2),
            "is_calibrated": m_scale is not None,
            "yellow_robots": yellow_robots_payload,
            "blue_robots": blue_robots_payload,
            "orange_balls": current_detected_orange_balls_info
        }
        udp_sender.send_data(vision_data)

        # --- デバッグ表示 ---
        if debug_view_enabled:
            print(vision_data)
            # --- 画面表示情報の描画 (キャリブレーション、FPSなど) ---
            calib_status_text = ""
            calib_color = (255, 255, 255)
            if m_scale is None:
                points_count = len(calib_points)
                if points_count == 0:
                    calib_status_text = f"Calibrate: Click 1st point ({REAL_CALIBRATION_DISTANCE_M}m)"
                    calib_color = (0, 255, 255)
                elif points_count == 1:
                    calib_status_text = f"Calibrate: Click 2nd point ({REAL_CALIBRATION_DISTANCE_M}m)"
                    calib_color = (0, 165, 255)
            else:
                calib_status_text = f"Scale: 1 pixel = {m_scale:.6f} m (Click to recalibrate)"
                calib_color = (0, 255, 0)

            cv2.putText(display_frame, calib_status_text, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, calib_status_text, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, calib_color, 1, cv2.LINE_AA)

            if m_scale is None:
                for i, pt in enumerate(calib_points):
                    point_color = (0, 255, 255) if i == 0 else (0, 165, 255)
                    cv2.circle(display_frame, pt, 5, point_color, -1)
                    pt_text = f"P{i+1}:({pt[0]},{pt[1]})"
                    cv2.putText(display_frame, pt_text, (
                        pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(display_frame, pt_text, (
                        pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(display_frame, fps_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, fps_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            ball_text_display = "Orange Ball"
            if m_scale is not None and current_detected_orange_balls_info:
                ball_pos_m = current_detected_orange_balls_info[0]['pos']
                ball_text_display += f" | M:({ball_pos_m[0]:.2f},{ball_pos_m[1]:.2f})"
            elif not current_detected_orange_balls_info:
                ball_text_display += ": Not Detected"

            cv2.putText(display_frame, ball_text_display, (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, ball_text_display, (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            status_text_size, _ = cv2.getTextSize(
                mask_status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            status_text_pos = (width - status_text_size[0] - 10, 25)
            cv2.putText(display_frame, mask_status_text, status_text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, mask_status_text, status_text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, mask_status_color, 1, cv2.LINE_AA)

            mouse_x, mouse_y = calibration.get_mouse_position()
            if 0 <= mouse_x < width and 0 <= mouse_y < height:
                hsv_value = hsv[mouse_y, mouse_x]
                mouse_hsv_text = f"HSV @({mouse_x},{mouse_y}): {hsv_value}"
                text_size_hsv, _ = cv2.getTextSize(
                    mouse_hsv_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                text_x_hsv = width - text_size_hsv[0] - 10
                text_y_hsv = height - 10
                # Avoid overlap with ball text
                if text_y_hsv <= height - 10 + text_size_hsv[1]:
                    text_y_hsv = height - 30
                cv2.putText(display_frame, mouse_hsv_text, (text_x_hsv, text_y_hsv),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(display_frame, mouse_hsv_text, (text_x_hsv, text_y_hsv),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == MASK_TOGGLE_KEY:
            mask_view_enabled = not mask_view_enabled
            print(
                f"Mask view {'enabled' if mask_view_enabled else 'disabled'}")
        elif key == DEBUG_TOGGLE_KEY:
            debug_view_enabled = not debug_view_enabled
            print(
                f"Debug view {'enabled' if debug_view_enabled else 'disabled'}")

    cap.release()
    cv2.destroyAllWindows()
    udp_sender.close()
    print("Application finished.")


if __name__ == "__main__":
    main()
