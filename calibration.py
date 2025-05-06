import cv2
from utils import distance  # utilsからdistance関数をインポート
from config import REAL_CALIBRATION_DISTANCE_CM  # configから定数をインポート

# --- グローバル変数 ---
# これらの変数はマウスコールバック関数やメインループからアクセス・変更される
calibration_points = []  # クリックされたキャリブレーション点の座標リスト [(x1, y1), (x2, y2)]
cm_per_pixel_scale = None  # 計算されたスケール（1ピクセルあたりのCM数）
mouse_x, mouse_y = -1, -1  # マウスカーソルの現在の座標


def mouse_hsv_callback(event, x, y, flags, param):
    """
    マウスイベントを処理するコールバック関数。
    マウス移動で座標を更新し、左クリックでキャリブレーション点を記録・リセットする。
    """
    global mouse_x, mouse_y, calibration_points, cm_per_pixel_scale

    # マウス移動イベント: カーソル座標を更新
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

    # 左ボタンクリックイベント: キャリブレーション処理
    if event == cv2.EVENT_LBUTTONDOWN:
        # 既にスケール計算済みの場合、クリックでキャリブレーションをリセット
        if cm_per_pixel_scale is not None:
            calibration_points = []
            cm_per_pixel_scale = None
            print("Calibration reset by left click.")
            return  # リセットしたら以降の処理は不要

        # スケール未計算の場合、キャリブレーション点を追加
        if len(calibration_points) < 2:
            calibration_points.append((x, y))
            print(
                f"Calibration point {len(calibration_points)} recorded: ({x}, {y})")

            # 2点目が記録されたらスケールを計算
            if len(calibration_points) == 2:
                p1 = calibration_points[0]
                p2 = calibration_points[1]
                pixel_dist = distance(p1, p2)

                if pixel_dist > 0:
                    cm_per_pixel_scale = REAL_CALIBRATION_DISTANCE_CM / pixel_dist
                    print(
                        f"Scale calculated: 1 pixel = {cm_per_pixel_scale:.4f} cm")
                    # スケール計算成功後も calibration_points はクリアしないでおく
                    # (mainループで描画やリセット判定に使うため)
                else:
                    print(
                        "Error: Pixel distance is zero. Cannot calculate scale. Click to reset.")
                    # 計算失敗時はリセットを促すため、ポイントをクリアする
                    calibration_points = []
                    cm_per_pixel_scale = None  # スケールも無効化


def get_calibration_info():
    """現在のキャリブレーション関連の情報を返す"""
    return calibration_points, cm_per_pixel_scale


def get_mouse_position():
    """現在のマウスカーソル座標を返す"""
    return mouse_x, mouse_y
