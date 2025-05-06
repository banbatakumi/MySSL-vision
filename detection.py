import cv2
import numpy as np
import math
from utils import distance, find_middle  # utilsからヘルパー関数をインポート

# config からインポートする代わりに、引数で受け取るように変更
# from config import MIN_ARROW_LENGTH_PIXELS


def detect_robot_from_reds(red_list, green_list, middle_list, middle_name,
                           frame_to_draw, cm_scale,
                           min_arrow_length=50):
    """
    赤、緑、中間色(黄or青)のマーカーの重心リストからロボットを検出し、
    指定されたフレームに結果を描画し、検出情報を返す。

    Args:
        red_list (list): 赤色マーカーの重心リスト [(x, y), ...]
        green_list (list): 緑色マーカーの重心リスト [(x, y), ...]
        middle_list (list): 中間色マーカーの重心リスト [(x, y), ...]
        middle_name (str): 中間色の名前 ('yellow' or 'blue')
        frame_to_draw (numpy.ndarray): 検出結果を描画するフレーム。
        cm_scale (float or None): 1ピクセルあたりのCM数。Noneの場合はCM座標計算と情報返却は行わない。
        min_arrow_length (int): 検出に必要な赤-緑間の最小距離(ピクセル)。

    Returns:
        list: 検出されたロボットの情報リスト。各要素は以下の辞書形式。
              {'type': str, 'angle': float, 'center_relative_cm': tuple(float, float)}
              cm_scaleがNoneの場合は空リストを返す。
    """
    height, width = frame_to_draw.shape[:2]
    center_x = width // 2
    center_y = height // 2

    detected_robots_info = []  # 送信用のロボット情報リスト

    # 各マーカーが既に使用されたかを追跡するためのセット
    used_red_indices = set()
    used_green_indices = set()
    used_middle_indices = set()

    # 赤マーカーを基準にロボットを探す
    for r_idx, r in enumerate(red_list):
        if r_idx in used_red_indices:
            continue  # この赤マーカーは既に使用済み

        # --- 最も近い未使用の緑マーカーを探す ---
        closest_g_info = None
        min_dist_g = float('inf')
        for g_idx, g in enumerate(green_list):
            if g_idx in used_green_indices:
                continue  # この緑マーカーは既に使用済み
            d = distance(r, g)
            # 最小距離条件もここでチェック
            if d >= min_arrow_length and d < min_dist_g:
                min_dist_g = d
                closest_g_info = (g, g_idx)  # (座標, インデックス)

        if closest_g_info is None:  # 条件に合う緑マーカーが見つからない
            continue

        g, g_idx = closest_g_info

        # --- 線分r-gの近くにある未使用の中間マーカーを探す ---
        potential_middles = []  # (座標, 元のインデックス) のリスト
        for m_idx, m in enumerate(middle_list):
            if m_idx not in used_middle_indices:
                potential_middles.append((m, m_idx))

        middle_coords_only = [item[0]
                              for item in potential_middles]  # find_middleには座標リストを渡す
        # allowed_dist_ratioはデフォルト値を使用
        best_m_coord = find_middle(r, g, middle_coords_only)

        if best_m_coord is None:  # 適切な中間マーカーが見つからなかった
            continue

        # 見つかった中間マーカーの元のインデックスを探す
        m_idx = -1
        for m_coord, original_idx in potential_middles:
            # numpy配列として比較
            if np.array_equal(m_coord, best_m_coord):
                m_idx = original_idx
                break

        if m_idx == -1:  # 基本的に起こらないはずだが念のため
            print("Warning: Could not find original index for the middle marker.")
            continue

        # --- ロボット検出成功 ---
        # 使用したマーカーを記録
        used_red_indices.add(r_idx)
        used_green_indices.add(g_idx)
        used_middle_indices.add(m_idx)

        m = best_m_coord  # 中間点の座標

        # ロボットの中心座標 (3点の平均)
        cx = int((r[0] + m[0] + g[0]) / 3)
        cy = int((r[1] + m[1] + g[1]) / 3)

        # 向きの計算 (赤 -> 緑のベクトル)
        angle_rad = math.atan2(g[1] - r[1], g[0] - r[0])  # Y軸下向きが正の座標系での角度
        angle_deg = math.degrees(angle_rad)
        # 角度を -180 から 180 の範囲に正規化 (一般的なロボット制御系に合わせる場合)
        angle_deg = (angle_deg + 180) % 360 - 180  # Example normalization

        robot_label = f"{middle_name.upper()} Robot"  # ラベル名を大文字に

        # --- CM単位の座標計算 ---
        cx_centered_cm, cy_centered_cm = None, None
        if cm_scale is not None:
            # 中心からの相対座標を計算 (画面中心を原点、X右向き正、Y上向き正)
            cx_centered_cm = (cx - center_x) * cm_scale
            cy_centered_cm = (center_y - cy) * cm_scale  # Y軸の向きを反転

        # --- 描画処理 (frame_to_draw に対して行う) ---
        # 各マーカーに円を描画
        cv2.circle(frame_to_draw, r, 8, (0, 0, 255), -1)  # 赤 BGR(Red)
        if middle_name == "yellow":
            cv2.circle(frame_to_draw, m, 8, (0, 255, 255), -1)  # 黄 BGR(Yellow)
        elif middle_name == "blue":
            cv2.circle(frame_to_draw, m, 8, (255, 0, 0), -1)  # 青 BGR(Blue)
        cv2.circle(frame_to_draw, g, 8, (0, 255, 0), -1)  # 緑 BGR(Green)

        # ロボットの中心から向きを示す矢印を描画
        arrow_length = 40
        # 矢印の向きは atan2 で計算した角度を使う
        arrow_start_x = int(cx - arrow_length * math.cos(angle_rad))
        arrow_start_y = int(cy - arrow_length * math.sin(angle_rad))
        arrow_end_x = int(cx + arrow_length * math.cos(angle_rad))
        arrow_end_y = int(cy + arrow_length * math.sin(angle_rad))
        cv2.arrowedLine(frame_to_draw, (arrow_start_x, arrow_start_y), (arrow_end_x, arrow_end_y),
                        (0, 0, 0), 5, tipLength=0.3)  # マゼンタ色の矢印

        # ロボットの中心に小さい黒丸を描画
        cv2.circle(frame_to_draw, (cx, cy), 5, (0, 0, 0), -1)

        # テキストで情報表示
        text = f"{robot_label} {angle_deg:.1f} deg"
        if cm_scale is not None:
            # CM座標も表示 (小数点以下1桁)
            text += f" | CM:({cx_centered_cm:.1f},{cy_centered_cm:.1f})"

        # テキスト描画 (白文字、黒縁取り)
        height, width = frame_to_draw.shape[:2]
        text_pos = text_pos = (10, height - 35)
        if middle_name == "yellow":
            text_pos = (10, height - 60)  # テキストの表示位置
        cv2.putText(frame_to_draw, text, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)  # 黒縁
        cv2.putText(frame_to_draw, text, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)  # 白文字
        # --- 描画終わり ---

        # --- 検出情報の格納 (cm_scaleが有効な場合のみ) ---
        if cm_scale is not None:
            robot_info = {
                "type": robot_label,
                "angle": round(angle_deg, 2),  # 小数点以下2桁で丸める
                "pos": (
                    round(cx_centered_cm, 2), round(cy_centered_cm, 2))  # 小数点以下2桁
            }
            detected_robots_info.append(robot_info)

    return detected_robots_info  # 検出されたロボットの情報リストを返す
