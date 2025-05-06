import cv2
import numpy as np
from config import COLOR_RANGES  # 設定ファイルから色範囲をインポート


def get_centroids(hsv, color_name, min_area=100):
    """
    指定された色の領域の中心座標(重心)をリストで返す。

    Args:
        hsv (numpy.ndarray): HSV色空間の画像。
        color_name (str): 検出したい色の名前 ('red', 'green', 'blue', 'yellow', 'orange')。
                          'red'の場合は'red'と'red2'の範囲を結合して処理する。
        min_area (int): 検出する領域の最小面積（ピクセル）。

    Returns:
        list: 検出された領域の重心座標のリスト [(x1, y1), (x2, y2), ...]
    """
    masks = []
    if color_name == "red":
        if "red" in COLOR_RANGES:
            masks.append(cv2.inRange(hsv, *COLOR_RANGES["red"]))
        if "red2" in COLOR_RANGES:
            masks.append(cv2.inRange(hsv, *COLOR_RANGES["red2"]))
        # 複数のマスクがあればOR演算で結合、なければ空マスク
        if len(masks) > 1:
            mask = cv2.bitwise_or(masks[0], masks[1])
        elif len(masks) == 1:
            mask = masks[0]
        else:
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)  # 色定義がない場合は空マスク
    else:
        if color_name not in COLOR_RANGES:
            # print(f"Warning: Color '{color_name}' not found in COLOR_RANGES.") # 頻繁に出る警告は抑制
            return []  # 未定義の色なら空リストを返す
        mask = cv2.inRange(hsv, *COLOR_RANGES[color_name])

    # モルフォロジー演算（ノイズ除去）
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # オープニング（小さいノイズ除去）
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # クロージング（穴埋め）

    # 輪郭検出
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭から重心を計算
    result = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:  # 最小面積より大きい領域のみを対象
            M = cv2.moments(contour)
            if M["m00"] != 0:  # ゼロ除算回避
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                result.append((cx, cy))
    return result


def generate_combined_mask(hsv):
    """
    設定ファイル(config.py)で定義されたすべての色範囲に基づいてマスクを生成し、
    それらを結合した単一のマスク画像を返す。

    Args:
        hsv (numpy.ndarray): HSV色空間の画像。

    Returns:
        numpy.ndarray: 検出されたすべての色の領域を示す二値マスク画像。
    """
    all_masks = []
    kernel = np.ones((5, 5), np.uint8)  # get_centroidsと同じカーネルを使用

    # 各色についてマスクを生成
    for color_name, ranges in COLOR_RANGES.items():
        if color_name == "red2":  # 'red'でまとめて処理するのでスキップ
            continue

        if color_name == "red":
            # 赤色の場合は2つの範囲を考慮
            mask1 = cv2.inRange(hsv, *COLOR_RANGES["red"])
            if "red2" in COLOR_RANGES:
                mask2 = cv2.inRange(hsv, *COLOR_RANGES["red2"])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = mask1  # red2が定義されていなければmask1のみ
        else:
            mask = cv2.inRange(hsv, *ranges)

        # ノイズ除去 (get_centroidsと同様の処理)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        all_masks.append(mask)

    if not all_masks:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)  # マスクがなければ黒画像

    # 全てのマスクをOR演算で結合
    combined_mask = all_masks[0]
    for i in range(1, len(all_masks)):
        combined_mask = cv2.bitwise_or(combined_mask, all_masks[i])

    return combined_mask
