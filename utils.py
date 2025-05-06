import numpy as np
import math


def distance(p1, p2):
    """2点間のユークリッド距離を計算する"""
    p1_arr = np.array(p1)
    p2_arr = np.array(p2)
    return np.linalg.norm(p1_arr - p2_arr)


def find_closest(p, points):
    """点pに最も近い点をpointsリストから見つける"""
    if not points:
        return None
    return min(points, key=lambda target_point: distance(p, target_point))


def find_middle(p1, p2, middle_candidates, allowed_dist_ratio=0.3):
    """
    線分p1-p2の近くにある点をmiddle_candidatesから探す。
    最も線分に近い点を返す。

    Args:
        p1 (tuple): 線分の始点座標 (x, y)
        p2 (tuple): 線分の終点座標 (x, y)
        middle_candidates (list): 中間点候補の座標リスト [(x, y), ...]
        allowed_dist_ratio (float): 線分長に対する、線分からの許容最大距離の割合。
                                    この割合以内の距離にある候補点のみを対象とする。

    Returns:
        tuple: 最も条件に合う中間点の座標 (x, y)。見つからなければNone。
    """
    if not middle_candidates:
        return None

    v_p1_p2 = np.array(p2) - np.array(p1)
    dist_p1_p2 = distance(p1, p2)  # 線分の長さ

    if dist_p1_p2 == 0:  # p1とp2が同じ点の場合
        return None

    dist_p1_p2_sq = dist_p1_p2**2  # 長さの二乗を事前計算

    best_m = None
    min_dist_from_line = float('inf')

    for m in middle_candidates:
        v_p1_m = np.array(m) - np.array(p1)
        # ベクトルv_p1_mのv_p1_p2への射影の長さを計算
        dot_product = np.dot(v_p1_m, v_p1_p2)
        # 射影の位置が線分p1-p2の間にあるかチェック (0 <= t <= 1)
        projection_ratio = dot_product / dist_p1_p2_sq

        if 0 <= projection_ratio <= 1:
            p1_arr = np.array(p1)
            # 線分p1-p2上で点mに最も近い点(垂線の足)p_primeを計算
            p_prime = p1_arr + projection_ratio * v_p1_p2
            # 点mと垂線の足p_primeとの距離を計算 (これが線分からの距離)
            distance_from_line = np.linalg.norm(np.array(m) - p_prime)

            # 線分からの距離が許容範囲内かチェック
            if distance_from_line < allowed_dist_ratio * dist_p1_p2:
                # 最も線分に近い点を中間点として採用
                if distance_from_line < min_dist_from_line:
                    min_dist_from_line = distance_from_line
                    best_m = m

    return best_m
