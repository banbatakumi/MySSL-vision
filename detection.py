import cv2
import numpy as np
import math
from itertools import combinations
from utils import distance  # utils.pyにdistance関数があると仮定

# utils.py に find_middle_point がない場合、ここで定義するか utils.py に追加
# def distance(p1, p2):
#     return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def _find_middle_point(p1, p2):  # 2点の中点
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


BGR_COLORS = {
    "magenta": (255, 0, 255),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "orange": (0, 165, 255)
}


def _calculate_pair_info(p1, p2, min_separation):
    """2点の中点と距離を計算。距離が最小分離距離未満ならNoneを返す。"""
    dist = distance(p1, p2)
    if dist < min_separation:
        return None, -1
    mid_x = (p1[0] + p2[0]) / 2
    mid_y = (p1[1] + p2[1]) / 2
    return (mid_x, mid_y), dist


def detect_fivel_marker_robots(
    all_magenta_coords_list,
    all_green_coords_list,
    middle_marker_coords_list,
    middle_marker_name,  # "yellow" or "blue"
    frame_to_draw,
    m_scale,  # cm_scale -> m_scale
    # Sets of (x,y) tuples, these are markers already used
    globally_used_magenta_coords_set,
    globally_used_green_coords_set,
    globally_used_middle_coords_set,  # For this specific middle_marker_name
    debug_view_enabled,
    # Config params
    max_dist_to_middle,
    min_pair_dist_diff,
    min_marker_separation_in_pair
):
    """
    5点マーカー(中央1 + マゼンタ2 + 緑2)からロボットを検出する。
    ロボットID: 0 (前マゼンタ), 1 (前緑)
    チーム: middle_marker_name (yellow or blue)
    ロボットの中心: 中央マーカーの重心
    """
    height, width = frame_to_draw.shape[:2]
    screen_center_x = width // 2
    screen_center_y = height // 2

    detected_robots_info_list = []
    newly_confirmed_used_magenta_coords = set()
    newly_confirmed_used_green_coords = set()
    newly_confirmed_used_middle_coords = set()

    # 1. フィルタリングされた中間マーカー候補
    available_middle_markers = [
        m_coord for m_coord in middle_marker_coords_list if tuple(m_coord) not in globally_used_middle_coords_set
    ]

    for m_center_coord in available_middle_markers:
        if tuple(m_center_coord) in newly_confirmed_used_middle_coords:
            continue

        potential_magentas = []
        for m_coord in all_magenta_coords_list:
            if tuple(m_coord) not in globally_used_magenta_coords_set and \
               tuple(m_coord) not in newly_confirmed_used_magenta_coords and \
               distance(m_center_coord, m_coord) < max_dist_to_middle:
                potential_magentas.append(m_coord)

        potential_greens = []
        for g_coord in all_green_coords_list:
            if tuple(g_coord) not in globally_used_green_coords_set and \
               tuple(g_coord) not in newly_confirmed_used_green_coords and \
               distance(m_center_coord, g_coord) < max_dist_to_middle:
                potential_greens.append(g_coord)

        if len(potential_magentas) < 2 or len(potential_greens) < 2:
            continue

        # best_robot_candidate = None # This variable was not used

        for m_pair in combinations(potential_magentas, 2):
            m1_coord, m2_coord = m_pair

            if tuple(m1_coord) in newly_confirmed_used_magenta_coords or \
               tuple(m2_coord) in newly_confirmed_used_magenta_coords:
                continue

            mid_magenta_coord, dist_magenta_pair = _calculate_pair_info(
                m1_coord, m2_coord, min_marker_separation_in_pair)
            if mid_magenta_coord is None:
                continue

            for g_pair in combinations(potential_greens, 2):
                g1_coord, g2_coord = g_pair

                if tuple(g1_coord) in newly_confirmed_used_green_coords or \
                   tuple(g2_coord) in newly_confirmed_used_green_coords:
                    continue

                mid_green_coord, dist_green_pair = _calculate_pair_info(
                    g1_coord, g2_coord, min_marker_separation_in_pair)
                if mid_green_coord is None:
                    continue

                robot_id = -1
                front_mid_coord, rear_mid_coord = None, None
                current_pair_dist_diff = abs(
                    dist_magenta_pair - dist_green_pair)

                if current_pair_dist_diff < min_pair_dist_diff:
                    continue

                if dist_magenta_pair > dist_green_pair:
                    robot_id = 0
                    front_mid_coord = mid_magenta_coord
                    rear_mid_coord = mid_green_coord
                elif dist_green_pair > dist_magenta_pair:
                    robot_id = 1
                    front_mid_coord = mid_green_coord
                    rear_mid_coord = mid_magenta_coord
                else:
                    continue

                robot_center_x = int(m_center_coord[0])
                robot_center_y = int(m_center_coord[1])
                robot_center_coord = (robot_center_x, robot_center_y)

                angle_rad = math.atan2(
                    front_mid_coord[1] - rear_mid_coord[1], front_mid_coord[0] - rear_mid_coord[0])
                angle_deg = math.degrees(angle_rad)
                angle_deg = (angle_deg + 180) % 360 - 180

                current_robot_used_magentas = {
                    tuple(m1_coord), tuple(m2_coord)}
                current_robot_used_greens = {tuple(g1_coord), tuple(g2_coord)}
                current_robot_used_middle = {tuple(m_center_coord)}

                newly_confirmed_used_magenta_coords.update(
                    current_robot_used_magentas)
                newly_confirmed_used_green_coords.update(
                    current_robot_used_greens)
                newly_confirmed_used_middle_coords.update(
                    current_robot_used_middle)

                cx_centered_m, cy_centered_m = None, None  # _cm -> _m
                if m_scale is not None:  # cm_scale -> m_scale
                    cx_centered_m = (  # _cm -> _m
                        robot_center_x - screen_center_x) * m_scale  # cm_scale -> m_scale
                    cy_centered_m = (screen_center_y -  # _cm -> _m
                                     robot_center_y) * m_scale  # cm_scale -> m_scale

                if debug_view_enabled:
                    cv2.circle(frame_to_draw, tuple(map(int, m1_coord)),
                               7, BGR_COLORS["magenta"], -1)
                    cv2.circle(frame_to_draw, tuple(map(int, m2_coord)),
                               7, BGR_COLORS["magenta"], -1)
                    cv2.circle(frame_to_draw, tuple(map(int, g1_coord)),
                               7, BGR_COLORS["green"], -1)
                    cv2.circle(frame_to_draw, tuple(map(int, g2_coord)),
                               7, BGR_COLORS["green"], -1)
                    cv2.circle(frame_to_draw, tuple(map(int, m_center_coord)),
                               8, BGR_COLORS[middle_marker_name], -1)

                    arrow_length = 30
                    arrow_end_x = int(
                        robot_center_x + arrow_length * math.cos(angle_rad))
                    arrow_end_y = int(
                        robot_center_y + arrow_length * math.sin(angle_rad))
                    cv2.arrowedLine(frame_to_draw, robot_center_coord, (arrow_end_x, arrow_end_y),
                                    (0, 0, 0), 3, tipLength=0.4)
                    cv2.circle(frame_to_draw, robot_center_coord,
                               5, (0, 0, 0), -1)

                    robot_label_text = f"ID:{robot_id} ({middle_marker_name.upper()})"
                    info_text = f"{robot_label_text} {angle_deg:.1f}d"
                    if m_scale is not None:  # cm_scale -> m_scale
                        # CM -> M, _cm -> _m, .1f -> .2f
                        info_text += f" M:({cx_centered_m:.2f},{cy_centered_m:.2f})"

                    text_pos_y_offset = -20
                    cv2.putText(frame_to_draw, info_text, (robot_center_x, robot_center_y + text_pos_y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame_to_draw, info_text, (robot_center_x, robot_center_y + text_pos_y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                robot_data = {
                    "id": robot_id,
                    "team": middle_marker_name,
                    "angle": round(angle_deg, 2),
                    # _cm -> _m
                    "pos": (round(cx_centered_m, 2), round(cy_centered_m, 2)) if m_scale else (None, None)
                }
                if m_scale is None:  # cm_scale -> m_scale
                    robot_data.pop("pos", None)

                detected_robots_info_list.append(robot_data)

                goto_next_middle_marker = True
                break  # Found a robot for this middle marker, move to next middle marker

            if 'goto_next_middle_marker' in locals() and goto_next_middle_marker:
                del goto_next_middle_marker  # clean up temp variable
                break  # Move to the next m_center_coord

    return (detected_robots_info_list,
            newly_confirmed_used_magenta_coords,
            newly_confirmed_used_green_coords,
            newly_confirmed_used_middle_coords)
