import time  # timeモジュールを追加
import math
import numpy as np
import cv2


COLOR_RANGES = {
    "red": ((0, 80, 80), (10, 130, 255)),
    # 赤色の後半範囲を追加 (HSVのH値は0-180で表現されるため、赤は0付近と180付近の両方に現れる可能性がある)
    "red2": ((170, 80, 80), (180, 130, 255)),
    "yellow": ((20, 80, 100), (40, 200, 255)),
    "green": ((30, 60, 150), (60, 155, 255)),
    "blue": ((85, 80, 80), (110, 255, 255)),
    "orange": ((10, 150, 150), (20, 255, 255)),
}


# 面積閾値を引数として受け取るように修正
def get_centroids(hsv, color_name, min_area=1000):
    masks = []
    if color_name == "red":
        # 赤色の前半と後半の両方の範囲でマスクを作成
        masks.append(cv2.inRange(hsv, *COLOR_RANGES["red"]))
        if "red2" in COLOR_RANGES:  # red2の定義があるか確認
            masks.append(cv2.inRange(hsv, *COLOR_RANGES["red2"]))
        mask = cv2.bitwise_or(
            # マスクを結合
            *masks) if masks else np.zeros(hsv.shape[:2], dtype=np.uint8)
    else:
        # 指定された色名がCOLOR_RANGESにあるかチェック
        if color_name not in COLOR_RANGES:
            print(f"Warning: Color '{color_name}' not found in COLOR_RANGES.")
            return []
        mask = cv2.inRange(hsv, *COLOR_RANGES[color_name])

    # モルフォロジー変換でノイズを除去し、対象を連結する
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # オープニング
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # クロージング

    # 輪郭検出の方が、connectedComponentsWithStatsより単純な重心検出に適している場合がある
    # こちらの方法で面積フィルタリングと重心計算を行う
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # 輪郭のモーメントを計算
            M = cv2.moments(contour)
            # モーメントから重心を計算
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                result.append((cx, cy))
    # ConnectedComponentsWithStats版を残しておきたい場合は、上記の輪郭検出コードと入れ替えるか選択
    # 以下はConnectedComponentsWithStats版
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    # result = []
    # for i in range(1, num_labels):  # ラベル0は背景なのでスキップ
    #     area = stats[i, cv2.CC_STAT_AREA]
    #     if area > min_area:
    #         cx, cy = centroids[i]
    #         result.append((int(cx), int(cy)))

    return result


def distance(p1, p2):
    # タプルを受け取れるように修正
    p1_arr = np.array(p1)
    p2_arr = np.array(p2)
    return np.linalg.norm(p1_arr - p2_arr)


def find_closest(p, points):
    # pointsが空の場合はNoneを返す
    if not points:
        return None
    return min(points, key=lambda g: distance(p, g))


def find_middle(p1, p2, middle_candidates):
    # middle_candidatesが空の場合はNoneを返す
    if not middle_candidates:
        return None

    # ロボットマーカーの中間色検出ロジック
    # 線分のベクトル
    v_p1_p2 = np.array(p2) - np.array(p1)
    dist_p1_p2_sq = np.sum(v_p1_p2**2)

    if dist_p1_p2_sq == 0:  # p1とp2が同じ点の場合は中間を見つけられない
        return None

    for m in middle_candidates:
        v_p1_m = np.array(m) - np.array(p1)

        # mが線分p1-p2上にあるかを内積で判定
        dot_product = np.dot(v_p1_m, v_p1_p2)

        # mがp1とp2の間にある条件: 内積 > 0 かつ 内積 < 線分長^2
        # dot_product = |v_p1_m| * |v_p1_p2| * cos(theta)
        # mが線分上にあれば cos(theta) = 1
        # この内積を |v_p1_p2|^2 で割ると、mがp1からどれだけp2方向に進んだかの割合になる (0～1の間なら線分上)
        # ただし、ここでは厳密な線分上ではなく、少し広めに判定したい
        # mをp1-p2直線に射影した点が線分p1-p2内にあるかをチェック
        projection_ratio = dot_product / dist_p1_p2_sq

        # 射影点が線分p1-p2の範囲内にあるか (0～1の少し外側まで許容)
        if 0 < projection_ratio < 1:  # 厳密に0と1を含めない（p1, p2自体ではない中間点を探すため）
            # さらに、mが直線p1-p2から大きく離れていないかをチェック
            # 射影点 p_prime = p1 + projection_ratio * v_p1_p2
            p1_arr = np.array(p1)
            p_prime = p1_arr + projection_ratio * v_p1_p2
            distance_from_line = np.linalg.norm(np.array(m) - p_prime)

            # 許容できる直線からの距離 (調整可能)
            # 例: 線分長の10%以内など
            if distance_from_line < distance(p1, p2) * 0.2:  # 0.2は許容度、調整してください
                return m

    return None


# マウスイベントの座標を保持するグローバル変数
mouse_x, mouse_y = -1, -1
# マウスイベントコールバック関数


def mouse_hsv_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y


def detect_robot_from_reds(red_list, green_list, middle_list, middle_name, frame, hsv_frame, min_arrow_length=50):
    height, width = frame.shape[:2]
    center_x = width // 2
    center_y = height // 2

    detected_robots = []  # 検出したロボット情報をリストで返す

    # 処理済みの緑マーカーのインデックスを保持
    used_green_indices = set()
    # 処理済みのミドルマーカーのインデックスを保持
    used_middle_indices = set()

    for r_idx, r in enumerate(red_list):
        # すでに他のロボットで使われた赤マーカーはスキップ（今回は赤を基点とするので不要かもしれないが、念のため）
        # closest_g = find_closest(r, green_list)
        # closest_m = find_closest(r, middle_list) # 赤に近いミドルマーカーも探す可能性がある

        # 赤マーカーに最も近い緑マーカーを探す (未使用のものから)
        available_greens = [(g, i) for i, g in enumerate(
            green_list) if i not in used_green_indices]
        closest_g_info = min(available_greens, key=lambda item: distance(
            r, item[0])) if available_greens else None

        if closest_g_info is None:
            continue

        g, g_idx = closest_g_info

        if distance(r, g) < min_arrow_length:
            continue

        # rとgの間に最も近いミドルマーカーを探す (未使用のものから)
        available_middles = [(m, i) for i, m in enumerate(
            middle_list) if i not in used_middle_indices]
        # find_middleは座標リストを期待
        m_info = find_middle(r, g, [item[0] for item in available_middles])

        if m_info is None:
            continue

        m = m_info
        # 使用したミドルマーカーのインデックスを見つける
        m_idx = -1
        for i, candidate in enumerate(middle_list):
            if np.array_equal(candidate, m) and i not in used_middle_indices:
                m_idx = i
                break

        if m_idx == -1:  # 見つからなかったらおかしいが念のため
            continue

        # この組み合わせ(r, m, g)でロボットを検出したとみなす
        # 使用したマーカーを追跡リストに追加
        # used_red_indices.add(r_idx) # 赤マーカーはループの基点なので追跡は不要
        used_green_indices.add(g_idx)
        used_middle_indices.add(m_idx)

        # ロボットの中心座標を計算 (3点の重心)
        cx = int((r[0] + m[0] + g[0]) / 3)
        cy = int((r[1] + m[1] + g[1]) / 3)

        # 中心基準の座標に変換 (表示や外部出力用)
        cx_centered = cx - center_x
        cy_centered = cy - center_y

        # 向きの計算 (赤 -> 緑 のベクトルを使用)
        # atan2の引数の順番は (dy, dx)
        # OpenCVのY軸は下向きが正なので、数学的な角度とは反時計回りが逆になることに注意
        # 多くのロボット制御システムでは、X+方向を0度とし、反時計回りを正とするため、
        # Y軸の向きを考慮して角度を調整する必要がある場合がある。
        # ここでは単純に atan2 の結果を返す (Y下向き正)
        # 画面表示としては、赤から緑への矢印方向が向き
        angle_rad = math.atan2(g[1] - r[1], g[0] - r[0])
        angle_deg = math.degrees(angle_rad)

        # 角度を0〜360度の範囲に正規化 (任意)
        if angle_deg < 0:
            angle_deg += 360

        label = f"{middle_name.upper()} robot"

        # --- デバッグ/表示用の描画 ---
        # マーカーの円を描画 (BGR)
        cv2.circle(frame, r, 8, (0, 0, 255), -1)  # 赤
        if middle_name == "yellow":
            cv2.circle(frame, m, 8, (0, 255, 255), -1)  # 黄色
        elif middle_name == "blue":
            cv2.circle(frame, m, 8, (255, 0, 0), -1)  # 青
        cv2.circle(frame, g, 8, (0, 255, 0), -1)  # 緑

        # 向きを示す矢印を描画
        # 矢印の始点を重心、終点を向きベクトル方向の少し先にする方が向きが分かりやすいかも
        # 向きベクトル: (g[0]-r[0], g[1]-r[1])
        arrow_length = 40  # 矢印の長さ（ピクセル）
        # ベクトルを正規化して長さを掛ける
        v_x = g[0] - r[0]
        v_y = g[1] - r[1]
        vec_len = math.sqrt(v_x**2 + v_y**2)
        if vec_len > 0:
            unit_v_x = v_x / vec_len
            unit_v_y = v_y / vec_len
            arrow_end_x = int(cx + unit_v_x * arrow_length)
            arrow_end_y = int(cy + unit_v_y * arrow_length)
            cv2.arrowedLine(frame, (cx, cy), (arrow_end_x,
                            arrow_end_y), (255, 0, 255), 2)  # マゼンタの矢印

        # ロボットの中心に円を描画
        cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)  # 中心に黒点

        # テキストで情報表示
        cv2.putText(
            frame,
            f"{label} @({cx_centered},{cy_centered}) {angle_deg:.1f} deg",
            (cx + 15, cy - 10),  # 中心から少しずらして表示
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,  # フォントサイズを調整
            (0, 0, 0),  # 黒
            1,
            cv2.LINE_AA
        )
        # --- 描画終わり ---

        # 検出結果をリストに追加
        detected_robots.append({
            "type": label,
            "center_pixel": (cx, cy),
            "center_relative": (cx_centered, cy_centered),
            "orientation_deg": angle_deg,
            "markers": {"red": r, "middle": m, "green": g}
        })

    return detected_robots  # 検出したロボットのリストを返す


def main():
    # カメラデバイスのインデックス (通常は0) を指定
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # ウィンドウを作成し、マウスイベントコールバックを設定
    window_name = "Detection Results"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_hsv_callback)

    # FPS計算用の変数
    prev_time = time.time()

    # マウス位置のHSV表示用の変数
    mouse_hsv_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)

        # FPS計算
        current_time = time.time()
        delta_time = current_time - prev_time
        prev_time = current_time
        fps = 1 / delta_time if delta_time > 0 else 0

        # フレームサイズを小さくすると処理が速くなる場合があります
        # frame = cv2.resize(frame, (640, 480))
        height, width = frame.shape[:2]  # 中心座標計算用にフレームサイズを取得
        center_x = width // 2
        center_y = height // 2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 各色の重心を取得
        # ロボットマーカーは比較的大きいため、閾値1000を使用
        robot_marker_min_area = 800  # ロボットマーカーの最小面積閾値 (調整してください)
        red_centers = get_centroids(hsv, "red", min_area=robot_marker_min_area)
        green_centers = get_centroids(
            hsv, "green", min_area=robot_marker_min_area)
        yellow_centers = get_centroids(
            hsv, "yellow", min_area=robot_marker_min_area)
        blue_centers = get_centroids(
            hsv, "blue", min_area=robot_marker_min_area)

        # オレンジボールはサイズが小さい可能性があるため、面積閾値を小さく設定
        orange_ball_min_area = 100  # オレンジボールの最小面積閾値 (調整してください)
        orange_centers = get_centroids(
            hsv, "orange", min_area=orange_ball_min_area)

        # ロボットの検出と描画
        # 検出結果は detect_robot_from_reds から返される
        yellow_robots = detect_robot_from_reds(
            red_centers, green_centers, yellow_centers, "yellow", frame, hsv)
        blue_robots = detect_robot_from_reds(
            red_centers, green_centers, blue_centers, "blue", frame, hsv)

        # 検出されたロボット情報の表示 (オプション)
        # print(f"Detected Yellow Robots: {len(yellow_robots)}")
        # print(f"Detected Blue Robots: {len(blue_robots)}")
        # for robot in yellow_robots + blue_robots:
        #      print(robot)

        # オレンジボールの検出位置に円とテキストを描画
        for (cx, cy) in orange_centers:
            # 中心基準の座標に変換 (表示用)
            cx_centered = cx - center_x
            cy_centered = cy - center_y

            cv2.circle(frame, (cx, cy), 15, (0, 165, 255), 3)  # オレンジ色の円 (BGR)
            cv2.putText(frame, f"Orange Ball @({cx_centered},{cy_centered})",
                        (cx + 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # --- FPS表示 ---
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2, cv2.LINE_AA)  # 黒縁
        cv2.putText(frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 1, cv2.LINE_AA)  # 白文字

        # --- マウスカーソル位置のHSV表示 ---
        # マウス座標が有効かつフレームサイズ内にあるかチェック
        if 0 <= mouse_x < width and 0 <= mouse_y < height:
            # HSV値を取得
            hsv_value = hsv[mouse_y, mouse_x]
            mouse_hsv_text = f"Mouse @({mouse_x},{mouse_y}) HSV: {hsv_value}"
        else:
            # マウスがウィンドウ外に出たら表示をクリアまたは非表示
            mouse_hsv_text = "Mouse outside window"  # または "" にする

        # HSV情報を画面に描画 (例: 右下)
        # テキストのサイズを取得して、右下に配置
        text_size, _ = cv2.getTextSize(
            mouse_hsv_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        text_x = width - text_size[0] - 10  # 右端から10ピクセル内側
        text_y = height - 10  # 下端から10ピクセル上

        cv2.putText(frame, mouse_hsv_text, (text_x, text_y),
                    # 黒縁
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, mouse_hsv_text, (text_x, text_y),
                    # 白文字
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 結果を表示
        cv2.imshow(window_name, frame)  # 設定したウィンドウ名を使用

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソースの解放
    cap.release()
    cv2.destroyAllWindows()


# スクリプトが直接実行された場合にmain関数を呼び出す
if __name__ == "__main__":
    main()
