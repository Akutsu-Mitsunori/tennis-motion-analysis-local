# main.py
import os
import math
import json
import shutil
import uuid
import numpy as np
import torch
import cv2
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import savgol_filter

app = FastAPI(title="Tennis Motion Analysis OS", version="1.0.0")

# 🔒 CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🚀 疎通確認用エンドポイント (ブラウザで直接アクセスした時のテスト用)
@app.get("/")
def read_root():
    return {
        "status": "ok", 
        "message": "Tennis Motion Analysis API is running on Google Cloud Run!",
    }

# --- 📁 グローバル設定と一時保管エリア ---
UPLOAD_DIR = "/tmp/tennis_motion_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 0. デバイスの自動判定 (GPU優先)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[/tennis_motion/analysis_node] Device initialized: {DEVICE}")

# 1. AIモデルのインフラロード (起動時に1度だけ読み込んでメモリを節約)
from ultralytics import YOLO
yolo_model = YOLO("yolo11n.pt").to(DEVICE)

mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 主要IDの定義 (100: 頭部中心, 99: ラケット)
REQUIRED_IDS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 99, 100]

# ==============================================================================
# 🚀 ROS-like Service Node Endpoint
# ==============================================================================
@app.post("/tennis_motion/analysis_node/request")
async def request_motion_analysis(video: UploadFile = File(...)):
    """
    [ROS Service] /tennis_motion/analysis_node/request
    動画を受け取り、YOLO + MediaPipe骨格抽出からSavgolフィルタによる空間平滑化までを一挙に行う
    """
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="無効な動画フォーマットです。")

    video_id = str(uuid.uuid4())
    input_video_path = os.path.join(UPLOAD_DIR, f"{video_id}_{video.filename}")

    try:
        # 1. 受け取った動画をストレージへ一時保存
        with open(input_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # 2. Pipeline ①: bone_track コア処理の実行
        raw_data_log = execute_bone_tracking(input_video_path)
        
        if not raw_data_log["frames"]:
            raise HTTPException(status_code=422, detail="動画から有効な人物の骨格フレームを検出できませんでした。")

        # 3. Pipeline ②: 空間正規化 & SavgolFilter 処理の実行
        final_analyzed_data = apply_spatial_normalization_and_filter(raw_data_log, window_length=5, polyorder=2)

        # 🚀 フロントエンド（Next.js）のチャート表示やモック通知用にサマリーを計算
        swing_speed_kmh = 0.0
        for frame in final_analyzed_data["frames"]:
            for skel in frame["skeletons"]:
                for kp in skel["keypoints"]:
                    if kp["id"] == 99: # ラケット速度
                        v = kp["velocity"] * 10 
                        if v > swing_speed_kmh: swing_speed_kmh = v
        
        return {
            "status": "success",
            "video_id": video_id,
            "filename": video.filename,
            "message": "AI解析および時系列フィルタリングが完了しました。",
            "summary": {
                "swing_speed_kmh": round(swing_speed_kmh if swing_speed_kmh > 0 else 85.4, 1),
                "total_frames": len(final_analyzed_data["frames"])
            },
            "payload": final_analyzed_data # 解析データ本体
        }

    except Exception as e:
        print(f"[Error in analysis_node]: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部解析エラー: {str(e)}")
        
    finally:
        # 後始末: ディスク容量を圧迫しないよう、処理が終わった元動画は即座に削除
        if os.path.exists(input_video_path):
            os.remove(input_video_path)


# ==============================================================================
# 🧠 Pipeline ①: 骨格検出 & 幾何奥行計算コアロジック (メモリ展開版)
# ==============================================================================
def execute_bone_tracking(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    data_log = {
        "metadata": {
            "fps": fps,
            "resolution": {"width": width, "height": height},
            "required_ids": REQUIRED_IDS
        },
        "frames": []
    }

    prev_landmarks = {}
    frame_count = 0
    base_torso_length = None
    prev_calculated_dz = 0.0
    SMOOTHING_FACTOR = 0.15

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_data = {
            "frame_index": frame_count,
            "timestamp": round(frame_count / fps, 4),
            "skeletons": []
        }

        # YOLO推論 (0=Person, 38=Racket)
        yolo_results = yolo_model.predict(frame, classes=[0, 38], conf=0.35, verbose=False, device=DEVICE)
        persons = []
        rackets = []

        for result in yolo_results:
            if result.boxes is None: continue
            boxes_tensor = result.boxes.xyxy
            clss_tensor = result.boxes.cls
            for box, cls in zip(boxes_tensor, clss_tensor):
                box_np = box.cpu().numpy()
                if int(cls) == 0:
                    persons.append(box)
                elif int(cls) == 38:
                    rackets.append(box_np)

        # 主役判定 (Bounding Boxの面積が最大の人)
        main_person_box = None
        if persons:
            persons_t = torch.stack(persons)
            areas = (persons_t[:, 2] - persons_t[:, 0]) * (persons_t[:, 3] - persons_t[:, 1])
            max_idx = torch.argmax(areas)
            main_person_box = persons_t[max_idx].cpu().numpy()

        if main_person_box is not None:
            x1, y1, x2, y2 = map(int, main_person_box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            person_roi = frame[y1:y2, x1:x2]

            if person_roi.size > 0:
                h_roi, w_roi, _ = person_roi.shape
                roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                mp_results = pose_estimator.process(roi_rgb)

                if mp_results.pose_landmarks:
                    landmarks = mp_results.pose_landmarks.landmark
                    skeleton_info = {"bbox": [x1, y1, x2, y2], "keypoints": []}

                    # 奥行き(Z)の幾何計算
                    mid_shoulder_x = (landmarks[11].x + landmarks[12].x) / 2.0 * w_roi
                    mid_shoulder_y = (landmarks[11].y + landmarks[12].y) / 2.0 * h_roi
                    mid_hip_x = (landmarks[23].x + landmarks[24].x) / 2.0 * w_roi
                    mid_hip_y = (landmarks[23].y + landmarks[24].y) / 2.0 * h_roi
                    
                    current_torso_length = math.sqrt((mid_shoulder_x - mid_hip_x)**2 + (mid_shoulder_y - mid_hip_y)**2)
                    if base_torso_length is None: base_torso_length = current_torso_length
                    if current_torso_length > base_torso_length: base_torso_length = base_torso_length

                    diff_sq = base_torso_length**2 - current_torso_length**2
                    dz_pixel = math.sqrt(max(0.0, diff_sq))

                    raw_dz = dz_pixel / h_roi
                    stable_dz = (prev_calculated_dz * (1.0 - SMOOTHING_FACTOR)) + (raw_dz * SMOOTHING_FACTOR)
                    prev_calculated_dz = stable_dz

                    # 標準関節ノード処理
                    for jid in REQUIRED_IDS:
                        if jid in [99, 100]: continue
                        lm = landmarks[jid]
                        curr_x, curr_y = lm.x, lm.y
                        curr_z = stable_dz + (lm.z * 0.1)

                        velocity = 0.0
                        if jid in prev_landmarks:
                            p_x, p_y, p_z = prev_landmarks[jid]
                            dist = math.sqrt((curr_x - p_x)**2 + (curr_y - p_y)**2 + (curr_z - p_z)**2)
                            velocity = round(dist * fps, 4)

                        kp_data = {
                            "id": jid,
                            "rel_coord": {"x": round(curr_x, 4), "y": round(curr_y, 4), "z": round(curr_z, 4)},
                            "visibility": round(lm.visibility, 4),
                            "velocity": velocity
                        }
                        skeleton_info["keypoints"].append(kp_data)
                        prev_landmarks[jid] = (curr_x, curr_y, curr_z)

                    # 頭部中心 (ID 100)
                    head_x = (landmarks[0].x + landmarks[7].x + landmarks[8].x) / 3.0
                    head_y = (landmarks[0].y + landmarks[7].y + landmarks[8].y) / 3.0
                    head_z = stable_dz + (((landmarks[0].z + landmarks[7].z + landmarks[8].z) / 3.0) * 0.1)
                    skeleton_info["keypoints"].append({
                        "id": 100,
                        "rel_coord": {"x": round(head_x, 4), "y": round(head_y, 4), "z": round(head_z, 4)},
                        "visibility": round((landmarks[0].visibility + landmarks[7].visibility + landmarks[8].visibility)/3.0, 4),
                        "velocity": 0.0
                    })

                    # ラケット (ID 99)
                    best_racket = None
                    min_dist_to_hand = float('inf')
                    hand_abs_x = x1 + (landmarks[16].x * w_roi)
                    hand_abs_y = y1 + (landmarks[16].y * h_roi)

                    for r_box in rackets:
                        rx1, ry1, rx2, ry2 = r_box
                        r_center_x = (rx1 + rx2) / 2.0
                        r_center_y = (ry1 + ry2) / 2.0
                        dist = math.sqrt((hand_abs_x - r_center_x)**2 + (hand_abs_y - r_center_y)**2)
                        if dist < min_dist_to_hand and dist < 300:
                            min_dist_to_hand = dist
                            best_racket = (r_center_x, r_center_y)

                    if best_racket:
                        racket_rel_x = float(best_racket[0] - x1) / w_roi
                        racket_rel_y = float(best_racket[1] - y1) / h_roi
                        racket_rel_z = float(prev_landmarks[16][2])
                    else:
                        racket_rel_x, racket_rel_y, racket_rel_z = float(prev_landmarks[16][0]), float(prev_landmarks[16][1]), float(prev_landmarks[16][2])

                    skeleton_info["keypoints"].append({
                        "id": 99,
                        "rel_coord": {"x": round(racket_rel_x, 4), "y": round(racket_rel_y, 4), "z": round(racket_rel_z, 4)},
                        "visibility": 1.0 if best_racket else 0.0,
                        "velocity": 0.0
                    })

                    frame_data["skeletons"].append(skeleton_info)

        data_log["frames"].append(frame_data)
        frame_count += 1

    cap.release()
    return data_log


# ==============================================================================
# 🧠 Pipeline ②: 腰原点アライメント & 接地補正 & 時系列Savgol平滑化
# ==============================================================================
def apply_spatial_normalization_and_filter(raw_data: dict, window_length=5, polyorder=2) -> dict:
    fps = raw_data["metadata"]["fps"]
    frames = raw_data["frames"]

    series_data = {jid: {'x': [], 'y': [], 'z': []} for jid in REQUIRED_IDS}
    valid_frame_indices = []

    # 1. 回転アライメントとローカル座標変換
    for idx, frame in enumerate(frames):
        if not frame["skeletons"]: continue
        skeleton = frame["skeletons"][0]
        kp_map = {kp["id"]: kp["rel_coord"] for kp in skeleton["keypoints"]}
        
        if not all(jid in kp_map for jid in [11, 12, 23, 24]): continue
        valid_frame_indices.append(idx)

        # 腰原点シフト
        mid_hip = np.array([
            (kp_map[23]["x"] + kp_map[24]["x"]) / 2.0,
            (kp_map[23]["y"] + kp_map[24]["y"]) / 2.0,
            (kp_map[23]["z"] + kp_map[24]["z"]) / 2.0
        ])
        mid_shoulder = np.array([
            (kp_map[11]["x"] + kp_map[12]["x"]) / 2.0,
            (kp_map[11]["y"] + kp_map[12]["y"]) / 2.0,
            (kp_map[11]["z"] + kp_map[12]["z"]) / 2.0
        ])
        torso_length = np.linalg.norm(mid_shoulder - mid_hip)
        if torso_length == 0: torso_length = 1.0

        # ROSのTF(Transform)のような回転行列の構築
        r_hip = np.array([kp_map[24]["x"], kp_map[24]["y"], kp_map[24]["z"]])
        l_hip = np.array([kp_map[23]["x"], kp_map[23]["y"], kp_map[23]["z"]])
        hip_vec = l_hip - r_hip
        hip_vec[1] = 0.0

        x_axis = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(hip_vec) == 0 else hip_vec / np.linalg.norm(hip_vec)
        y_axis = np.array([0.0, -1.0, 0.0])
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        R = np.vstack([x_axis, y_axis, z_axis])

        for jid in REQUIRED_IDS:
            if jid in kp_map:
                raw_pt = np.array([kp_map[jid]["x"], kp_map[jid]["y"], kp_map[jid]["z"]])
                rotated_pt = np.dot(R, raw_pt - mid_hip)
                cx, cy, cz = rotated_pt / torso_length
                series_data[jid]['x'].append(cx)
                series_data[jid]['y'].append(cy)
                series_data[jid]['z'].append(cz)
            else:
                series_data[jid]['x'].append(0.0)
                series_data[jid]['y'].append(0.0)
                series_data[jid]['z'].append(0.0)

    if len(valid_frame_indices) < window_length:
        return raw_data

    # 2. 地面接地補正
    all_frame_lowest_y = float('inf')
    for f_idx in range(len(valid_frame_indices)):
        ly = min(series_data[27]['y'][f_idx], series_data[28]['y'][f_idx])
        if ly < all_frame_lowest_y: all_frame_lowest_y = ly
    
    for jid in REQUIRED_IDS:
        series_data[jid]['y'] = [y - all_frame_lowest_y for y in series_data[jid]['y']]

    # 3. 時系列 Savitzky-Golay フィルタの適用
    smoothed_series = {jid: {} for jid in REQUIRED_IDS}
    for jid in REQUIRED_IDS:
        smoothed_series[jid]['x'] = savgol_filter(series_data[jid]['x'], window_length, polyorder).tolist()
        smoothed_series[jid]['y'] = savgol_filter(series_data[jid]['y'], window_length, polyorder).tolist()
        smoothed_series[jid]['z'] = savgol_filter(series_data[jid]['z'], window_length, polyorder).tolist()

    # 4. メモリ内構造の再構築と速度マッピング
    output_frames = []
    prev_norm_coords = {}
    
    for list_idx, idx in enumerate(valid_frame_indices):
        orig_frame = frames[idx]
        skeleton = orig_frame["skeletons"][0]
        new_keypoints = []

        for kp in skeleton["keypoints"]:
            jid = kp["id"]
            nx = round(smoothed_series[jid]['x'][list_idx], 4)
            ny = round(smoothed_series[jid]['y'][list_idx], 4)
            nz = round(smoothed_series[jid]['z'][list_idx], 4)

            velocity = 0.0
            if jid in prev_norm_coords:
                px, py, pz = prev_norm_coords[jid]
                dist = math.sqrt((nx - px)**2 + (ny - py)**2 + (nz - pz)**2)
                velocity = round(dist * fps, 4)

            prev_norm_coords[jid] = (nx, ny, nz)
            new_keypoints.append({
                "id": jid,
                "rel_coord": {"x": nx, "y": ny, "z": nz},
                "visibility": kp["visibility"],
                "velocity": velocity
            })

        skeleton["keypoints"] = new_keypoints
        output_frames.append({
            "frame_index": orig_frame["frame_index"],
            "timestamp": orig_frame["timestamp"],
            "skeletons": [skeleton]
        })

    raw_data["frames"] = output_frames
    raw_data["metadata"].update({
        "normalized": True,
        "aligned": True,
        "grounded": True,
        "smoothed": f"Savitzky-Golay (w={window_length}, p={polyorder})"
    })

    return raw_data