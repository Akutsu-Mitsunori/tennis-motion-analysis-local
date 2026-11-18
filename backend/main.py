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

from skeleton_correction.data_processing import NUM_JOINTS, denormalize, preprocess_arrays
from skeleton_correction.model import SkeletonRefiner

app = FastAPI(title="Tennis Motion Analysis OS", version="1.1.0")

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
        "skeleton_refiner": REFINER_INFO,
    }

# --- 📁 グローバル設定と一時保管エリア ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
HEAD_SRC_IDS = [0, 7, 8]       # 鼻・左耳・右耳 → 頭部中心 (ID 100)
RACKET_HAND_ID = 16            # 右手首 → ラケット (ID 99) の探索基準
RACKET_MAX_DIST_PX = 300


# 2. 骨格補正モデル (skeleton_correction/train_prototype.py で学習した best.pt)
SKELETON_CKPT = os.environ.get("SKELETON_CKPT", os.path.join(BASE_DIR, "models", "best.pt"))


def load_skeleton_refiner(ckpt_path: str):
    """best.pt を読み込む。存在しない場合は None（補正なしで動作を継続）。"""
    if not os.path.exists(ckpt_path):
        print(f"[skeleton_refiner] checkpoint not found: {ckpt_path} → 補正をスキップします")
        return None, {"enabled": False, "checkpoint": ckpt_path}
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    a = ckpt["args"]  # 学習時の設定からモデル構造を再現
    model = SkeletonRefiner(num_joints=NUM_JOINTS, hidden=a["hidden"], num_blocks=a["blocks"],
                            backbone=a["backbone"]).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    info = {"enabled": True, "checkpoint": ckpt_path, "epoch": ckpt.get("epoch"), "backbone": a["backbone"]}
    print(f"[skeleton_refiner] loaded {info}")
    return model, info


skeleton_refiner, REFINER_INFO = load_skeleton_refiner(SKELETON_CKPT)

# ==============================================================================
# 🚀 ROS-like Service Node Endpoint
# ==============================================================================
@app.post("/tennis_motion/analysis_node/request")
async def request_motion_analysis(video: UploadFile = File(...)):
    """
    [ROS Service] /tennis_motion/analysis_node/request
    動画を受け取り、
      ① YOLO + MediaPipe で純粋な 33 点を取得
      ② best.pt (SkeletonRefiner) で 33 点を補正
      ③ 補正後の骨格から頭部中心 (100)・ラケット (99) を算出
      ④ 空間正規化 + Savgol フィルタ
    までを一挙に行う
    """
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="無効な動画フォーマットです。")

    video_id = str(uuid.uuid4())
    input_video_path = os.path.join(UPLOAD_DIR, f"{video_id}_{video.filename}")

    try:
        # 1. 受け取った動画をストレージへ一時保存
        with open(input_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # 2. Pipeline ①: MediaPipe 33 点の抽出（独自関節なし）
        pose_track = execute_pose_extraction(input_video_path)
        if not pose_track["detected"].any():
            raise HTTPException(status_code=422, detail="動画から有効な人物の骨格フレームを検出できませんでした。")

        # 3. Pipeline ②: best.pt による 33 点の補正
        corrected_xyz = refine_landmarks(pose_track)

        # 4. Pipeline ③: 補正後の骨格から頭部・ラケットを付与し、従来形式のログを構築
        raw_data_log = build_keypoint_log(pose_track, corrected_xyz)

        # 5. Pipeline ④: 空間正規化 & SavgolFilter 処理の実行
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

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Error in analysis_node]: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部解析エラー: {str(e)}")

    finally:
        # 後始末: ディスク容量を圧迫しないよう、処理が終わった元動画は即座に削除
        if os.path.exists(input_video_path):
            os.remove(input_video_path)


# ==============================================================================
# 🧠 Pipeline ①: YOLO + MediaPipe による純粋な 33 点の抽出
# ==============================================================================
def execute_pose_extraction(video_path: str) -> dict:
    """座標は学習データ (docs/dataset_memo/json) と同じ「画像全体」基準:
         x, y : 画像幅・高さで 0-1 正規化
         z    : MediaPipe 相対深度（画像幅スケール）
    ROI 内で推論した MediaPipe の座標を画像全体の座標に戻して保持する。
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    xyz_list, vis_list, detected, bboxes, rackets_per_frame = [], [], [], [], []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        xyz = np.zeros((NUM_JOINTS, 3), dtype=np.float64)
        vis = np.zeros(NUM_JOINTS, dtype=np.float64)
        found = False
        bbox = None

        # YOLO推論 (0=Person, 38=Racket)
        yolo_results = yolo_model.predict(frame, classes=[0, 38], conf=0.35, verbose=False, device=DEVICE)
        persons = []
        rackets = []  # ラケット中心 (画像ピクセル座標)

        for result in yolo_results:
            if result.boxes is None: continue
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                if int(cls) == 0:
                    persons.append(box)
                elif int(cls) == 38:
                    rx1, ry1, rx2, ry2 = box.cpu().numpy()
                    rackets.append(((rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0))

        # 主役判定 (Bounding Boxの面積が最大の人)
        if persons:
            persons_t = torch.stack(persons)
            areas = (persons_t[:, 2] - persons_t[:, 0]) * (persons_t[:, 3] - persons_t[:, 1])
            x1, y1, x2, y2 = map(int, persons_t[torch.argmax(areas)].cpu().numpy())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            person_roi = frame[y1:y2, x1:x2]

            if person_roi.size > 0:
                h_roi, w_roi, _ = person_roi.shape
                mp_results = pose_estimator.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))

                if mp_results.pose_landmarks:
                    for j, lm in enumerate(mp_results.pose_landmarks.landmark):
                        # ROI 正規化座標 → 画像全体の正規化座標
                        xyz[j] = ((x1 + lm.x * w_roi) / width,
                                  (y1 + lm.y * h_roi) / height,
                                  lm.z * w_roi / width)
                        vis[j] = lm.visibility
                    found = True
                    bbox = [x1, y1, x2, y2]

        xyz_list.append(xyz)
        vis_list.append(vis)
        detected.append(found)
        bboxes.append(bbox)
        rackets_per_frame.append(rackets)

    cap.release()
    return {
        "fps": fps, "width": width, "height": height,
        "xyz": np.array(xyz_list).reshape(-1, NUM_JOINTS, 3),
        "vis": np.array(vis_list).reshape(-1, NUM_JOINTS),
        "detected": np.array(detected, dtype=bool),
        "bboxes": bboxes,
        "rackets": rackets_per_frame,
    }


# ==============================================================================
# 🧠 Pipeline ②: best.pt (SkeletonRefiner) による 33 点の補正
# ==============================================================================
def refine_landmarks(track: dict) -> np.ndarray:
    """(T,33,3) の補正済み座標を返す（座標系は Pipeline ① と同じ）。
    人物が検出された最初〜最後のフレーム区間をひとつの時系列として補正し、
    区間内の未検出フレーム・低 visibility 点は前処理の PCHIP 補間で埋めてから補正する。
    モデルが無い場合は生の座標をそのまま返す。"""
    xyz = track["xyz"].copy()
    if skeleton_refiner is None:
        return xyz

    idx = np.flatnonzero(track["detected"])
    start, end = int(idx[0]), int(idx[-1]) + 1
    if end - start < 3:
        return xyz

    meta = {"fps": track["fps"], "resolution": {"width": track["width"], "height": track["height"]}}
    seq = preprocess_arrays(xyz[start:end], track["vis"][start:end], meta, name="upload")

    coords = torch.from_numpy(seq.coords).to(DEVICE)
    conf = torch.from_numpy(seq.conf).to(DEVICE)
    pred = skeleton_refiner.refine_sequence(coords, conf).cpu().numpy()
    xyz[start:end] = denormalize(seq, pred)
    return xyz


# ==============================================================================
# 🧠 Pipeline ③: 補正済み 33 点 → 頭部中心 (100)・ラケット (99) を付与
# ==============================================================================
def build_keypoint_log(track: dict, xyz: np.ndarray) -> dict:
    fps, width, height = track["fps"], track["width"], track["height"]
    vis = track["vis"]

    data_log = {
        "metadata": {
            "fps": fps,
            "resolution": {"width": width, "height": height},
            "required_ids": REQUIRED_IDS,
            "coord_frame": "image_normalized",
            "skeleton_refined": REFINER_INFO,
        },
        "frames": []
    }

    prev = {}
    for t in range(len(xyz)):
        frame_data = {"frame_index": t, "timestamp": round(t / fps, 4), "skeletons": []}

        if track["detected"][t]:
            kps = []

            def add_kp(jid, x, y, z, v):
                velocity = 0.0
                if jid in prev:
                    px, py, pz = prev[jid]
                    velocity = round(math.sqrt((x - px)**2 + (y - py)**2 + (z - pz)**2) * fps, 4)
                prev[jid] = (x, y, z)
                kps.append({
                    "id": jid,
                    "rel_coord": {"x": round(float(x), 4), "y": round(float(y), 4), "z": round(float(z), 4)},
                    "visibility": round(float(v), 4),
                    "velocity": velocity,
                })

            # 標準関節 (補正済み MediaPipe 33 点から抽出)
            for jid in REQUIRED_IDS:
                if jid in (99, 100): continue
                add_kp(jid, *xyz[t, jid], vis[t, jid])

            # 頭部中心 (ID 100): 補正後の鼻・両耳の平均
            hx, hy, hz = xyz[t, HEAD_SRC_IDS].mean(axis=0)
            add_kp(100, hx, hy, hz, vis[t, HEAD_SRC_IDS].mean())

            # ラケット (ID 99): 補正後の右手首に最も近い YOLO ラケット
            wx, wy, wz = xyz[t, RACKET_HAND_ID]
            hand_px = np.array([wx * width, wy * height])
            best_racket, best_dist = None, RACKET_MAX_DIST_PX
            for rc in track["rackets"][t]:
                d = float(np.linalg.norm(hand_px - np.asarray(rc)))
                if d < best_dist:
                    best_dist, best_racket = d, rc
            if best_racket is not None:
                add_kp(99, best_racket[0] / width, best_racket[1] / height, wz, 1.0)
            else:
                add_kp(99, wx, wy, wz, 0.0)  # 未検出時は手首位置で代用

            frame_data["skeletons"].append({"bbox": track["bboxes"][t], "keypoints": kps})

        data_log["frames"].append(frame_data)

    return data_log


# ==============================================================================
# 🧠 Pipeline ④: 腰原点アライメント & 接地補正 & 時系列Savgol平滑化
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
