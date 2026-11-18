"""
① データクレンジング・前処理・スケール不変正規化パイプライン

単眼 MediaPipe Pose の時系列 JSON (docs/dataset_memo/json/*.json) を読み込み、
  1. 低 visibility キーポイントの欠損扱い + PCHIP 補間（端は最近傍ホールド）
  2. 信頼度重み conf ∈ [0,1]（ロスの重み付け・除外に使用）
  3. アスペクト比補正 → 腰中点を原点とするセンタリング → 動画ごとの体幹長でスケール正規化
  4. 動画ごとの「相対骨長比率」の参照値を算出・保持
を行い、PyTorch Dataset / pandas DataFrame として提供する。

座標系メモ:
  MediaPipe の x, y は画像幅・高さでそれぞれ 0-1 正規化、z は画像幅とほぼ同スケールの
  相対深度（小さいほどカメラに近い）。y に H/W を掛けて等方的な座標 (x右, y下, z奥) に揃える。
  この座標系は右手系。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # 前処理のみ使う場合は torch 不要
    torch = None
    Dataset = object

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "docs" / "dataset_memo" / "json"

NUM_JOINTS = 33

# MediaPipe Pose landmark index
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# 左右反転 (mirror augmentation) 用の対応ペア
FLIP_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
              (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
FLIP_INDEX = np.arange(NUM_JOINTS)
for _a, _b in FLIP_PAIRS:
    FLIP_INDEX[_a], FLIP_INDEX[_b] = _b, _a

# 骨長制約に使う主要な骨（顔・手指・足先はノイズが大きいので除外）
BONES: list[tuple[str, int, int]] = [
    ("shoulder_width", L_SHOULDER, R_SHOULDER),
    ("hip_width", L_HIP, R_HIP),
    ("torso_L", L_SHOULDER, L_HIP),
    ("torso_R", R_SHOULDER, R_HIP),
    ("upper_arm_L", L_SHOULDER, L_ELBOW),
    ("upper_arm_R", R_SHOULDER, R_ELBOW),
    ("forearm_L", L_ELBOW, L_WRIST),
    ("forearm_R", R_ELBOW, R_WRIST),
    ("thigh_L", L_HIP, L_KNEE),
    ("thigh_R", R_HIP, R_KNEE),
    ("shin_L", L_KNEE, L_ANKLE),
    ("shin_R", R_KNEE, R_ANKLE),
]
BONE_NAMES = [b[0] for b in BONES]
BONE_INDEX = np.array([[b[1], b[2]] for b in BONES], dtype=np.int64)  # (NB, 2)
# 左右対称な骨のペア（BONES 内のインデックス）。左右で骨長比を共有する。
SYMMETRIC_BONE_PAIRS = [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]


@dataclass
class PreprocessConfig:
    vis_threshold: float = 0.5      # これ未満は欠損扱い → 補間対象・ロスから除外
    vis_full: float = 0.9           # これ以上は重み 1.0（間は線形）
    bone_ref_quantile: float = 0.95  # 骨長参照値 = 2D 骨長の上位分位点（画像平面と平行に近い瞬間）
    min_valid_ratio: float = 0.2    # 有効フレームがこれ未満の関節は補間せず生値のまま（重み 0）


@dataclass
class SequenceData:
    """1 動画分の前処理済みデータ（座標はすべて正規化空間）。"""
    name: str
    motion: str
    view: str
    fps: float
    aspect: float                   # H / W
    coords: np.ndarray              # (T, J, 3) 補間・正規化済み (x, y, z)
    raw_coords: np.ndarray          # (T, J, 3) 補間前・正規化済み（比較・評価用）
    visibility: np.ndarray          # (T, J) 元の visibility
    conf: np.ndarray                # (T, J) ロス用信頼度重み [0,1]（欠損は 0）
    missing: np.ndarray             # (T, J) bool 補間で埋めた点
    root: np.ndarray                # (T, 3) 正規化前（アスペクト補正後）の腰中点
    scale: float                    # 正規化に使った体幹長
    bone_ref_ratio: np.ndarray      # (NB,) 動画ごとの相対骨長比率（平均 1 に正規化）
    meta: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return self.coords.shape[0]


# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------

def read_json(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """JSON → (T,J,3) 座標, (T,J) visibility, メタ情報。"""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    frames = sorted(d["frames"], key=lambda fr: fr["frame_index"])
    T = len(frames)
    xyz = np.zeros((T, NUM_JOINTS, 3), dtype=np.float64)
    vis = np.zeros((T, NUM_JOINTS), dtype=np.float64)
    for t, fr in enumerate(frames):
        for kp in fr["keypoints"]:
            j = kp["id"]
            c = kp["coord"]
            xyz[t, j] = (c["x"], c["y"], c["z"])
            vis[t, j] = kp.get("visibility", 1.0)
    meta = {k: v for k, v in d.items() if k != "frames"}
    return xyz, vis, meta


def list_json_files(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[Path]:
    files = sorted(Path(data_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"JSON が見つかりません: {data_dir}")
    return files


# ---------------------------------------------------------------------------
# クレンジング
# ---------------------------------------------------------------------------

def visibility_to_conf(vis: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    return np.clip((vis - cfg.vis_threshold) / (cfg.vis_full - cfg.vis_threshold), 0.0, 1.0)


def interpolate_missing(xyz: np.ndarray, valid: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, np.ndarray]:
    """関節ごとに有効フレームから PCHIP 補間（オーバーシュートしない単調保存補間）。
    系列端の外挿は行わず最近傍の有効値でホールドする。

    Returns: 補間後座標, 補間で埋めたかどうかの mask
    """
    T, J, _ = xyz.shape
    out = xyz.copy()
    filled = np.zeros((T, J), dtype=bool)
    t_all = np.arange(T)
    for j in range(J):
        v = valid[:, j]
        n_valid = int(v.sum())
        if n_valid == T or n_valid < max(1, cfg.min_valid_ratio * T):
            continue  # 欠損なし / 有効点が少なすぎて補間が信用できない
        tv = t_all[v]
        if n_valid == 1:
            out[:, j] = xyz[tv[0], j]
        else:
            interp = PchipInterpolator(tv, xyz[tv, j], axis=0, extrapolate=False)
            filled_vals = interp(t_all)
            # 端はホールド
            filled_vals[: tv[0]] = xyz[tv[0], j]
            filled_vals[tv[-1] + 1:] = xyz[tv[-1], j]
            out[:, j] = filled_vals
        filled[~v, j] = True
    return out, filled


# ---------------------------------------------------------------------------
# スケール不変正規化
# ---------------------------------------------------------------------------

def hip_center(xyz: np.ndarray) -> np.ndarray:
    return 0.5 * (xyz[..., L_HIP, :] + xyz[..., R_HIP, :])


def shoulder_center(xyz: np.ndarray) -> np.ndarray:
    return 0.5 * (xyz[..., L_SHOULDER, :] + xyz[..., R_SHOULDER, :])


def torso_scale(xyz: np.ndarray, conf: np.ndarray) -> float:
    """動画ごとのスケール = 2D 体幹長（肩中点–腰中点）の中央値。
    深度 z は信頼できないため 2D のみで測る。フレームごとではなく動画で 1 値にして
    時間的な一貫性を保つ。"""
    torso = np.linalg.norm(shoulder_center(xyz)[:, :2] - hip_center(xyz)[:, :2], axis=-1)
    ok = conf[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]].min(axis=1) > 0
    vals = torso[ok] if ok.sum() >= 3 else torso
    s = float(np.median(vals))
    return s if s > 1e-6 else 1.0


def bone_lengths_np(xyz: np.ndarray, dims: slice = slice(None)) -> np.ndarray:
    a = xyz[..., BONE_INDEX[:, 0], dims]
    b = xyz[..., BONE_INDEX[:, 1], dims]
    return np.linalg.norm(a - b, axis=-1)


def estimate_bone_ref_ratio(xyz: np.ndarray, conf: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """動画ごとの相対骨長比率の参照値。

    正射影では 2D 骨長 ≤ 3D 骨長であり、スイング中に骨が画像平面と平行に近づく瞬間に
    2D 長 ≈ 3D 長となる。そこで信頼できるフレームの 2D 骨長の上位分位点を 3D 骨長の推定値とする
    （MediaPipe の z はスケールが過大になりがちなので使わない）。
    左右対称な骨は平均を取って共有し、最後に平均 1 になるよう比率化してスケールを消す。
    """
    L2 = bone_lengths_np(xyz, slice(0, 2))                       # (T, NB)
    w = np.minimum(conf[:, BONE_INDEX[:, 0]], conf[:, BONE_INDEX[:, 1]])  # (T, NB)
    ref = np.zeros(len(BONES))
    for b in range(len(BONES)):
        vals = L2[w[:, b] > 0.5, b]
        if len(vals) < 3:
            vals = L2[:, b]
        ref[b] = np.quantile(vals, cfg.bone_ref_quantile)
    for a, b in SYMMETRIC_BONE_PAIRS:
        m = 0.5 * (ref[a] + ref[b])
        ref[a] = ref[b] = m
    return ref / ref.mean()


def load_sequence(path: str | Path, cfg: PreprocessConfig | None = None) -> SequenceData:
    path = Path(path)
    xyz, vis, meta = read_json(path)
    return preprocess_arrays(xyz, vis, meta, name=path.stem, cfg=cfg)


def preprocess_arrays(xyz: np.ndarray, vis: np.ndarray, meta: dict, name: str = "sequence",
                      cfg: PreprocessConfig | None = None) -> SequenceData:
    """メモリ上の (T,J,3) 座標 + (T,J) visibility から SequenceData を作る（API からの推論用）。
    xyz は JSON と同じ座標系（x,y は画像全体で 0-1 正規化、z は MediaPipe 相対深度）。"""
    cfg = cfg or PreprocessConfig()
    xyz = np.asarray(xyz, dtype=np.float64)
    vis = np.asarray(vis, dtype=np.float64)
    res = meta.get("resolution") or {"width": 1, "height": 1}
    aspect = float(res["height"]) / float(res["width"])
    xyz = xyz.copy()
    xyz[..., 1] *= aspect                                         # 等方的な座標へ

    conf = visibility_to_conf(vis, cfg)
    valid = vis >= cfg.vis_threshold
    clean, filled = interpolate_missing(xyz, valid, cfg)
    conf = np.where(filled, 0.0, conf)

    root = hip_center(clean)                                      # (T, 3)
    scale = torso_scale(clean, conf)
    norm_clean = (clean - root[:, None]) / scale
    norm_raw = (xyz - root[:, None]) / scale

    return SequenceData(
        name=name,
        motion=meta.get("motion", "unknown"),
        view=meta.get("view", "unknown"),
        fps=float(meta.get("fps", 30.0)),
        aspect=aspect,
        coords=norm_clean.astype(np.float32),
        raw_coords=norm_raw.astype(np.float32),
        visibility=vis.astype(np.float32),
        conf=conf.astype(np.float32),
        missing=filled,
        root=root.astype(np.float32),
        scale=scale,
        bone_ref_ratio=estimate_bone_ref_ratio(norm_clean, conf, cfg).astype(np.float32),
        meta=meta,
    )


def load_dataset(data_dir: str | Path = DEFAULT_DATA_DIR, cfg: PreprocessConfig | None = None) -> list[SequenceData]:
    return [load_sequence(p, cfg) for p in list_json_files(data_dir)]


def denormalize(seq: SequenceData, coords: np.ndarray) -> np.ndarray:
    """正規化空間の (T,J,3) を元の JSON 座標系 (x,y は 0-1, z は MediaPipe 相対深度) に戻す。"""
    out = coords.astype(np.float64) * seq.scale + seq.root[:, None]
    out[..., 1] /= seq.aspect
    return out


# ---------------------------------------------------------------------------
# DataFrame 変換 / 結果保存
# ---------------------------------------------------------------------------

def sequence_to_dataframe(seq: SequenceData, coords: np.ndarray | None = None):
    """long 形式の DataFrame (1 行 = 1 フレーム × 1 関節)。"""
    import pandas as pd

    c = seq.coords if coords is None else coords
    T, J, _ = c.shape
    t_idx, j_idx = np.meshgrid(np.arange(T), np.arange(J), indexing="ij")
    return pd.DataFrame({
        "sequence": seq.name, "motion": seq.motion, "view": seq.view,
        "frame_index": t_idx.ravel(), "joint_id": j_idx.ravel(),
        "x": c[..., 0].ravel(), "y": c[..., 1].ravel(), "z": c[..., 2].ravel(),
        "visibility": seq.visibility.ravel(), "conf": seq.conf.ravel(),
        "interpolated": seq.missing.ravel(),
    })


def bone_ratio_table(seqs: list[SequenceData]):
    """動画ごとの相対骨長比率の一覧 (DataFrame)。"""
    import pandas as pd

    return pd.DataFrame([s.bone_ref_ratio for s in seqs], index=[s.name for s in seqs], columns=BONE_NAMES)


def save_corrected_json(seq: SequenceData, coords_norm: np.ndarray, src_path: str | Path, out_path: str | Path) -> None:
    """補正結果を元 JSON と同じスキーマで保存（座標のみ差し替え、元の visibility は保持）。"""
    with open(src_path, encoding="utf-8") as f:
        d = json.load(f)
    xyz = denormalize(seq, coords_norm)
    frames = sorted(d["frames"], key=lambda fr: fr["frame_index"])
    for t, fr in enumerate(frames):
        for kp in fr["keypoints"]:
            x, y, z = xyz[t, kp["id"]]
            kp["coord"] = {"x": round(float(x), 4), "y": round(float(y), 4), "z": round(float(z), 4)}
    d["frames"] = frames
    d["corrected_by"] = "skeleton_correction"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

def make_model_input(coords: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """(T,J,3)+(T,J) → (T,J,4) [x, y, z, conf]"""
    return np.concatenate([coords, conf[..., None]], axis=-1).astype(np.float32)


class SkeletonWindowDataset(Dataset):
    """動画を固定長ウィンドウに切り出して返す Dataset。

    返り値 (dict of tensors):
      input      (W,J,4)  モデル入力（augment 時は z にノイズ・関節ドロップアウトを付与）
      target_xy  (W,J,2)  再投影ロスの目標（観測 2D）
      clean_z    (W,J)    ノイズ付与前の z（深度の前後関係 prior 用）
      conf       (W,J)    ロス重み
      bone_ref   (NB,)    動画ごとの相対骨長比率
      video_idx  ()       動画インデックス（骨長比パラメータ共有用）
    """

    def __init__(self, seqs: list[SequenceData], video_ids: list[int], window: int = 48, stride: int = 8,
                 augment: bool = False, z_noise: float = 0.05, z_spike_prob: float = 0.02,
                 dropout_prob: float = 0.05, mirror_prob: float = 0.5, seed: int = 0):
        assert torch is not None, "torch が必要です"
        self.seqs = seqs
        self.video_ids = video_ids
        self.window = window
        self.augment = augment
        self.z_noise = z_noise
        self.z_spike_prob = z_spike_prob
        self.dropout_prob = dropout_prob
        self.mirror_prob = mirror_prob
        self.rng = np.random.default_rng(seed)
        self.index: list[tuple[int, int]] = []
        for si, s in enumerate(seqs):
            T = s.num_frames
            if T <= window:
                self.index.append((si, 0))
                continue
            starts = list(range(0, T - window + 1, stride))
            if starts[-1] != T - window:
                starts.append(T - window)
            self.index += [(si, st) for st in starts]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        si, st = self.index[i]
        s = self.seqs[si]
        sl = slice(st, st + self.window)
        coords = s.coords[sl].copy()
        conf = s.conf[sl].copy()
        if coords.shape[0] < self.window:  # 短い動画は端をホールドしてパディング（重み 0）
            pad = self.window - coords.shape[0]
            coords = np.concatenate([coords, np.repeat(coords[-1:], pad, 0)])
            conf = np.concatenate([conf, np.zeros((pad, NUM_JOINTS), np.float32)])

        if self.augment and self.rng.random() < self.mirror_prob:
            coords = coords[:, FLIP_INDEX].copy()
            coords[..., 0] *= -1.0
            conf = conf[:, FLIP_INDEX].copy()

        clean_z = coords[..., 2].copy()
        inp_coords = coords.copy()
        inp_conf = conf.copy()
        if self.augment:
            W = self.window
            inp_coords[..., 2] += self.rng.normal(0, self.z_noise, (W, NUM_JOINTS))
            spikes = self.rng.random((W, NUM_JOINTS)) < self.z_spike_prob
            inp_coords[..., 2] += spikes * self.rng.normal(0, 0.3, (W, NUM_JOINTS))
            drop = self.rng.random((W, NUM_JOINTS)) < self.dropout_prob
            inp_conf[drop] = 0.0
            inp_coords[drop] += self.rng.normal(0, 0.05, (int(drop.sum()), 3))

        return {
            "input": torch.from_numpy(make_model_input(inp_coords, inp_conf)),
            "target_xy": torch.from_numpy(coords[..., :2].astype(np.float32)),
            "clean_z": torch.from_numpy(clean_z.astype(np.float32)),
            "conf": torch.from_numpy(conf.astype(np.float32)),
            "bone_ref": torch.from_numpy(s.bone_ref_ratio),
            "video_idx": torch.tensor(self.video_ids[si], dtype=torch.long),
        }


if __name__ == "__main__":
    seqs = load_dataset()
    print(f"{len(seqs)} sequences loaded from {DEFAULT_DATA_DIR}")
    for s in seqs[:5]:
        print(f"  {s.name:14s} {s.motion:8s} {s.view:10s} T={s.num_frames:4d} scale={s.scale:.3f} "
              f"missing={s.missing.mean():.1%}")
    print(bone_ratio_table(seqs).round(2).head())
    print(sequence_to_dataframe(seqs[0]).head())
