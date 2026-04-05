import os
import json
import argparse
import base64
import io
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image


# =========================================================
# SETTINGS
# =========================================================
PROJECT_ROOT = "/content/drive/MyDrive/models deployment"
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "regions.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

SEQ_LEN = 8
PATCH_SIZE = 512

# Precomputed weekly frames available for all regions
PRECOMPUTED_DATES = [
    "2026-01-01", "2026-01-08", "2026-01-15", "2026-01-22",
    "2026-02-05", "2026-02-12", "2026-02-19"
]


# =========================================================
# MODEL
# =========================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=(3, 3), bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, x, state):
        h_cur, c_cur = state
        combined = torch.cat([x, h_cur], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device):
        H, W = spatial_size
        h = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=(3, 3), num_layers=1, out_channels=None, simple_out_conv=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim] * num_layers
        self.kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size] * num_layers
        self.num_layers = num_layers
        self.out_channels = out_channels or input_dim

        assert len(self.hidden_dim) == num_layers
        assert len(self.kernel_size) == num_layers

        cells = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else self.hidden_dim[i - 1]
            cells.append(ConvLSTMCell(in_dim, self.hidden_dim[i], self.kernel_size[i]))
        self.cells = nn.ModuleList(cells)

        # simple_out_conv=True: single Conv2d (older checkpoints e.g. dubai)
        # simple_out_conv=False: two-layer Sequential (standard checkpoints)
        if simple_out_conv:
            self.out_conv = nn.Conv2d(self.hidden_dim[-1], self.out_channels, kernel_size=1)
        else:
            self.out_conv = nn.Sequential(
                nn.Conv2d(self.hidden_dim[-1], self.hidden_dim[-1], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_dim[-1], self.out_channels, kernel_size=1)
            )

    def forward(self, x):
        B, T, C, H, W = x.shape
        device = x.device
        states = [cell.init_hidden(B, (H, W), device) for cell in self.cells]

        for t in range(T):
            inp = x[:, t]
            for layer_idx, cell in enumerate(self.cells):
                h, c = states[layer_idx]
                h_new, c_new = cell(inp, (h, c))
                states[layer_idx] = (h_new, c_new)
                inp = h_new

        last_h = states[-1][0]
        residual = x[:, -1]

        out = self.out_conv(last_h)
        out = out + residual
        out = torch.clamp(out, 0.0, 1.0)
        return out


# =========================================================
# HELPERS
# =========================================================
def load_png_as_array(path):
    """Load an RGB png back as a (3, H, W) float32 array in [0,1]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return arr.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)


def frame_to_base64(frame, channel_order=(0, 1, 2), stretch=True):
    """Convert a (C, H, W) frame to a base64-encoded PNG string."""
    rgb = make_rgb(frame, channel_order=channel_order, stretch=stretch)
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    img = Image.fromarray(rgb_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def find_nearest_precomputed(region, date_str):
    """
    If date falls within the precomputed range, find the nearest precomputed
    PNG by date and return its path. Returns None if no match found.
    """
    target = datetime.strptime(date_str, "%Y-%m-%d")
    precomputed = [datetime.strptime(d, "%Y-%m-%d") for d in PRECOMPUTED_DATES]

    range_start = precomputed[0]
    range_end   = precomputed[-1]

    # Only use precomputed if within the precomputed date range
    if target < range_start or target > range_end:
        return None

    # Find nearest precomputed date
    nearest = min(precomputed, key=lambda d: abs((d - target).days))
    nearest_str = nearest.strftime("%Y-%m-%d")

    path = os.path.join(OUTPUT_DIR, f"{region}_{nearest_str}_predicted_full_rgb.png")
    if os.path.exists(path):
        print(f"Found precomputed PNG for {nearest_str}: {path}")
        return path, nearest_str

    print(f"Warning: expected precomputed PNG not found: {path}")
    return None


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def full_path(rel_path):
    return os.path.join(PROJECT_ROOT, rel_path)


def load_series(path):
    npz = np.load(path)
    arr = npz["data"] if "data" in npz else npz[npz.files[0]]

    arr = arr.astype(np.float32)

    # Only normalize if values are clearly in 0-255 range
    if arr.max() > 2.0:
        arr = arr / 255.0

    return arr


def date_to_index(date_str, start_date_str, end_date_str, num_frames):
    """
    Map a date string to a frame index by linearly interpolating across
    the real date range of the dataset. This avoids the broken assumption
    of perfectly regular weekly steps and absorbs gaps in the data.
    """
    target = datetime.strptime(date_str, "%Y-%m-%d")
    start  = datetime.strptime(start_date_str, "%Y-%m-%d")
    end    = datetime.strptime(end_date_str, "%Y-%m-%d")

    total_days  = (end - start).days
    target_days = (target - start).days

    idx = round((target_days / total_days) * (num_frames - 1))
    return max(0, idx)


def load_model(model_path, device):
    state = torch.load(model_path, map_location=device)
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state

    # Auto-detect out_conv architecture from checkpoint keys:
    # older checkpoints (e.g. dubai) have a single Conv2d -> "out_conv.weight"
    # standard checkpoints have a Sequential -> "out_conv.0.weight"
    simple_out_conv = "out_conv.weight" in sd

    if simple_out_conv:
        print("Note: detected simple out_conv architecture in checkpoint")

    model = ConvLSTM(
        input_dim=4,
        hidden_dim=[32, 64],
        kernel_size=(3, 3),
        num_layers=2,
        out_channels=4,
        simple_out_conv=simple_out_conv
    ).to(device)

    model.load_state_dict(sd)
    model.eval()
    return model


def find_best_index(data, target_idx, search_radius=10):
    """Find the least-cloudy frame near the target index."""
    start = max(0, target_idx - search_radius)
    end = min(len(data), target_idx + search_radius + 1)

    best_idx = target_idx
    best_score = -1

    for i in range(start, end):
        frame = data[i]
        zero_frac = np.mean(frame == 0)
        if zero_frac > 0.3:  # skip if >30% masked/cloud
            continue
        score = frame.mean()
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def get_seed_seq(data, seq_len, search_window=50):
    """
    Get seq_len cleanest frames from the tail of the dataset, in temporal order.
    Picks frames with least cloud/mask coverage rather than just the last seq_len.
    """
    window_start = max(0, len(data) - search_window)
    window = [
        (i, data[i].mean(), np.mean(data[i] == 0))
        for i in range(window_start, len(data))
    ]

    # Try progressively relaxed thresholds until we have enough frames
    for max_zero_frac in (0.5, 0.8, 1.0):
        valid = [(i, mean) for i, mean, zero_frac in window if zero_frac < max_zero_frac]
        if len(valid) >= seq_len:
            break

    if len(valid) < seq_len:
        raise ValueError(
            f"Not enough clean frames in last {search_window} frames "
            f"(found {len(valid)}, need {seq_len})"
        )

    # Pick the seq_len highest-quality frames, then re-sort by index to preserve time order
    valid.sort(key=lambda x: x[1], reverse=True)
    best_indices = sorted([i for i, _ in valid[:seq_len]])

    print(f"Seed frame indices: {best_indices}")
    print(f"Seed frame means:   {[round(float(data[i].mean()), 3) for i in best_indices]}")

    return data[best_indices]


def predict_scene_patches(model, seed_seq, device, patch_size=PATCH_SIZE):
    """
    Run the model patch-by-patch over the full scene and stitch results
    back together. seed_seq shape: (SEQ_LEN, C, H, W).
    Returns a full-resolution (C, H, W) prediction.
    """
    T, C, H, W = seed_seq.shape
    output = np.zeros((C, H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    y_steps = list(range(0, H, patch_size))
    x_steps = list(range(0, W, patch_size))
    total = len(y_steps) * len(x_steps)
    patch_num = 0

    for y in y_steps:
        for x in x_steps:
            patch_num += 1
            print(f"  Patch {patch_num}/{total} (y={y}, x={x})", flush=True)

            y1 = min(y + patch_size, H)
            x1 = min(x + patch_size, W)

            patch = seed_seq[:, :, y:y1, x:x1]
            ph, pw = patch.shape[2], patch.shape[3]

            if ph < patch_size or pw < patch_size:
                padded = np.zeros((T, C, patch_size, patch_size), dtype=patch.dtype)
                padded[:, :, :ph, :pw] = patch
                patch = padded

            tensor = torch.tensor(patch, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                pred = model(tensor)
                pred_np = pred.squeeze(0).cpu().numpy()
            del tensor, pred
            if device.type == "cuda":
                torch.cuda.empty_cache()

            output[:, y:y1, x:x1] += pred_np[:, :y1 - y, :x1 - x]
            counts[y:y1, x:x1] += 1

    output /= counts
    return output


def roll_forward_patches(model, seed_seq, steps_ahead, device, patch_size=PATCH_SIZE):
    """
    Recursively predict steps_ahead frames on the full scene using
    patch-based inference at every step.
    seed_seq shape: (SEQ_LEN, C, H, W)
    """
    MAX_SAFE_STEPS = 4
    if steps_ahead > MAX_SAFE_STEPS:
        print(
            f"Warning: {steps_ahead} recursive steps requested. "
            f"Quality degrades significantly beyond {MAX_SAFE_STEPS} steps."
        )

    current_seq = seed_seq.copy()   # (SEQ_LEN, C, H, W)

    pred = None
    for step in range(steps_ahead):
        print(f"  Recursive step {step + 1}/{steps_ahead} ...")
        pred = predict_scene_patches(model, current_seq, device, patch_size)   # (C, H, W)
        # Slide the window forward
        current_seq = np.concatenate([current_seq[1:], pred[np.newaxis]], axis=0)

    return pred   # (C, H, W)


def stretch_channel(img, low=2, high=98):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [low, high])

    if hi - lo < 1e-8:
        return np.zeros_like(img, dtype=np.float32)

    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-8)
    return img


def make_rgb(frame, channel_order=(0, 1, 2), stretch=True):
    rgb = np.stack([frame[c] for c in channel_order], axis=-1).astype(np.float32)

    if stretch:
        for i in range(3):
            rgb[..., i] = stretch_channel(rgb[..., i])

    rgb = np.clip(rgb, 0, 1)
    return rgb


def save_rgb_png(frame, out_path, channel_order=(0, 1, 2), stretch=True):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rgb = make_rgb(frame, channel_order=channel_order, stretch=stretch)
    plt.imsave(out_path, rgb)


def save_channel_debug(frame, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(1, frame.shape[0], figsize=(4 * frame.shape[0], 4))
    if frame.shape[0] == 1:
        axes = [axes]

    for i in range(frame.shape[0]):
        ch = stretch_channel(frame[i])
        axes[i].imshow(ch, cmap="gray")
        axes[i].set_title(
            f"Ch {i}\nmin={frame[i].min():.3f}, max={frame[i].max():.3f}\nmean={frame[i].mean():.3f}"
        )
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_channel_stats(frame):
    print("Frame shape:", frame.shape)
    for i in range(frame.shape[0]):
        print(
            f"Channel {i}: "
            f"min={frame[i].min():.6f}, "
            f"max={frame[i].max():.6f}, "
            f"mean={frame[i].mean():.6f}"
        )


# =========================================================
# MAIN
# =========================================================
def predict(
    region,
    date_str,
    patch_mode=True,
    max_future_steps=8,
    rgb_channels=(0, 1, 2),
    stretch_rgb=True
):
    regions = load_config()

    if region not in regions:
        raise ValueError(f"Unknown region: {region}")

    cfg = regions[region]

    # end_date is required in regions.json now
    if "end_date" not in cfg:
        raise ValueError(
            f"Region '{region}' is missing 'end_date' in regions.json. "
            f"Please add it (e.g. \"end_date\": \"2025-12-31\")."
        )

    device = torch.device("cpu" if cfg.get("force_cpu") else ("cuda" if torch.cuda.is_available() else "cpu"))

    data_path  = full_path(cfg["data_path"])
    model_path = full_path(cfg["model_path"])

    print(f"\n=== REGION: {region} ===")
    print("Data path:", data_path)
    print("Model path:", model_path)
    print("Device:", device)

    # Check for precomputed PNG first — skips model loading entirely
    precomputed = find_nearest_precomputed(region, date_str)
    if precomputed is not None:
        png_path, nearest_date = precomputed
        frame = load_png_as_array(png_path)
        print(f"Source: precomputed (nearest date: {nearest_date})")
        return {
            "region":     region,
            "date":       date_str,
            "index":      -1,
            "source":     "precomputed",
            "shape":      frame.shape,
            "rgb_path":   png_path,
            "debug_path": None,
            "frame":      frame,
            "image_b64":  base64.b64encode(open(png_path, "rb").read()).decode("utf-8")
        }

    data  = load_series(data_path)   # (T, C, H, W)
    model = load_model(model_path, device)

    print("Loaded data shape:", data.shape)

    idx = date_to_index(date_str, cfg["start_date"], cfg["end_date"], len(data))
    print("Requested date:", date_str)
    print("Mapped index:", idx)
    print("Dataset length:", len(data))

    if idx < 0:
        raise ValueError(f"Date {date_str} is before start_date {cfg['start_date']}")

    if idx < len(data):
        # --- Real frame: find cleanest frame near the target index ---
        best_idx = find_best_index(data, idx)
        if best_idx != idx:
            print(f"Note: shifted index {idx} -> {best_idx} to avoid cloudy/masked frame")
        frame  = data[best_idx]
        source = "real"
        print(f"Source: real frame (index {best_idx})")
    else:
        # --- Future frame: recursive patch-based prediction ---
        steps_ahead = idx - len(data) + 1
        print("Steps ahead:", steps_ahead)

        if steps_ahead > max_future_steps:
            raise ValueError(
                f"Requested prediction is {steps_ahead} steps ahead. "
                f"Max allowed is {max_future_steps}."
            )

        if len(data) < SEQ_LEN:
            raise ValueError(
                f"Region {region} has only {len(data)} timesteps, need at least {SEQ_LEN}."
            )

        seed_seq = data[-SEQ_LEN:] if cfg.get("raw_seed") else get_seed_seq(data, SEQ_LEN)   # (SEQ_LEN, C, H, W) — clean frames only
        print("Seed sequence shape:", seed_seq.shape)

        frame  = roll_forward_patches(model, seed_seq, steps_ahead, device, PATCH_SIZE)
        source = "predicted"

    print("\n=== CHANNEL STATS ===")
    print_channel_stats(frame)

    rgb_name = f"{region}_{date_str}_{source}_full_rgb.png"
    dbg_name = f"{region}_{date_str}_{source}_full_debug.png"

    rgb_path = os.path.join(OUTPUT_DIR, rgb_name)
    dbg_path = os.path.join(OUTPUT_DIR, dbg_name)

    save_rgb_png(frame, rgb_path, channel_order=rgb_channels, stretch=stretch_rgb)
    save_channel_debug(frame, dbg_path)

    return {
        "region":     region,
        "date":       date_str,
        "index":      idx,
        "source":     source,
        "shape":      frame.shape,
        "rgb_path":   rgb_path,
        "debug_path": dbg_path,
        "frame":      frame,
        "image_b64":  frame_to_base64(frame, channel_order=rgb_channels, stretch=stretch_rgb)
    }


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--date",   type=str, required=True)
    parser.add_argument("--max_future_steps", type=int, default=8)
    parser.add_argument("--rgb_channels", type=int, nargs=3, default=[0, 1, 2],
                        help="Three channel indices for RGB output, e.g. 0 1 2 or 1 2 3")
    parser.add_argument("--no_stretch", action="store_true", help="Disable contrast stretching")

    args = parser.parse_args()

    result = predict(
        region          = args.region,
        date_str        = args.date,
        max_future_steps= args.max_future_steps,
        rgb_channels    = tuple(args.rgb_channels),
        stretch_rgb     = not args.no_stretch
    )

    print("\n=== RESULT ===")
    print("Region:",     result["region"])
    print("Date:",       result["date"])
    print("Source:",     result["source"])
    print("Shape:",      result["shape"])
    print("RGB image:",  result["rgb_path"])
    print("Debug image:", result["debug_path"])
