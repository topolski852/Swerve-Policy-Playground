# prep_media.py — copy the clips the deck uses into presentation/media/ and
# grab a poster frame (shown before you press play) for each one.
#
# Run from the repo root:  python presentation/prep_media.py
# Re-run after a new path_randomizer training run to pick up its videos.

import glob
import json
import os
import shutil
import subprocess

import imageio_ffmpeg

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA = os.path.join(ROOT, "presentation", "media")
FF    = imageio_ffmpeg.get_ffmpeg_exe()

# id -> source mp4 (relative to repo root)
CLIPS = {
    # Chapter 1: blank background, set nodes
    "ch1_500":   "recordings/run_20260618_1400/eval_0000500_steps.mp4",
    "ch1_8k":    "recordings/run_20260618_1400/eval_0008000_steps.mp4",
    "ch1_12k":   "recordings/run_20260618_1400/eval_0012000_steps.mp4",
    # Chapter 2: FRC field + collisions
    "ch2_500":   "recordings/run_20260618_2112/eval_0000500_steps.mp4",
    "ch2_25k":   "recordings/run_20260618_2112/eval_0025000_steps.mp4",
    "ch2_100k":  "recordings/run_20260618_2112/eval_0100000_steps.mp4",
    "ch2_park":  "recordings/run_20260619_0156/eval_0150000_steps.mp4",
    "ch2_18k":   "recordings/run_20260619_1111/eval_0018000_steps.mp4",
    "hook":      "recordings/run_20260619_1111/eval_0050000_steps.mp4",
}


def duration(path):
    out = subprocess.run([FF, "-i", path], capture_output=True, text=True).stderr
    for line in out.splitlines():
        if "Duration:" in line:
            h, m, s = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def size(path):
    out = subprocess.run([FF, "-i", path], capture_output=True, text=True).stderr
    for line in out.splitlines():
        if "Video:" in line:
            for tok in line.replace(",", " ").split():
                if "x" in tok and tok.split("x")[0].isdigit() and tok.split("x")[1].isdigit():
                    w, h = tok.split("x")
                    return int(w), int(h)
    return 16, 9


def poster(src, dst, at):
    subprocess.run([FF, "-y", "-loglevel", "error", "-ss", f"{at:.2f}", "-i", src,
                    "-frames:v", "1", "-q:v", "3", dst], check=True)


def latest_randomizer_clips():
    """Earliest and latest clip of the newest path_randomizer recording run."""
    runs = sorted(glob.glob(os.path.join(ROOT, "path_randomizer", "recordings", "run_*")))
    if not runs:
        return {}
    clips = sorted(glob.glob(os.path.join(runs[-1], "eval_*_steps.mp4")))
    # Skip the clip training is still writing (no duration yet).
    clips = [c for c in clips if os.path.getsize(c) > 0 and duration(c) > 0]
    if not clips:
        return {}
    picked = {"ch3_early": clips[0]}
    if len(clips) > 1:
        picked["ch3_late"] = clips[-1]
    return picked


def main():
    os.makedirs(MEDIA, exist_ok=True)
    sources = {k: os.path.join(ROOT, v) for k, v in CLIPS.items()}
    sources.update(latest_randomizer_clips())

    manifest = {}
    for key, src in sources.items():
        if not os.path.exists(src):
            print(f"  missing: {src}")
            continue
        mp4 = os.path.join(MEDIA, key + ".mp4")
        jpg = os.path.join(MEDIA, key + ".jpg")
        shutil.copyfile(src, mp4)
        dur = duration(src)
        # 40% in shows the robot mid-drive; the first frames are the start pose.
        poster(src, jpg, max(0.0, dur * 0.4))
        w, h = size(src)
        step = int(os.path.basename(src).split("_")[1])
        manifest[key] = {"mp4": mp4, "poster": jpg, "seconds": round(dur, 1),
                         "w": w, "h": h, "train_step": step, "source": src}
        print(f"  {key:10s} {step:>9,} steps  {dur:5.1f}s  {w}x{h}")

    with open(os.path.join(MEDIA, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
