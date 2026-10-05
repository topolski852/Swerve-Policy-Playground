# prep_media.py — copy the clips the deck uses into presentation/media/ and
# grab a poster frame (shown before you press play) for each one.
#
# Run from the repo root:  python presentation/prep_media.py

import csv
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
    # Chapter 3: random paths. The 2026-10-03 run that never learned...
    "ch3_fail":  "path_randomizer/recordings/run_20261003_1430/eval_0500000_steps.mp4",
    # ...and the working runs. Every stage-2 clip from these two runs drives the
    # SAME route (RECORD_SEED 1507: 10 waypoints, 2 legs around an obstacle).
    "ch3_150k":  "path_randomizer/recordings/run_20261004_2019/eval_0150000_steps.mp4",
    "ch3_200k":  "path_randomizer/recordings/run_20261004_2019/eval_0200000_steps.mp4",
    "ch3_500k":  "path_randomizer/recordings/run_20261004_2019/eval_0500000_steps.mp4",
    "ch3_1m":    "path_randomizer/recordings/run_20261004_2019/eval_1000000_steps.mp4",
    "ch3_final": "path_randomizer/recordings/run_20261005_0027/eval_2000000_steps.mp4",
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


# Chapter 3 charts, read from the training logs (path_randomizer/logs/).
FAILED_REWARDS = "path_randomizer/logs/rewards_20261003_1430.csv"
ROUTE_TESTS    = ["path_randomizer/logs/route_test_20261004_2019.csv",   # fresh run, 0-1M
                  "path_randomizer/logs/route_test_20261005_0027.csv"]   # resumed overnight, 1M-2.5M


def chart_data():
    """Failed run: mean episode reward per 25k steps. Working run: route-test
    complete/crash rates on the full problem (stage 2), averaged per 50k steps so the
    swings don't hide the trend."""
    out = {}
    path = os.path.join(ROOT, FAILED_REWARDS)
    if os.path.exists(path):
        bins = {}
        for r in csv.DictReader(open(path)):
            b = int(r["timestep"]) // 25_000
            bins.setdefault(b, []).append(float(r["episode_reward"]))
        out["failed"] = [{"k": (b + 1) * 25, "reward": round(sum(v) / len(v), 1)}
                         for b, v in sorted(bins.items())]
    rows = []
    for f in ROUTE_TESTS:
        path = os.path.join(ROOT, f)
        if os.path.exists(path):
            rows += [r for r in csv.DictReader(open(path)) if r["stage"] == "2"]   # the full problem only
    if rows:
        bins = {}
        for r in rows:
            b = (int(r["timestep"]) - 1) // 50_000
            bins.setdefault(b, []).append(r)
        out["route_test"] = [{"k": (b + 1) * 50,
                              "complete": round(100 * sum(float(x["complete"]) for x in v) / len(v)),
                              "crash":    round(100 * sum(float(x["crash"]) for x in v) / len(v))}
                             for b, v in sorted(bins.items())]
    return out


def main():
    os.makedirs(MEDIA, exist_ok=True)
    sources = {k: os.path.join(ROOT, v) for k, v in CLIPS.items()}

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

    charts = chart_data()
    with open(os.path.join(MEDIA, "charts.json"), "w") as f:
        json.dump(charts, f, indent=2)
    print("  charts:", ", ".join(f"{k} ({len(v)} points)" for k, v in charts.items()))


if __name__ == "__main__":
    main()
