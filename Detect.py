"""Live WIDS/WIPS.
  Replay a capture : python detect.py --pcap demo.pcap [--wips]
  Live (monitor-mode interface on YOUR lab network, needs root):
                     sudo python detect.py --iface wlan0mon --wips
Frames are grouped into 1-second windows; each (window, transmitter MAC) row
is classified. Malicious => ALERT (IDS). With --wips the MAC is also written
to blocklist.json (simple automatic response).

LIVE CONFUSION MATRIX: if you know when the attack(s) really happened (seconds
from the start of the capture, same format as train.py's --attack), pass
--attack to score each window against that ground truth as it's seen. The
confusion matrix plot (live_confusion.png) is redrawn every time a real
attack window is processed, so it updates live as attacks occur.
Example:  python detect.py --pcap demo.pcap --wips --attack 10-25,60-78"""
import argparse, json, time, datetime, subprocess
import numpy as np
import joblib, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import Wids

ap = argparse.ArgumentParser()
src = ap.add_mutually_exclusive_group(required=True)
src.add_argument("--pcap"); src.add_argument("--iface")
ap.add_argument("--model", default="wids_model.joblib")
ap.add_argument("--window", type=float, default=1.0)
ap.add_argument("--wips", action="store_true", help="enable automatic response")
ap.add_argument("--enforce", action="store_true", help="actually run the block command (needs sudo/ebtables on a bridge host); requires --wips")
ap.add_argument("--blocklist", default="blocklist.json")
ap.add_argument("--alert-log", default="alerts.log")
ap.add_argument("--attack", help="known attack intervals 'a-b,c-d' (seconds from capture start) for a LIVE ground-truth confusion matrix")
ap.add_argument("--live-plot", default="live_confusion.png")
a = ap.parse_args()

model = joblib.load(a.model)
blocked = {}
buf, win_start = [], None
stats = {"windows": 0, "alerts": 0}
cm = {"TN": 0, "FP": 0, "FN": 0, "TP": 0}          # running confusion counts
t0 = {"t": None}                                    # capture start, set on first frame
intervals = ([tuple(map(float, s.split("-"))) for s in a.attack.split(",")]
             if a.attack else None)

def now(t): return datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S")

def enforce_block(mac):
    subprocess.run(["sudo", "ebtables", "-A", "INPUT", "-s", mac, "-j", "DROP"], check=False)

def label_window(df):
    """Mark deauth/disassoc frames inside the known --attack spans as truth=1,
    using ONE fixed t0 (the capture's first-ever frame) so spans line up the
    same way across every window, not just this buffer."""
    t = df["time"] - t0["t"]
    is_kill = (df["type"] == 0) & df["subtype"].isin([10, 12])
    lab = np.zeros(len(df), dtype=int)
    for s, e in intervals:
        lab |= ((t >= s) & (t <= e) & is_kill).values.astype(int)
    df = df.copy(); df["label"] = lab
    return df

def update_live_plot():
    arr = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    total = arr.sum(); acc = (arr[0, 0] + arr[1, 1]) / total if total else 0
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(arr, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if arr[i, j] > arr.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "malicious"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["normal", "malicious"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (from --attack)")
    ax.set_title(f"Live confusion matrix  n={total}  acc={acc:.3f}")
    plt.tight_layout(); plt.savefig(a.live_plot, dpi=150); plt.close(fig)

def flush():
    global buf
    if not buf: return
    df = pd.DataFrame(buf)
    if intervals is not None:
        df = label_window(df)
    feats = Wids.extract_features(df, a.window)
    feats["pred"] = model.predict(feats[Wids.FEATURES])
    stats["windows"] += len(feats)

    if "label" in feats:
        attack_window_seen = bool(feats.label.max())
        for lbl, pred in zip(feats.label, feats.pred):
            if lbl == 0 and pred == 0: cm["TN"] += 1
            elif lbl == 0 and pred == 1: cm["FP"] += 1
            elif lbl == 1 and pred == 0: cm["FN"] += 1
            else: cm["TP"] += 1
        if attack_window_seen:
            update_live_plot()
            print(f"   [live confusion matrix updated -> {a.live_plot}]  "
                  f"TN={cm['TN']} FP={cm['FP']} FN={cm['FN']} TP={cm['TP']}")

    for _, r in feats[feats.pred == 1].iterrows():
        stats["alerts"] += 1
        msg = (f"[{now(r.win_start)}] ALERT possible deauth/disassoc flood from {r.src} "
               f"({int(r.n_deauth + r.n_disassoc)} mgmt-kill frames/s, {r.frames_per_sec:.0f} fps, "
               f"rssi {r.rssi_mean:.0f} dBm)")
        print(msg); open(a.alert_log, "a").write(msg + "\n")
        if a.wips and r.src not in blocked:
            blocked[r.src] = {"first_seen": now(r.win_start), "rssi": round(r.rssi_mean),
                              "reason": "deauth flood"}
            json.dump(blocked, open(a.blocklist, "w"), indent=2)
            print(f"   WIPS: {r.src} added to {a.blocklist} "
                  f"(lab hook: ebtables -A INPUT -s {r.src} -j DROP)")
            if a.enforce:
                enforce_block(r.src)
                print(f"   WIPS: enforced block for {r.src}")
    buf = []

def on_frame(f):
    global win_start
    if f is None: return
    if t0["t"] is None: t0["t"] = f["time"]
    if win_start is None: win_start = f["time"]
    if f["time"] - win_start >= a.window:
        flush(); win_start = f["time"]
    buf.append(f)

if a.pcap:
    from scapy.all import PcapReader
    with PcapReader(a.pcap) as pr:
        for pkt in pr: on_frame(Wids.parse_packet(pkt))
    flush()
else:
    from scapy.all import sniff
    print(f"sniffing on {a.iface} (Ctrl-C to stop)")
    try: sniff(iface=a.iface, store=False, prn=lambda p: on_frame(Wids.parse_packet(p)))
    except KeyboardInterrupt: flush()

print(f"done: {stats['windows']} windows scored, {stats['alerts']} alerts")
if intervals is not None:
    total = sum(cm.values()); acc = (cm["TN"] + cm["TP"]) / total if total else 0
    prec = cm["TP"] / (cm["TP"] + cm["FP"]) if (cm["TP"] + cm["FP"]) else 0
    rec = cm["TP"] / (cm["TP"] + cm["FN"]) if (cm["TP"] + cm["FN"]) else 0
    print(f"final: accuracy={acc:.3f} precision={prec:.3f} recall={rec:.3f}  "
          f"TN={cm['TN']} FP={cm['FP']} FN={cm['FN']} TP={cm['TP']}  -> {a.live_plot}")