"""Core library: frame schema, feature extraction, simulator, pcap reader.

Every data source (simulator, pcap, AWID CSV) is converted to ONE common
"frames" table, so the rest of the pipeline never cares where data came from.

frames columns: time (s), src, dst, type, subtype, rssi (dBm), [label 0/1]
802.11 type 0 = management, 1 = control, 2 = data.
Management subtypes: 4 probe-req, 8 beacon, 10 disassoc, 12 deauth.
"""
import numpy as np
import pandas as pd

BCAST = "ff:ff:ff:ff:ff:ff"
FEATURES = ["n_frames", "frames_per_sec", "n_mgmt", "n_data", "n_beacon",
            "n_deauth", "n_disassoc", "deauth_ratio", "n_unique_dst",
            "bcast_ratio", "rssi_mean", "rssi_std"]


# ---------------------------------------------------------------- features
def extract_features(frames: pd.DataFrame, window: float = 1.0) -> pd.DataFrame:
    """One row per (time window, transmitter MAC)."""
    df = frames.copy()
    df["win"] = np.floor((df["time"] - df["time"].min()) / window).astype(int)
    df["is_mgmt"] = (df["type"] == 0)
    df["is_data"] = (df["type"] == 2)
    df["is_beacon"] = df.is_mgmt & (df["subtype"] == 8)
    df["is_deauth"] = df.is_mgmt & (df["subtype"] == 12)
    df["is_disassoc"] = df.is_mgmt & (df["subtype"] == 10)
    df["is_bcast"] = (df["dst"] == BCAST)
    g = df.groupby(["win", "src"])
    out = pd.DataFrame({
        "n_frames": g.size(),
        "n_mgmt": g["is_mgmt"].sum(),
        "n_data": g["is_data"].sum(),
        "n_beacon": g["is_beacon"].sum(),
        "n_deauth": g["is_deauth"].sum(),
        "n_disassoc": g["is_disassoc"].sum(),
        "n_unique_dst": g["dst"].nunique(),
        "bcast_ratio": g["is_bcast"].mean(),
        "rssi_mean": g["rssi"].mean(),
        "rssi_std": g["rssi"].std().fillna(0.0),
    })
    out["frames_per_sec"] = out["n_frames"] / window
    out["deauth_ratio"] = out["n_deauth"] / out["n_frames"]
    if "label" in df:
        out["label"] = g["label"].max()
    out = out.reset_index()
    out["win_start"] = df["time"].min() + out["win"] * window
    return out


# --------------------------------------------------------------- simulator
def simulate(duration=300, seed=0, n_attacks=8):
    """Generate synthetic 802.11 frame metadata (NOTHING is transmitted).

    Normal: beacons, data, probe requests, rare legitimate deauths, plus a
    'noisy but benign' scanner (probe bursts) so rate alone is not enough.
    Attacks: deauth floods, either spoofing the AP's MAC or using the
    attacker's own, sent to broadcast or to one victim, 1.5-300 frames/s.
    """
    rng = np.random.default_rng(seed)
    rows = []

    def emit(src, dst, typ, sub, rate, t0, t1, mu, sd, label):
        n = rng.poisson(rate * (t1 - t0))
        if n == 0:
            return
        t = rng.uniform(t0, t1, n)
        rows.append(pd.DataFrame({
            "time": t, "src": src, "dst": dst, "type": typ, "subtype": sub,
            "rssi": rng.normal(mu, sd, n).round(), "label": label}))

    mac = lambda: ":".join(f"{b:02x}" for b in rng.integers(0, 256, 6) | np.array([2, 0, 0, 0, 0, 0]) & 0xfe | 2)
    aps = [(mac(), rng.uniform(-60, -35)) for _ in range(2)]
    stas = [(mac(), rng.uniform(-75, -40)) for _ in range(8)]
    for ap, mu in aps:
        emit(ap, BCAST, 0, 8, 9.8, 0, duration, mu, 2, 0)           # beacons
        emit(ap, BCAST, 0, 12, 0.25, 0, duration, mu, 2, 0)          # rare legit deauth
        for sta, _ in stas[:4]:
            emit(ap, sta, 2, 0, rng.uniform(3, 30), 0, duration, mu, 2, 0)
    for sta, mu in stas:
        ap = aps[rng.integers(0, 2)][0]
        emit(sta, ap, 2, 0, rng.uniform(2, 40), 0, duration, mu, 3, 0)
        emit(sta, BCAST, 0, 4, rng.uniform(0.1, 1.5), 0, duration, mu, 3, 0)
    # benign noisy scanner (bursts of probe requests)
    sc = mac()
    for _ in range(6):
        s = rng.uniform(0, duration - 10)
        emit(sc, BCAST, 0, 4, rng.uniform(30, 120), s, s + rng.uniform(3, 8), -55, 3, 0)
    # attacks
    for _ in range(n_attacks):
        s = rng.uniform(5, duration - 25)
        e = s + rng.uniform(8, 20)
        rate = float(np.exp(rng.uniform(np.log(1.5), np.log(300))))
        victim_ap, ap_mu = aps[rng.integers(0, 2)]
        src = victim_ap if rng.random() < 0.5 else mac()             # spoofed or own MAC
        dst = BCAST if rng.random() < 0.5 else stas[rng.integers(0, 8)][0]
        sub = 12 if rng.random() < 0.85 else 10                      # mostly deauth, some disassoc
        emit(src, dst, 0, sub, rate, s, e, rng.uniform(-75, -30), 3, 1)
    df = pd.concat(rows).sort_values("time").reset_index(drop=True)
    return df


# ------------------------------------------------------------ pcap reading
def pcap_to_frames(path, max_packets=None):
    """Read an 802.11 monitor-mode pcap (RadioTap) into a frames table."""
    from scapy.layers.dot11 import Dot11, RadioTap
    from scapy.utils import PcapReader
    recs = []
    with PcapReader(path) as pr:
        for i, pkt in enumerate(pr):
            if max_packets and i >= max_packets:
                break
            r = parse_packet(pkt)
            if r:
                recs.append(r)
    return pd.DataFrame(recs)


def parse_packet(pkt):
    from scapy.layers.dot11 import Dot11, RadioTap
    if not pkt.haslayer(Dot11):
        return None
    d = pkt[Dot11]
    if d.addr2 is None:            # ACK/CTS have no transmitter address
        return None
    rssi = getattr(pkt[RadioTap], "dBm_AntSignal", None) if pkt.haslayer(RadioTap) else None
    return {"time": float(pkt.time), "src": d.addr2, "dst": d.addr1 or BCAST,
            "type": int(d.type), "subtype": int(d.subtype),
            "rssi": float(rssi) if rssi is not None else -90.0}


def label_by_intervals(frames, intervals):
    """For your OWN lab captures: mark deauth/disassoc frames inside the time
    spans (seconds from capture start) when you ran your authorised test."""
    t = frames["time"] - frames["time"].min()
    lab = np.zeros(len(frames), dtype=int)
    is_attack_type = (frames["type"] == 0) & frames["subtype"].isin([10, 12])
    for a, b in intervals:
        lab |= ((t >= a) & (t <= b) & is_attack_type).values.astype(int)
    frames = frames.copy()
    frames["label"] = lab
    return frames


def awid3_csv_to_frames(path):
    """Best-effort loader for an AWID3 CSV export (Wireshark-style columns).
    Adjust column names to match the CSV you download."""
    d = pd.read_csv(path, low_memory=False)
    st = d["wlan.fc.type_subtype"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    typ = d["wlan.fc.type"] if "wlan.fc.type" in d else (st // 16)
    out = pd.DataFrame({
        "time": d["frame.time_epoch"].astype(float),
        "src": d["wlan.sa"].fillna(d.get("wlan.ta", "00:00:00:00:00:00")),
        "dst": d["wlan.da"].fillna(BCAST),
        "type": typ.astype(float).fillna(0).astype(int),
        "subtype": (st % 16).fillna(0).astype(int) if st.max() > 15 else st.fillna(0).astype(int),
        "rssi": pd.to_numeric(d.get("radiotap.dbm_antsignal", -90), errors="coerce").fillna(-90),
    })
    if "Label" in d:
        out["label"] = (d["Label"].astype(str).str.lower().str.contains("deauth")).astype(int)
    return out.dropna(subset=["src"])