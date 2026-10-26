"""Write the SIMULATED frames (new seed) to a pcap file so detect.py can be
demonstrated offline. Nothing is transmitted - it only writes a file."""
from scapy.layers.dot11 import RadioTap, Dot11, Dot11Deauth
from scapy.utils import wrpcap
import Wids
df = Wids.simulate(duration=120, seed=7, n_attacks=3)
pkts = []
for r in df.itertuples():
    p = RadioTap(present="dBm_AntSignal", dBm_AntSignal=int(r.rssi)) / Dot11(type=r.type, subtype=r.subtype,
            addr1=r.dst, addr2=r.src, addr3=r.src)
    if r.type == 0 and r.subtype == 12: p = p / Dot11Deauth(reason=7)
    p.time = 1_700_000_000 + r.time
    pkts.append(p)
wrpcap("demo.pcap", pkts)
atk = df[df.label == 1]
print(f"wrote demo.pcap ({len(pkts)} frames); ground-truth attack spans (s):")
print(atk.groupby("src").time.agg(["min", "max", "size"]).round(1))
spans = atk.groupby("src").time.agg(["min", "max"])
attack_str = ",".join(f"{r['min']:.0f}-{r['max']+1:.0f}" for _, r in spans.iterrows())
print(f"\ncopy-paste for detect.py's live confusion matrix:\n"
      f"  python detect.py --pcap demo.pcap --wips --attack {attack_str}")