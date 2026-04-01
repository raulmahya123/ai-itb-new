from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy import Stream
from datetime import timezone, timedelta
import os
import json
import yaml
import signal
import sys
import torch
import numpy as np

# =========================
# CONFIG
# =========================
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

SEEDLINK_SERVER = cfg.get("seedlink_server", "rtserve.iris.washington.edu")
STATIONS = cfg.get("stations", [])
WINDOW_SEC = cfg.get("window_sec", 60)

buffers = {}
running = True

# =========================
# PHASENET LOAD
# =========================
from seisbench.models import PhaseNet

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PhaseNet.from_pretrained("instance").to(device)
model.eval()

# =========================
# UTILS
# =========================
def safe(v, default=""):
    if v is None:
        return default
    v = str(v).strip()
    return v if v != "" else default

def shutdown(sig, frame):
    global running
    print("\n🛑 Shutting down...")
    running = False
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)

# =========================
# PHASENET
# =========================
def run_phasenet(stream, station, start_dt):
    try:
        st = stream.copy()
        st.detrend("demean")

        tr = st[0]
        sr = tr.stats.sampling_rate

        nyquist = sr / 2
        freqmax = min(0.45 * sr, nyquist - 0.1)

        st.filter("bandpass", freqmin=1, freqmax=freqmax)
        st.normalize()

        if tr.stats.npts < sr * 10:
            return []

        annotations = model.annotate(st)
        picks = []

        for tr in annotations:
            prob = tr.data
            sr = tr.stats.sampling_rate
            phase = tr.stats.channel

            last_time = -999

            for i in range(len(prob)):
                t = i / sr

                if prob[i] > 0.6 and (t - last_time) > 3:

                    if "P" in phase.upper():
                        ph = "P"
                    elif "S" in phase.upper():
                        ph = "S"
                    else:
                        continue

                    pick_time = start_dt + timedelta(seconds=t)

                    picks.append({
                        "phase": ph,
                        "time": pick_time.isoformat(),
                        "prob": float(prob[i])
                    })

                    last_time = t

        return picks

    except Exception as e:
        print("❌ PhaseNet ERROR:", e)
        return []

# =========================
# SEEDLINK CLIENT
# =========================
class SeedlinkMonitor(EasySeedLinkClient):

    def on_data(self, trace):

        net = safe(trace.stats.network)
        sta = safe(trace.stats.station)

        loc = safe(trace.stats.location) or "00"
        cha = safe(trace.stats.channel)

        if not cha.endswith("Z"):
            return

        key = f"{net}.{sta}.{loc}.{cha}"

        trace.data = trace.data.astype("float32")

        if key not in buffers:
            buffers[key] = Stream()

        buffers[key] += trace

        # 🔥 FIX GAP
        buffers[key].merge(method=1, fill_value=0)

        tr = buffers[key][0]
        duration = tr.stats.endtime - tr.stats.starttime

        if duration < WINDOW_SEC:
            return

        # ALIGN WINDOW
        start = tr.stats.starttime
        aligned_start = start - (start.timestamp % WINDOW_SEC)

        buffers[key].trim(
            starttime=aligned_start,
            endtime=aligned_start + WINDOW_SEC
        )

        tr = buffers[key][0]

        start_dt = tr.stats.starttime.datetime.replace(tzinfo=timezone.utc)

        # =========================
        # PHASENET
        # =========================
        picks = run_phasenet(buffers[key], f"{net}.{sta}", start_dt)

        print(f"⚡ {net}.{sta} PICKS:", len(picks))

        # =========================
        # SAVE MSEED
        # =========================
        if np.ma.isMaskedArray(tr.data):
            tr.data = tr.data.filled(0)

        ts_iso = start_dt.strftime("%Y-%m-%dT%H-%M-%SZ")
        ts = int(start_dt.timestamp() * 1000)

        folder = os.path.join(
            "out", "waveform.win",
            start_dt.strftime("%Y"),
            start_dt.strftime("%m"),
            start_dt.strftime("%d")
        )
        os.makedirs(folder, exist_ok=True)

        filename = f"{key}__{ts_iso}.mseed"
        mseed_path = os.path.join(folder, filename)

        buffers[key].write(mseed_path, format="MSEED")

        # =========================
        # DATA QUALITY
        # =========================
        gaps = buffers[key].get_gaps()
        gap_count = len(gaps)
        overlap_count = sum(1 for g in gaps if g[6] < 0)

        # =========================
        # JSON PICKS
        # =========================
        picks_json = [
            {
                "phase": p["phase"],
                "time": p["time"],
                "prob": round(p["prob"], 3)
            }
            for p in picks
        ]

        # =========================
        # FULL METADATA
        # =========================
        metadata = {
            "key": key,
            "ts": ts,
            "ts_iso_utc": ts_iso,
            "mseed_path": mseed_path,

            "network": net,
            "station": sta,
            "location": loc,
            "channel": cha,

            "starttime_utc": ts_iso,
            "endtime_utc": tr.stats.endtime.datetime.replace(tzinfo=timezone.utc).isoformat(),

            "sampling_rate": tr.stats.sampling_rate,
            "npts": tr.stats.npts,
            "duration_sec": round(duration, 2),

            "gap_count": gap_count,
            "overlap_count": overlap_count,
            "pct_filled": 1 if gap_count == 0 else 0,

            "bytes": os.path.getsize(mseed_path),
            "window_sec": WINDOW_SEC,

            # 🔥 PICKS
            "picks": picks_json,
            "pick_count": len(picks_json),
            "has_picks": len(picks_json) > 0
        }

        json_path = mseed_path.replace(".mseed", ".json")

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Saved: {mseed_path}")
        print(f"📄 JSON: {json_path}")

        # SHIFT WINDOW
        buffers[key].trim(
            starttime=aligned_start + WINDOW_SEC,
            nearest_sample=False
        )

# =========================
# INIT
# =========================
client = SeedlinkMonitor(SEEDLINK_SERVER)

print("🚀 Connecting SeedLink:", SEEDLINK_SERVER)

for sta in STATIONS:
    if not sta.get("enabled", True):
        continue

    channel = sta.get("channel", "BHZ")

    print(f"📡 Subscribe: {sta['network']}.{sta['station']}.{channel}")
    client.select_stream(sta["network"], sta["station"], channel)

# =========================
# RUN
# =========================
client.run()