from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy import Stream
from datetime import timezone
import os
import json
import yaml
import signal
import sys

# =========================
# CONFIG
# =========================
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

SEEDLINK_SERVER = cfg.get("seedlink_server", "rtserve.iris.washington.edu")
STATIONS = cfg.get("stations", [])
WINDOW_SEC = cfg.get("window_sec", 60)

# =========================
# GLOBAL
# =========================
buffers = {}
running = True

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
# SEEDLINK CLIENT
# =========================
class SeedlinkMonitor(EasySeedLinkClient):

    def on_data(self, trace):

        net = safe(trace.stats.network)
        sta = safe(trace.stats.station)

        # FIX LOCATION
        loc = safe(trace.stats.location)
        if loc == "":
            loc = "00"

        cha = safe(trace.stats.channel)

        key = f"{net}.{sta}.{loc}.{cha}"

        trace.data = trace.data.astype("float32")

        if key not in buffers:
            buffers[key] = Stream()

        buffers[key] += trace
        buffers[key].merge(method=1)

        tr = buffers[key][0]
        duration = tr.stats.endtime - tr.stats.starttime

        if duration < WINDOW_SEC:
            return

        # =========================
        # ALIGN WINDOW (FIX PRESISI)
        # =========================
        start = tr.stats.starttime
        aligned_start = start - (start.timestamp % WINDOW_SEC)

        buffers[key].trim(
            starttime=aligned_start,
            endtime=aligned_start + WINDOW_SEC
        )

        tr = buffers[key][0]

        # =========================
        # TIME (🔥 FIX UTC EXPLICIT)
        # =========================
        start_dt = tr.stats.starttime.datetime.replace(tzinfo=timezone.utc)

        ts_iso = start_dt.strftime("%Y-%m-%dT%H-%M-%SZ")
        ts = int(start_dt.timestamp() * 1000)

        # =========================
        # PATH
        # =========================
        folder = os.path.join(
            "out", "waveform.win",
            start_dt.strftime("%Y"),
            start_dt.strftime("%m"),
            start_dt.strftime("%d")
        )
        os.makedirs(folder, exist_ok=True)

        filename = f"{key}__{ts_iso}.mseed"
        mseed_path = os.path.join(folder, filename)

        # =========================
        # WRITE MSEED
        # =========================
        try:
            buffers[key].write(mseed_path, format="MSEED")
        except Exception as e:
            print("❌ WRITE ERROR:", e)
            buffers[key] = Stream()
            return

        # =========================
        # METADATA
        # =========================
        gaps = buffers[key].get_gaps()
        gap_count = len(gaps)
        overlap_count = sum(1 for g in gaps if g[6] < 0)

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
            "window_sec": WINDOW_SEC
        }

        # =========================
        # SAVE JSON
        # =========================
        try:
            with open(mseed_path.replace(".mseed", ".json"), "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print("❌ JSON ERROR:", e)

        print(f"✅ Window saved: {mseed_path}")

        # =========================
        # SHIFT WINDOW
        # =========================
        buffers[key].trim(
            starttime=aligned_start + WINDOW_SEC,
            nearest_sample=False
        )

# =========================
# INIT CLIENT
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