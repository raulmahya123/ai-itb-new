from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy import Stream
import os
import json
import yaml

WINDOW_SEC = 60
buffers = {}

# =========================
# LOAD CONFIG YAML
# =========================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# SeedLink server
server = config.get("seedlink_server", "rtserve.iris.washington.edu")

# =========================
# HELPER
# =========================
def safe(v, default=""):
    if v is None:
        return default
    v = str(v).strip()
    if v == "":
        return default
    return v.replace(" ", "")

# =========================
# SEEDLINK CLIENT
# =========================
class SeedlinkMonitor(EasySeedLinkClient):

    def on_data(self, trace):

        net = safe(trace.stats.network)
        sta = safe(trace.stats.station)
        loc = safe(trace.stats.location, "00")
        cha = safe(trace.stats.channel)

        key = f"{net}.{sta}.{loc}.{cha}"

        trace.data = trace.data.astype("float32")

        if key not in buffers:
            buffers[key] = Stream()

        buffers[key] += trace
        buffers[key].merge(method=1)

        tr = buffers[key][0]
        duration = tr.stats.endtime - tr.stats.starttime

        if duration >= WINDOW_SEC:

            ts_iso = tr.stats.starttime.strftime("%Y-%m-%dT%H-%M-%SZ")
            ts = int(tr.stats.starttime.timestamp * 1000)

            year = tr.stats.starttime.strftime("%Y")
            month = tr.stats.starttime.strftime("%m")
            day = tr.stats.starttime.strftime("%d")

            folder = os.path.join("out", "waveform.win", year, month, day)
            os.makedirs(folder, exist_ok=True)

            filename = f"{key}__{ts_iso}.mseed"
            mseed_path = os.path.join(folder, filename)

            # =========================
            # SAVE MSEED
            # =========================
            try:
                buffers[key].write(mseed_path, format="MSEED")
            except Exception as e:
                print("WRITE ERROR:", e)
                buffers[key] = Stream()
                return

            # =========================
            # METADATA
            # =========================
            gaps = buffers[key].get_gaps()
            gap_count = len(gaps)
            overlap_count = sum(1 for g in gaps if g[6] < 0)

            pct_filled = 1 if gap_count == 0 else 0

            metadata = {
                "key": key,
                "ts": ts,
                "ts_iso_utc": ts_iso,
                "mseed_path": mseed_path,
                "network": net,
                "station": sta,
                "location": loc,
                "channel": cha,
                "starttime": str(tr.stats.starttime),
                "endtime": str(tr.stats.endtime),
                "sampling_rate": tr.stats.sampling_rate,
                "npts": tr.stats.npts,
                "duration_sec": round(duration, 2),
                "gap_count": gap_count,
                "overlap_count": overlap_count,
                "pct_filled": pct_filled,
                "bytes": os.path.getsize(mseed_path),
                "version": "1.0",
                "window_sec": WINDOW_SEC
            }

            # =========================
            # SAVE JSON
            # =========================
            json_file = mseed_path.replace(".mseed", ".json")

            try:
                with open(json_file, "w") as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                print("JSON ERROR:", e)

            print("✅ Saved:", mseed_path)

            buffers[key] = Stream()


# =========================
# INIT CLIENT
# =========================
client = SeedlinkMonitor(server)

# =========================
# REGISTER STATIONS
# =========================
for sta in config["stations"]:

    if not sta.get("enabled", True):
        continue

    net = sta["network"]
    station = sta["station"]
    cha = sta["channel"]

    print(f"📡 Subscribe: {net}.{station}.{cha}")

    client.select_stream(net, station, cha)

# =========================
# RUN CLIENT
# =========================
client.run()