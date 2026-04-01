from obspy import Stream
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy.clients.fdsn import Client
from datetime import datetime, timezone, timedelta
import os
import yaml
import signal
import sys
import torch
import pandas as pd
import numpy as np
import json

# =========================
# CONFIG
# =========================
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

SEEDLINK_SERVER = cfg.get("seedlink_server", "rtserve.iris.washington.edu")
STATIONS = cfg.get("stations", [])
WINDOW_SEC = cfg.get("window_sec", 60)
FDSN_SERVER = cfg["config"]["fdsn_server"]

OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD STATIONS
# =========================
def load_stations():
    client = Client(FDSN_SERVER)
    rows = []

    for s in STATIONS:
        if not s.get("enabled", True):
            continue

        try:
            inv = client.get_stations(
                network=s["network"],
                station=s["station"],
                level="station"
            )

            sta = inv[0][0]
            sid = f"{s['network']}.{s['station']}"

            rows.append({
                "id": sid,
                "station": sid,
                "x": sta.longitude,
                "y": sta.latitude,
                "z": -sta.elevation / 1000.0
            })

            print(f"✅ Loaded station: {sid}")

        except Exception as e:
            print(f"❌ Failed station {s['station']}:", e)

    return pd.DataFrame(rows)

STATIONS_DF = load_stations()

# =========================
# PHASENET
# =========================
from seisbench.models import PhaseNet

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PhaseNet.from_pretrained("instance").to(device)
model.eval()

# =========================
# GAMMA
# =========================
sys.path.append("GaMMA")
from gamma.utils import association

global_picks = []
buffers = {}

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

                if prob[i] > 0.6 and (t - last_time) > 3.0:

                    if "P" in phase.upper():
                        ph = "P"
                    elif "S" in phase.upper():
                        ph = "S"
                    else:
                        continue

                    pick_time = start_dt + timedelta(seconds=t)

                    picks.append({
                        "timestamp": pick_time,
                        "station": station,
                        "phase": ph,
                        "prob": float(prob[i])
                    })

                    last_time = t

        return picks

    except Exception as e:
        print("❌ PhaseNet ERROR:", e)
        return []

# =========================
# GAMMA
# =========================
def run_gamma():
    global global_picks

    if len(global_picks) < 6:
        return

    df = pd.DataFrame(global_picks)

    if df.empty or "station" not in df.columns:
        return

    now = df["timestamp"].max()
    df = df[df["timestamp"] > now - timedelta(seconds=120)]

    if df.empty:
        return

    df = df.sort_values("timestamp")
    df = df.groupby("station").tail(60)

    df = df.merge(STATIONS_DF, left_on="station", right_on="id", how="left")
    df = df.dropna(subset=["x", "y", "z"])

    if df.empty or df["station"].nunique() < 3:
        return

    df["id"] = df["station"] + "_" + df["timestamp"].astype(str)
    df["type"] = df["phase"]
    df["phase_time"] = df["timestamp"]
    df["phase_type"] = df["phase"]
    df["phase_score"] = df["prob"]
    df = df.rename(columns={"station": "station_id"})

    config = {
        "dims": ["x", "y", "z"],
        "method": "BGMM",
        "vel": {"p": 6.0, "s": 3.5},
    }

    try:
        events, _ = association(df, STATIONS_DF, config)

        if len(events) > 0:
            print("\n🌋 EVENTS:", events)

    except Exception as e:
        print("❌ GAMMA ERROR:", e)

# =========================
# SEEDLINK
# =========================
class SeedlinkMonitor(EasySeedLinkClient):

    def on_data(self, trace):

        net = trace.stats.network
        sta = trace.stats.station
        loc = trace.stats.location or "00"
        cha = trace.stats.channel

        if not cha.endswith("Z"):
            return

        key = f"{net}.{sta}.{loc}.{cha}"

        trace.data = trace.data.astype("float32")

        if key not in buffers:
            buffers[key] = Stream()

        buffers[key] += trace
        buffers[key].merge(method=1, fill_value=0)

        tr = buffers[key][0]

        if tr.stats.npts < tr.stats.sampling_rate * 10:
            return

        duration = tr.stats.endtime - tr.stats.starttime
        if duration < WINDOW_SEC:
            return

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
        station_id = f"{net}.{sta}"
        picks = run_phasenet(buffers[key], station_id, start_dt)

        print(f"⚡ {station_id} PICKS:", len(picks))

        for p in picks:
            global_picks.append(p)
            print(f"📍 {p['phase']} | {p['timestamp']} | {p['prob']:.2f}")

        # =========================
        # SAVE MSEED
        # =========================
        if np.ma.isMaskedArray(tr.data):
            tr.data = tr.data.filled(0)

        ts_iso = start_dt.strftime("%Y-%m-%dT%H-%M-%SZ")

        folder = os.path.join(
            "out", "waveform.win",
            start_dt.strftime("%Y"),
            start_dt.strftime("%m"),
            start_dt.strftime("%d")
        )
        os.makedirs(folder, exist_ok=True)

        filename = f"{key}__{ts_iso}.mseed"
        path = os.path.join(folder, filename)

        buffers[key].write(path, format="MSEED")

        # =========================
        # SAVE JSON + PICKS 🔥
        # =========================
        picks_json = [
            {
                "phase": p["phase"],
                "time": p["timestamp"].isoformat(),
                "prob": round(p["prob"], 3)
            }
            for p in picks
        ]

        metadata = {
            "key": key,
            "network": net,
            "station": sta,
            "channel": cha,
            "starttime_utc": start_dt.isoformat(),
            "endtime_utc": tr.stats.endtime.datetime.replace(tzinfo=timezone.utc).isoformat(),
            "sampling_rate": tr.stats.sampling_rate,
            "npts": tr.stats.npts,
            "picks": picks_json  # 🔥 INI YANG KAMU MAU
        }

        json_path = path.replace(".mseed", ".json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print("💾 Saved:", path)
        print("📄 JSON:", json_path)

        # =========================
        # SHIFT WINDOW
        # =========================
        buffers[key].trim(
            starttime=aligned_start + WINDOW_SEC,
            nearest_sample=False
        )

# =========================
# MAIN
# =========================
client = SeedlinkMonitor(SEEDLINK_SERVER)

print("🚀 START REALTIME PIPELINE")

for sta in STATIONS:
    if not sta.get("enabled", True):
        continue

    channel = sta.get("channel", "BHZ")
    client.select_stream(sta["network"], sta["station"], channel)

client.run()