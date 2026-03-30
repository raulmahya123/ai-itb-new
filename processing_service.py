from obspy import read, Stream
from obspy.clients.fdsn import Client
import numpy as np
import asyncio
import torch
import pandas as pd
import glob
import os
import yaml
from datetime import datetime, timezone, timedelta

# =========================
# CONFIG
# =========================
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

FDSN_SERVER = cfg["config"]["fdsn_server"]
STATION_LIST = cfg["stations"]

# =========================
# LOAD STATIONS
# =========================
def load_stations():
    client = Client(FDSN_SERVER)
    rows = []

    for s in STATION_LIST:
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
model = PhaseNet.from_pretrained("instance")
model.to(device)
model.eval()

# =========================
# GAMMA
# =========================
import sys
sys.path.append("GaMMA")
from gamma.utils import association

global_picks = []

# =========================
# PHASENET
# =========================
def run_phasenet_sync(stream):
    try:
        annotations = model.annotate(stream)
        picks = []

        for tr in annotations:
            prob = tr.data
            phase = tr.stats.channel
            sr = tr.stats.sampling_rate

            last = -9999

            for i in range(len(prob)):
                if prob[i] > 0.75 and (i - last) > 800:

                    if "P" in phase.upper():
                        ph = "P"
                    elif "S" in phase.upper():
                        ph = "S"
                    else:
                        continue

                    picks.append({
                        "time": i / sr,
                        "type": ph,
                        "prob": float(prob[i])
                    })
                    last = i

        return picks

    except Exception as e:
        print("❌ PhaseNet ERROR:", e)
        return []

# =========================
# GAMMA FINAL (TUNED)
# =========================
def run_gamma():
    global global_picks

    if len(global_picks) < 6:
        return []

    df = pd.DataFrame(global_picks)

    # 🔥 WINDOW LEBIH BESAR (BIAR STATION MASUK)
    now = df["timestamp"].max()
    df = df[df["timestamp"] > now - timedelta(seconds=120)]

    print("\n📊 AFTER TIME FILTER:", len(df))

    if df.empty:
        return []

    df = df.sort_values("timestamp")

    # 🔥 LEBIH LONGGAR
    df = df.groupby("station").tail(60)

    # =========================
    # CLUSTER (LEBIH FLEXIBLE)
    # =========================
    if len(df) > 10:
        mid = df["timestamp"].median()

        df_cluster = df[
            (df["timestamp"] > mid - timedelta(seconds=15)) &
            (df["timestamp"] < mid + timedelta(seconds=15))
        ]

        if df_cluster["station"].nunique() >= 3:
            df = df_cluster

    print("📊 AFTER CLUSTER:", len(df))

    # =========================
    # MERGE
    # =========================
    df = df.merge(STATIONS_DF, left_on="station", right_on="id", how="left")

    if "station_x" in df.columns:
        df = df.rename(columns={"station_x": "station"})

    df = df.drop(columns=[c for c in ["station_y", "id_y"] if c in df.columns])

    print("📊 AFTER MERGE:", len(df))

    print("\n📊 STATIONS USED:")
    print(df["station"].unique())
    print("📊 STATION COUNT:", df["station"].nunique())

    df = df.dropna(subset=["x", "y", "z"])

    if df["station"].nunique() < 3:
        print("⚠️ Not enough stations")
        return []

    # =========================
    # SPREAD
    # =========================
    spread = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
    print(f"⏱ SPREAD: {spread:.2f}s")

    if spread > 18:
        print("⚠️ Spread terlalu besar")
        return []

    print("\n📊 PHASE COUNT:")
    print(df["phase"].value_counts())

    # =========================
    # FORMAT FINAL
    # =========================
    df["id"] = df["station"] + "_" + df["timestamp"].astype(str)
    df["type"] = df["phase"]
    df["phase_time"] = df["timestamp"]
    df["phase_type"] = df["phase"]
    df["phase_score"] = df["prob"]

    df = df.rename(columns={"station": "station_id"})

    # =========================
    # CONFIG SUPER TUNED 🔥
    # =========================
    config = {
        "dims": ["x", "y", "z"],
        "method": "BGMM",
        "oversample_factor": 10,
        "time_sigma": 5.0,
        "vel": {"p": 6.0, "s": 3.5},
        "min_picks_per_eq": 1,
        "use_amplitude": False
    }

    try:
        events, _ = association(df, STATIONS_DF, config)

        print("\n📊 EVENTS RAW:", events)
        print("\n🌋 GAMMA EVENTS:", events)

        global_picks = []
        return events

    except Exception as e:
        print("❌ GaMMA ERROR:", e)
        return []

# =========================
# PROCESS FILE
# =========================
async def process_file(path):
    try:
        filename = os.path.basename(path)
        parts = filename.split("__")[0].split(".")
        station = f"{parts[0]}.{parts[1]}"

        print("📡 FILE STATION:", station)

        st = read(path)

        st.merge(method=1, fill_value="interpolate")
        st = st.select(channel="BHZ")

        if len(st) == 0:
            return

        tr = max(st, key=lambda x: x.stats.npts)

        if tr.stats.npts < 1000:
            return

        st = Stream([tr])
        st.detrend("demean")
        st.normalize()

        picks = await asyncio.to_thread(run_phasenet_sync, st)

        print(f"⚡ Picks: {len(picks)}")

        start_dt = tr.stats.starttime.datetime.replace(tzinfo=timezone.utc)

        for i, p in enumerate(picks):
            pick_time = start_dt + timedelta(seconds=p["time"])

            global_picks.append({
                "id": f"{station}_{i}_{pick_time.timestamp()}",
                "timestamp": pick_time,
                "station": station,
                "phase": p["type"],
                "prob": p["prob"]
            })

        print("📊 GLOBAL PICKS:", len(global_picks))

        run_gamma()

    except Exception as e:
        print("❌ ERROR:", e)

# =========================
# WATCH
# =========================
async def watch_folder(folder):
    processed = set()

    while True:
        files = glob.glob(os.path.join(folder, "**", "*.mseed"), recursive=True)

        for f in files:
            if f not in processed:
                await process_file(f)
                processed.add(f)

        await asyncio.sleep(2)

# =========================
# MAIN
# =========================
async def main():
    DATA_DIR = os.path.join(os.getcwd(), "out", "waveform.win")
    print("🚀 FINAL PIPELINE (EVENT FRIENDLY MODE)")
    await watch_folder(DATA_DIR)

if __name__ == "__main__":
    asyncio.run(main())