"""CLI example:
python Data/DataProcessPipline.py \
  --poi ./Data/CombinedPOI.csv \
  --geolife ./Data/Geolife.csv \
  --model ./SemiSupervised/traj_anomaly_rf.joblib \
  --out-pkl ./Data/beijing_geolife_incident_strict.pkl \
  --out-labels ./Data/beijing_geolife_labels_strict.csv \
  --resolution 10 --max-kmh 110 --max-idle-min 60 --min-points 10 \
  --ratio 0.3 --alpha 4.0 --beta 0.15 --base-scale 0.9 --poisson
"""
from __future__ import annotations
import argparse
import os
import pickle
from typing import Dict, Tuple, Iterable

import h3
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

# ------------------------
# Helpers
# ------------------------
R_EARTH = 6_371_000.0

def haversine(lon1, lat1, lon2, lat2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    return 2*R_EARTH*np.arcsin(np.sqrt(a))


def assign_h3_index(df: gpd.GeoDataFrame, geometry_col: str = "geometry", resolution: int = 11):
    return df.assign(
        h3_index=df[geometry_col].apply(lambda geom: h3.latlng_to_cell(geom.y, geom.x, resolution))
    )

# ------------------------
# Data hygiene filters
# ------------------------

def drop_bad_coords(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    m = (
        df[lat_col].between(-90, 90) &
        df[lon_col].between(-180, 180) &
        df[[lat_col, lon_col]].notna().all(axis=1)
    )
    return df.loc[m].copy()


def dedupe_consecutive(df: pd.DataFrame, subset: Iterable[str]) -> pd.DataFrame:
    mask = (df[list(subset)].shift(1) != df[list(subset)]).any(axis=1)
    mask.iloc[0] = True
    return df.loc[mask].copy()


def enforce_monotonic_time(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    ts = pd.to_datetime(df[time_col])
    m = ts.diff().dt.total_seconds().fillna(1) >= 0
    return df.loc[m].copy()


def filter_speed_spikes(df: pd.DataFrame, max_kmh: float, lat_col: str, lon_col: str, time_col: str) -> pd.DataFrame:
    ts = pd.to_datetime(df[time_col]).astype("int64") / 1e9
    lat = df[lat_col].to_numpy(); lon = df[lon_col].to_numpy()
    dsec = np.r_[np.nan, np.diff(ts)]
    dist = np.r_[np.nan, haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])]
    kmh = (dist / 1000.0) / np.maximum(dsec / 3600.0, 1e-9)
    keep = (np.isnan(kmh)) | (kmh <= max_kmh)
    return df.loc[keep].copy()


def clip_large_gaps(df: pd.DataFrame, time_col: str, max_idle_min: int) -> pd.DataFrame:
    ts = pd.to_datetime(df[time_col])
    gaps = ts.diff().dt.total_seconds().fillna(0)
    keep = gaps <= max_idle_min * 60
    keep.iloc[0] = True
    return df.loc[keep].copy()

# ------------------------
# POI processing (robust)
# ------------------------

def process_data_poi(path_csv: str, nrows: int | None, resolution: int) -> pd.DataFrame:
    print("Loading POI data...")
    poi = pd.read_csv(path_csv, nrows=nrows)
    # normalize columns
    rename_map = {}
    if '小类' in poi.columns: rename_map['小类'] = 'poi_type'
    if '大类' in poi.columns: rename_map['大类'] = 'poi_major'
    lon_col = 'wgs84_x' if 'wgs84_x' in poi.columns else ('lon' if 'lon' in poi.columns else None)
    lat_col = 'wgs84_y' if 'wgs84_y' in poi.columns else ('lat' if 'lat' in poi.columns else None)
    if not lon_col or not lat_col:
        raise ValueError("POI CSV 缺少经纬度列 (wgs84_x/wgs84_y 或 lon/lat)")
    poi.rename(columns=rename_map, inplace=True)

    # hygiene: coordinates
    poi = drop_bad_coords(poi, lat_col, lon_col)

    geometry = [Point(lon, lat) for lon, lat in tqdm(zip(poi[lon_col], poi[lat_col]), total=len(poi[lon_col]), desc="POI geometry...")]
    poi.drop(columns=[c for c in [lon_col, lat_col] if c in poi.columns], inplace=True)
    poi = gpd.GeoDataFrame(poi, geometry=geometry, crs="EPSG:4326")

    print("Assign H3 for POI...")
    poi = assign_h3_index(poi, resolution=resolution)

    print("Aggregate POIs by cell...")
    has_type = 'poi_type' in poi.columns
    agg = poi.groupby('h3_index').agg({ 'poi_type': (lambda x: list(x) if has_type else ['unknown'] * len(x)) }).reset_index()
    agg.rename(columns={'poi_type': 'poi_type_list'}, inplace=True)
    return agg

# ------------------------
# Trajectory processing (robust)
# ------------------------

def process_data_traj(path_csv: str, nrows: int | None, resolution: int,
                      max_kmh: float, min_points: int, max_idle_min: int) -> gpd.GeoDataFrame:
    print("Loading Geolife data...")
    df = pd.read_csv(path_csv, nrows=nrows, index_col=0)
    if 'file_id' in df.columns and 'traj_id' not in df.columns:
        df.rename(columns={'file_id': 'traj_id'}, inplace=True)
    for c in ['zero', 'alt']:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Hygiene pipeline per traj_id
    out = []
    for tid, g in df.groupby('traj_id', sort=False):
        # drop bad coords
        g = drop_bad_coords(g, 'lat', 'lon')
        # build datetime
        g['datetime'] = pd.to_datetime(g['date'] + ' ' + g['time'], errors='coerce')
        g = g.dropna(subset=['datetime'])
        # sort by time and enforce monotonic
        g = g.sort_values('datetime')
        g = enforce_monotonic_time(g, 'datetime')
        # dedupe same (lat,lon,datetime)
        g = dedupe_consecutive(g, ['lat', 'lon', 'datetime'])
        # clip large gaps that likely indicate breaks
        g = clip_large_gaps(g, 'datetime', max_idle_min=max_idle_min)
        # speed spikes
        g2 = g[['lat','lon','datetime']].copy()
        g2['date'] = g['date']; g2['time'] = g['time']
        g2['traj_id'] = tid
        g2 = filter_speed_spikes(g2, max_kmh=max_kmh, lat_col='lat', lon_col='lon', time_col='datetime')
        if len(g2) >= min_points:
            out.append(g2)
    if not out:
        raise ValueError("所有轨迹在质量控制后都被过滤；请放宽阈值或检查数据。")

    dfc = pd.concat(out, ignore_index=True)

    print("Process Geolife geometry + H3...")
    geometry = [Point(lon, lat) for lon, lat in tqdm(zip(dfc['lon'], dfc['lat']), total=len(dfc['lon']), desc="Traj geometry...")]
    dfc = gpd.GeoDataFrame(dfc.drop(columns=['lat','lon']).copy(), geometry=geometry, crs="EPSG:4326")

    dfc = assign_h3_index(dfc, resolution=resolution)

    print("Feature engineering (stay_duration/hour) with run-length compression...")
    # merge consecutive identical cells
    dfc = dfc[dfc['h3_index'] != dfc['h3_index'].shift(1)].reset_index(drop=True)
    dfc['stay_duration'] = dfc.groupby('traj_id')['datetime'].shift(-1) - dfc['datetime']
    dfc['stay_duration'] = dfc['stay_duration'].dt.total_seconds().fillna(0)
    # cap extreme durations (e.g., parked logger)
    dfc['stay_duration'] = dfc['stay_duration'].clip(lower=0, upper=24*3600)
    dfc['hour'] = dfc['datetime'].dt.hour.astype(int)
    for c in ['time','date']:
        if c in dfc.columns:
            dfc.drop(columns=[c], inplace=True)
    return dfc

# ------------------------
# Feature building (traj_id level)
# ------------------------

def build_geolife_features(DataTraj: gpd.GeoDataFrame) -> pd.DataFrame:
    feats = []
    for tid, g in DataTraj.groupby("traj_id"):
        g = g.sort_values("stay_duration").copy()
        lon = g.geometry.x.to_numpy(); lat = g.geometry.y.to_numpy()
        if len(g) >= 2:
            dists = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
        else:
            dists = np.array([0.0])
        durs = g["stay_duration"].fillna(0).clip(lower=0).to_numpy()
        gaps = np.r_[durs[1:], 0.0] if len(durs) > 1 else np.array([0.0])
        speeds = (dists / np.maximum(gaps[:-1], 1.0)) if len(g) > 1 else np.array([0.0])
        hours = g["hour"].astype(int).to_numpy() if "hour" in g else np.array([], dtype=int)
        hour_hist, _ = np.histogram(hours, bins=np.arange(25), density=False) if len(hours) > 0 else (np.zeros(24, dtype=int), None)
        time_span = float(np.sum(durs))
        row = {
            "traj_id": tid,
            "n_stops": int(len(g)),
            "unique_types": 1,
            "entropy_types": 0.0,
            "dur_mean": float(np.mean(durs)) if len(durs)>0 else 0.0,
            "dur_std": float(np.std(durs)) if len(durs)>0 else 0.0,
            "dur_max": float(np.max(durs)) if len(durs)>0 else 0.0,
            "gap_mean": float(np.mean(gaps)) if len(gaps)>0 else 0.0,
            "gap_std": float(np.std(gaps)) if len(gaps)>0 else 0.0,
            "dist_sum": float(np.sum(dists)),
            "dist_mean": float(np.mean(dists)),
            "speed_gap_mean": float(np.mean(speeds)) if len(speeds)>0 else 0.0,
            "speed_gap_max": float(np.max(speeds)) if len(speeds)>0 else 0.0,
            "time_span": float(time_span),
            "stop_density": float(len(g) / (time_span + 1.0) if time_span>0 else 0.0),
        }
        for h in range(24):
            row[f"hour_{h}"] = int(hour_hist[h])
        feats.append(row)
    return pd.DataFrame(feats).fillna(0.0)

# ------------------------
# Labeling with external model
# ------------------------

def label_geolife_with_model(DataTraj: gpd.GeoDataFrame, model_path: str, ratio: float, out_csv: str | None):
    bundle = joblib.load(model_path)
    model = bundle["model"]; feature_cols = bundle["feature_cols"]

    geofeat = build_geolife_features(DataTraj)
    for c in feature_cols:
        if c not in geofeat.columns:
            geofeat[c] = 0.0
    Xg = geofeat[feature_cols].values

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xg)[:, 1]
    else:
        proba = model.predict(Xg).astype(float)

    geofeat["anomaly_score"] = proba
    k = max(1, int(len(geofeat) * ratio))
    # robust threshold: include ties at the cutoff
    sorted_scores = np.sort(proba)[::-1]
    thr = sorted_scores[k-1] if len(sorted_scores) >= k else 1.0
    geofeat["is_anomaly"] = (geofeat["anomaly_score"] >= thr).astype(int)
    geofeat = geofeat.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    if out_csv:
        geofeat[["traj_id","anomaly_score","is_anomaly"]].to_csv(out_csv, index=False, encoding="utf-8")
        print(f"Wrote labels to {out_csv} (top {ratio:.0%} flagged; ties included).")

    label_map = geofeat.set_index("traj_id")["anomaly_score"].to_dict()
    label_bin = geofeat.set_index("traj_id")["is_anomaly"].to_dict()
    return label_map, label_bin, geofeat

# ------------------------
# Build traj_dict (with neighbor POIs)
# ------------------------

def build_traj_dict(Dataset: pd.DataFrame, score_map: Dict, bin_map: Dict) -> Dict:
    D = Dataset.copy()
    if 'poi_type_list' in D.columns:
        D['poi_type_list'] = D['poi_type_list'].apply(lambda x: ['空区域'] if not isinstance(x, list) or len(x) == 0 else sorted(x))

    print("Prepare neighbor POI...")
    if 'h3_index' in D.columns:
        Poi = D.drop_duplicates(subset=["h3_index"]).loc[:, ["h3_index","poi_type_list"]]
        h3_poi_map = Poi.set_index('h3_index')[["poi_type_list"]].to_dict('index')
    else:
        h3_poi_map = {}

    def get_neighbor_poi(h3_idx, k=1):
        if pd.isna(h3_idx):
            return {'neighbor_poi_type_list': [['空区域']]}
        neighbors = list(set(h3.grid_disk(h3_idx, k)) - {h3_idx})
        neighbor_types = []
        for n in neighbors:
            if n in h3_poi_map:
                neighbor_types.append(sorted(h3_poi_map[n]['poi_type_list']))
            else:
                neighbor_types.append(['空区域'])
        return {'neighbor_poi_type_list': neighbor_types}

    neighbor_poi = D['h3_index'].apply(get_neighbor_poi)
    neighbor_df = pd.DataFrame(list(neighbor_poi))
    D = pd.concat([D.reset_index(drop=True), neighbor_df.reset_index(drop=True)], axis=1)

    traj_dict = {}
    for traj_id, g in D.groupby('traj_id'):
        score = float(score_map.get(traj_id, 0.0))
        label = int(bin_map.get(traj_id, 0))
        traj_dict[traj_id] = {
            'poi_type': g.get('poi_type_list', pd.Series([['空区域']]*len(g))).tolist(),
            'hour': g['hour'].astype(int).tolist() if 'hour' in g else [],
            'neighbor_poi_type': g['neighbor_poi_type_list'].tolist(),
            'stay_duration': g['stay_duration'].tolist() if 'stay_duration' in g else [],
            'h3_index': g['h3_index'].tolist() if 'h3_index' in g else [],
            'is_anomaly': bool(label),
            'anomaly_score': float(score),
        }
    return traj_dict

# ------------------------
# Incident synthesis (deterministic by default)
# ------------------------
DEFAULT_REGION_KEYS = ["h3_index", "h3", "region", "cell", "hex", "grid_id"]

def detect_region_key(traj_dict, region_key=None):
    if region_key:
        return region_key, True
    sample_key = next(iter(traj_dict))
    sample = traj_dict[sample_key]
    for k in DEFAULT_REGION_KEYS:
        if k in sample:
            if isinstance(sample[k], (list, tuple)) and 'hour' in sample and len(sample[k]) == len(sample['hour']):
                return k, True
    return None, False


def diurnal_curve():
    H = 24
    t = np.arange(H)
    return 0.6 + 0.9 * np.sin(2 * np.pi * (t - 18) / 24.0) ** 2


def per_region_scaler(regions):
    def norm_hash(x):
        return (abs(hash(x)) % 10_000_019) / 10_000_019.0
    scalers = {}
    for r in regions:
        u = norm_hash(str(r))
        scalers[r] = 0.3 + 1.7 * u
    return scalers


def build_rate_table(regions, base_scale=0.8):
    diurnal = diurnal_curve()
    region_scale = per_region_scaler(regions)
    rate = {}
    for r in regions:
        hourly = np.clip(base_scale * region_scale[r] * diurnal, 0.05, None)
        rate[r] = hourly
    return rate


def attach_incidents_from_anomaly(traj_dict, region_key, alpha=4.0, beta=0.15, base_scale=0.8, poisson=False):
    regions = set()
    for v in traj_dict.values():
        for rid in v.get(region_key, []):
            regions.add(rid)
    rate_table = build_rate_table(sorted(regions), base_scale=base_scale)

    rng = np.random.default_rng(42)
    for tid, v in traj_dict.items():
        hours = v.get("hour", [])
        rids  = v.get(region_key, [])
        if len(hours) != len(rids):
            raise ValueError(f"[{tid}] hour length {len(hours)} != {region_key} length {len(rids)}")
        score = float(v.get("anomaly_score", 0.0))
        is_ab = 1 if bool(v.get("is_anomaly", 0)) else 0
        inc_seq = []
        for h, rid in zip(hours, rids):
            h_int = int(h) % 24
            base = float(rate_table[rid][h_int])
            lam = base * (1.0 + alpha * score) if is_ab else base * max(beta * score, 1e-3)
            val = float(rng.poisson(lam)) if poisson else float(lam)
            inc_seq.append(val)
        v["incident"] = inc_seq

    return {"num_regions": len(regions), "example_regions": list(regions)[:10]}

# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poi", required=True)
    ap.add_argument("--geolife", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-pkl", required=True, dest="out_pkl")
    ap.add_argument("--out-labels", default=None, dest="out_labels")
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--resolution", type=int, default=10)
    ap.add_argument("--nrows", type=int, default=None)
    # hygiene params
    ap.add_argument("--max-kmh", type=float, default=100.0, help="Max allowed segment speed (km/h) before dropping point")
    ap.add_argument("--min-points", type=int, default=5, help="Min points per trajectory after QC")
    ap.add_argument("--max-idle-min", type=int, default=120, help="Drop points after idle gap larger than this (min)")
    # incident params
    ap.add_argument("--region-key", type=str, default=None)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--beta", type=float, default=0.15)
    ap.add_argument("--base-scale", type=float, default=0.8)
    ap.add_argument("--poisson", action="store_true")

    args = ap.parse_args()

    # 1) Load & process
    Poi = process_data_poi(args.poi, args.nrows, args.resolution)
    Traj = process_data_traj(
        args.geolife, args.nrows, args.resolution,
        max_kmh=args.max_kmh, min_points=args.min_points, max_idle_min=args.max_idle_min
    )

    # 2) Merge POI onto trajectory cells
    Traj['h3_index_traj'] = Traj['h3_index']
    Dataset = pd.merge(Traj, Poi, on='h3_index', how='left').reset_index(drop=True)

    # 3) Label trajectories using external model
    score_map, bin_map, geofeat = label_geolife_with_model(Traj, args.model, args.ratio, args.out_labels)

    # 4) Build traj_dict
    traj_dict = build_traj_dict(Dataset, score_map, bin_map)

    # 5) Attach incidents (deterministic by default)
    region_key, ok = detect_region_key(traj_dict, args.region_key)
    if not ok:
        raise KeyError("trajectory.pkl 缺少与 hour 等长的区域字段(如 h3_index)。可以通过 --region-key 指定。")
    _stats = attach_incidents_from_anomaly(
        traj_dict, region_key=region_key,
        alpha=args.alpha, beta=args.beta, base_scale=args.base_scale, poisson=args.poisson
    )

    # 6) Save
    os.makedirs(os.path.dirname(args.out_pkl) or ".", exist_ok=True)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(traj_dict, f)
    print(f"Saved trajectory with incidents to {args.out_pkl}")
    if args.out_labels:
        print(f"Labels saved to {args.out_labels}")
    print("Regions:", _stats["num_regions"], "Example:", _stats["example_regions"]) 

if __name__ == "__main__":
    main()
