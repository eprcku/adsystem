
import os
import argparse
import numpy as np
import pandas as pd


CROPS = ["RICE", "TEA", "VEG"]

CROP_BASES = [
    "SYN_PURCHASE", "SYN_TRANS", "COMPOST_IMPORT_PURCHASE",
    "COMPOST_LOC_TRANS", "COMPOST_IMPORT_TRANS", "DIGEST_TRANS",
    "AGROCHEM_PURCHASE", "AGROCHEM_TRANS", "DIGEST_MACHINERY",
    "AG_DIESEL", "AG_GASOL", "AG_KEROS", "AG_ELEC",
]

TOTAL_BASES = [
    "MANURE_TRANS", "RESIDUE_TRANS", "COMPOST_PROCESS", "COMPOST_FOOD_PROCESS", "MANURE_MWAD_PROCESS",
    "WASTE_COLLECT", "INC_PROCESS", "ASH_TRANS", "TDAD_PROCESS", "TDAD_REF_TRANS",
    "TDAD_DIGSOLID_TRANS", "TDAD_DIGWW", "RDF_PROCESS", "RDF_TRANS", "RDF_REF_TRANS",
    "MWAD_PROCESS", "MWAD_DIGWW", "MWAD_REF_TRANS",
    "INC_CAPEX", "TDAD_CAPEX", "RDF_CAPEX", "MWAD_CAPEX",
]

REGION_MAP = {
    "NORTH": {201, 202, 203, 205, 212, 463, 465},
    "CENTRAL": {101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 206, 213, 407},
    "SOUTH": {204, 207, 208, 209, 210, 211, 214, 303, 322, 343, 344, 364, 365, 366, 367},
}

FACILITY_CITY_PREFIX = {
    "KYOTO_NORTH": 108,
    "KYOTO_NE": 103,
    "KYOTO_SOUTH": 109,
    "FUKUCHIYAMA": 201,
    "MAIZURU": 202,
    "SAKURAZUKA": 206,
    "KYOTANABE": 211,
    "MINEYAMA": 212,
    "HASEYAMA": 207,
    "ORII": 204,
    "KIZUGAWA": 214,
    "OTOKUNI": 303,
    "MIYAZUYOSA": 205,
    "AYABE": 203,
    "FUSHIMI": 109,
    "JOYO": 207,
    "KAMEOKA": 206,
    "KYOTANGO": 212,
    "NANTAN": 213,
    "KYOTANBA": 407,
}



# 単位整合性チェック

UNIT_FACTORS = {
    ("kg", "kg"): 1.0, ("kilogram", "kg"): 1.0, ("kilograms", "kg"): 1.0,
    ("t", "kg"): 1000.0, ("ton", "kg"): 1000.0, ("tons", "kg"): 1000.0, ("tonne", "kg"): 1000.0, ("tonnes", "kg"): 1000.0,
    ("g", "kg"): 0.001,
    ("km", "km"): 1.0, ("m", "km"): 0.001,
    ("kwh", "kwh"): 1.0, ("wh", "kwh"): 0.001, ("mwh", "kwh"): 1000.0,
    ("l", "l"): 1.0, ("liter", "l"): 1.0, ("litre", "l"): 1.0,
    ("m3", "m3"): 1.0, ("m^3", "m3"): 1.0,
    ("yen", "yen"): 1.0, ("jpy", "yen"): 1.0,
    ("fraction", "fraction"): 1.0, ("%", "fraction"): 0.01, ("percent", "fraction"): 0.01,
}

def _norm_unit(unit) -> str:
    return str(unit).strip().lower().replace(" ", "").replace("㎥", "m3").replace("ｍ3", "m3")

def convert_series_units(values, from_units, to_unit: str, column_name: str) -> pd.Series:
    vals = to_num(values, 0.0)
    if not isinstance(vals, pd.Series):
        vals = pd.Series(vals)
    if from_units is None:
        return vals
    units = from_units if isinstance(from_units, pd.Series) else pd.Series(from_units, index=vals.index)
    out = vals.copy().astype(float)
    to_unit_n = _norm_unit(to_unit)
    norm_units = units.map(_norm_unit)
    for unit_n in norm_units.dropna().unique():
        factor = UNIT_FACTORS.get((unit_n, to_unit_n))
        if factor is None:
            raise ValueError(f"Unsupported unit conversion for {column_name}: {unit_n!r} -> {to_unit!r}")
        out.loc[norm_units == unit_n] = vals.loc[norm_units == unit_n].astype(float) * factor
    return out

UNIT_SCHEMA = {
    "02_inventory_crop.csv": {
        "SYN_KG": "kg", "COMPOST_LOC_KG": "kg", "COMPOST_IMP_KG": "kg", "DIGEST_KG": "kg",
        "AGROCHEM_KG": "kg", "AG_DIESEL_L": "l", "AG_GASOL_L": "l", "AG_KEROS_L": "l",
        "AG_ELEC_KWH": "kwh", "SYN_KM": "km", "COMPOST_LOC_KM": "km", "COMPOST_IMP_KM": "km",
        "DIGEST_KM": "km", "AGROCHEM_KM": "km",
        "DIGEST_MACHINERY_UNITS": "fraction",
        "DIGEST_MACHINERY_NEW_UNITS": "fraction",
    },
    "02_inventory_manure.csv": {
        "MANURE_KG": "kg", "RESIDUE_SUPPLY_KG": "kg", "RESIDUE_IMPORT_KG": "kg",
        "MANURE_KM": "km", "RESIDUE_SUPPLIED_KM": "km", "RESIDUE_IMPORT_KM": "km",
        "CAPACITY": "kg",
    },
    "02_inventory_waste.csv": {
        "CAPACITY": "kg", "WASTE_KG": "kg", "FOOD_KG": "kg", "INC_INPUT_KG": "kg", "ASH_KG": "kg",
        "TDAD_REFUSE_KG": "kg", "TDAD_DIGSOLID_KG": "kg", "TDAD_DIGEST_KG": "kg", "RDF_KG": "kg", "RDF_REFUSE_KG": "kg",
        "MWAD_REFUSE_KG": "kg", "MWAD_DIGEST_KG": "kg", "DIGEST_UNUSED_KG": "kg",
        "WASTE_KM": "km", "ASH_KM": "km", "TDAD_REFUSE_KM": "km", "RDF_KM": "km",
        "RDF_REFUSE_KM": "km", "MWAD_REFUSE_KM": "km",
        "INC_CAPEX": "yen", "TDAD_CAPEX": "yen", "RDF_CAPEX": "yen", "MWAD_CAPEX": "yen",
    },
    "02_fare_transport.csv": {"DIST": "km", "SMALL": "yen", "LARGE": "yen", "TANKER": "yen"},
}

def standardize_units(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = df.copy()
    for col, target_unit in UNIT_SCHEMA.get(dataset_name, {}).items():
        if col not in out.columns:
            continue
        unit_col = f"{col}_UNIT"
        out[col] = convert_series_units(out[col], out[unit_col] if unit_col in out.columns else None, target_unit, col)
    return out

def require_nonnegative(df: pd.DataFrame, dataset_name: str) -> None:
    bad = []
    for col in UNIT_SCHEMA.get(dataset_name, {}):
        if col in df.columns:
            s = to_num(df[col], 0.0)
            if isinstance(s, pd.Series) and (s < -1e-9).any():
                bad.append(col)
    if bad:
        raise ValueError(f"{dataset_name} has negative values in nonnegative unit columns: {bad}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default=os.getenv("SCENARIO", "CONV"))
    p.add_argument("--input-dir", default=os.getenv("INPUT_DIR", "."))
    p.add_argument("--dataset-dir", default=os.getenv("DATASET_DIR", "."))
    p.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "."))
    return p.parse_args()

ARGS = parse_args()
SCENARIO = str(ARGS.scenario).upper().strip()
INPUT_DIR = ARGS.input_dir
DATASET_DIR = ARGS.dataset_dir
OUTPUT_DIR = ARGS.output_dir

def to_num(x, default=0.0):
    out = pd.to_numeric(x, errors="coerce")
    if isinstance(out, pd.Series):
        return out.fillna(default)
    return default if pd.isna(out) else float(out)

def ensure_cols(df: pd.DataFrame, cols, default=0.0) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = default
    return out

def prep_numeric(df: pd.DataFrame, exclude=None) -> pd.DataFrame:
    exclude = set([] if exclude is None else exclude)
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        out[c] = to_num(out[c], 0.0)
    return out


def add_incremental_digest_machinery_units(
    df: pd.DataFrame,
    *,
    unit_capacity_t_per_year: float = 2500.0,
    grouping_cols: tuple[str, ...] = ("CITY", "CROP"),
) -> pd.DataFrame:

    out = df.copy()
    if out.empty:
        out["DIGEST_MACHINERY_UNITS"] = pd.Series(dtype=int)
        out["DIGEST_MACHINERY_NEW_UNITS"] = pd.Series(dtype=int)
        return out
    for col in ["CITY", "YEAR", "CROP", "DIGEST_KG"]:
        if col not in out.columns:
            out[col] = 0.0 if col != "CROP" else ""
    out["CITY"] = to_num(out["CITY"], 0).astype(int)
    out["YEAR"] = to_num(out["YEAR"], 0).astype(int)
    out["CROP"] = out["CROP"].astype(str).str.upper().str.strip()
    out["DIGEST_KG"] = to_num(out["DIGEST_KG"], 0.0).clip(lower=0.0)

    out["DIGEST_MACHINERY_UNITS"] = 0
    out["DIGEST_MACHINERY_NEW_UNITS"] = 0

    valid_group_cols = [c for c in grouping_cols if c in out.columns] or ["CROP"]
    work = out.reset_index().rename(columns={"index": "_ROW_INDEX"})
    work["_REQUIRED_RAW_UNITS"] = np.where(
        work["CROP"].eq("RICE"),
        work["DIGEST_KG"] * 0.001 / float(unit_capacity_t_per_year),
        0.0,
    )
    totals = (
        work.groupby(valid_group_cols + ["YEAR"], dropna=False, as_index=False)
        .agg(_REQUIRED_RAW_UNITS=("_REQUIRED_RAW_UNITS", "sum"), _GROUP_DIGEST_KG=("DIGEST_KG", "sum"))
        .sort_values(valid_group_cols + ["YEAR"])
    )
    totals["DIGEST_MACHINERY_GROUP_UNITS"] = np.ceil(
        np.maximum(to_num(totals["_REQUIRED_RAW_UNITS"], 0.0), 0.0) - 1e-12
    ).astype(int)
    totals.loc[totals["_GROUP_DIGEST_KG"] <= 0, "DIGEST_MACHINERY_GROUP_UNITS"] = 0
    totals["DIGEST_MACHINERY_GROUP_NEW_UNITS"] = 0
    for _, g in totals.groupby(valid_group_cols, dropna=False, sort=False):
        max_stock = 0
        for idx, row in g.sort_values("YEAR").iterrows():
            required = int(max(row["DIGEST_MACHINERY_GROUP_UNITS"], 0))
            add = max(required - max_stock, 0)
            totals.at[idx, "DIGEST_MACHINERY_GROUP_NEW_UNITS"] = int(add)
            max_stock = max(max_stock, required)

    work = work.merge(
        totals[valid_group_cols + ["YEAR", "DIGEST_MACHINERY_GROUP_UNITS", "DIGEST_MACHINERY_GROUP_NEW_UNITS"]],
        on=valid_group_cols + ["YEAR"], how="left"
    )
    work["DIGEST_MACHINERY_GROUP_UNITS"] = to_num(work["DIGEST_MACHINERY_GROUP_UNITS"], 0).astype(int)
    work["DIGEST_MACHINERY_GROUP_NEW_UNITS"] = to_num(work["DIGEST_MACHINERY_GROUP_NEW_UNITS"], 0).astype(int)
    reps = (
        work.sort_values(valid_group_cols + ["YEAR", "DIGEST_KG", "_ROW_INDEX"], ascending=[True] * (len(valid_group_cols) + 1) + [False, True])
        .groupby(valid_group_cols + ["YEAR"], dropna=False, as_index=False)
        .head(1)
    )
    is_rep = work["_ROW_INDEX"].isin(set(reps["_ROW_INDEX"].tolist()))
    work["DIGEST_MACHINERY_UNITS"] = np.where(is_rep, work["DIGEST_MACHINERY_GROUP_UNITS"], 0).astype(int)
    work["DIGEST_MACHINERY_NEW_UNITS"] = np.where(is_rep, work["DIGEST_MACHINERY_GROUP_NEW_UNITS"], 0).astype(int)

    out.loc[work["_ROW_INDEX"].to_numpy(), "DIGEST_MACHINERY_UNITS"] = work["DIGEST_MACHINERY_UNITS"].to_numpy(dtype=int)
    out.loc[work["_ROW_INDEX"].to_numpy(), "DIGEST_MACHINERY_NEW_UNITS"] = work["DIGEST_MACHINERY_NEW_UNITS"].to_numpy(dtype=int)
    out["DIGEST_MACHINERY_UNITS"] = out["DIGEST_MACHINERY_UNITS"].astype(int)
    out["DIGEST_MACHINERY_NEW_UNITS"] = out["DIGEST_MACHINERY_NEW_UNITS"].astype(int)
    return out


def regress_year_table(df: pd.DataFrame, value_cols) -> pd.DataFrame:
    years = np.arange(2030, 2051, dtype=int)
    out = pd.DataFrame({"YEAR": years})
    for c in value_cols:
        if c == "YEAR":
            continue
        s = df[["YEAR", c]].dropna()
        if len(s) == 0:
            out[c] = 0.0
            continue
        x = to_num(s["YEAR"]).to_numpy(dtype=float)
        y = to_num(s[c]).to_numpy(dtype=float)
        if len(x) < 2:
            out[c] = float(y[-1])
        else:
            a, b = np.polyfit(x, y, 1)
            out[c] = a * years + b
    return out

def regress_coeff(df: pd.DataFrame, col: str):
    s = df[["YEAR", col]].dropna()
    if len(s) == 0:
        return 0.0, 0.0
    x = to_num(s["YEAR"]).to_numpy(dtype=float)
    y = to_num(s[col]).to_numpy(dtype=float)
    if len(x) < 2:
        return 0.0, float(y[-1])
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)

def base_ratio(price_hist: pd.DataFrame, price_y: pd.DataFrame, col: str, base_year: int) -> pd.Series:
    if col not in price_hist.columns or col not in price_y.columns:
        return pd.Series(np.ones(len(price_y)), index=price_y.index, dtype=float)
    a, b = regress_coeff(price_hist, col)
    base_val = a * float(base_year) + b
    if base_val == 0 or np.isnan(base_val):
        return pd.Series(np.ones(len(price_y)), index=price_y.index, dtype=float)
    return to_num(price_y[col], 0.0) / float(base_val)

def fare_interp(fare_df: pd.DataFrame):
    f = standardize_units(fare_df, "02_fare_transport.csv")
    f = ensure_cols(f, ["DIST", "SMALL", "LARGE", "TANKER"], 0.0)
    f["DIST"] = to_num(f["DIST"], 0.0)
    f = f.sort_values("DIST")
    dist = f["DIST"].to_numpy(dtype=float)
    fares = {t: to_num(f[t], 0.0).to_numpy(dtype=float) for t in ["SMALL", "LARGE", "TANKER"]}

    def _f(truck: str, km):
        truck = str(truck).upper()
        km_arr = np.asarray(km, dtype=float)
        if len(dist) == 0 or truck not in fares:
            return np.zeros_like(km_arr)
        return np.interp(km_arr, dist, fares[truck], left=fares[truck][0], right=fares[truck][-1])

    return _f

def sewage_fare_interp(df_any: pd.DataFrame):
    df = df_any.copy()
    if "VOLUME" in df.columns and "FARE_SEWAGE" in df.columns:
        vcol, fcol = "VOLUME", "FARE_SEWAGE"
    else:
        cols = list(df.columns)
        if len(cols) < 2:
            def _f(v):
                return np.zeros_like(np.asarray(v, dtype=float))
            return _f
        vcol, fcol = cols[0], cols[1]
    df[vcol] = to_num(df[vcol], 0.0)
    df[fcol] = to_num(df[fcol], 0.0)
    df = df.sort_values(vcol)
    vol = df[vcol].to_numpy(dtype=float)
    fee = df[fcol].to_numpy(dtype=float)

    def _f(v):
        v_arr = np.asarray(v, dtype=float)
        if len(vol) == 0:
            return np.zeros_like(v_arr)
        return np.interp(v_arr, vol, fee, left=fee[0], right=fee[-1])

    return _f

def city_to_region(city: int) -> str:
    try:
        city = int(city)
    except Exception:
        return "UNKNOWN"
    for region, vals in REGION_MAP.items():
        if city in vals:
            return region
    return "UNKNOWN"

def facility_to_city(facility: str) -> int:
    s = str(facility).upper()
    for prefix, city in FACILITY_CITY_PREFIX.items():
        if s.startswith(prefix):
            return city
    return -1

def build_price_table(cost_path: str) -> pd.DataFrame:
    price = pd.read_csv(cost_path)
    price_y = regress_year_table(price, [c for c in price.columns if c != "YEAR"])
    price_y["CHEM_INFL_2024"] = base_ratio(price, price_y, "CHEM", 2024)
    price_y["PUBLIC_INFL_2024"] = base_ratio(price, price_y, "PUBLIC", 2024)
    price_y["PUBLIC_INFL_2020"] = base_ratio(price, price_y, "PUBLIC", 2020)
    price_y["PUBLIC_INFL_2015"] = base_ratio(price, price_y, "PUBLIC", 2015)
    price_y["MACHINERY_INFL_2024"] = base_ratio(price, price_y, "MACHINERY", 2024)
    return price_y

def calc_crop_cost_long(crop: pd.DataFrame, price_y: pd.DataFrame, fare_fn) -> pd.DataFrame:
    crop = standardize_units(crop, "02_inventory_crop.csv")
    require_nonnegative(crop, "02_inventory_crop.csv")
    crop = ensure_cols(
        prep_numeric(crop, exclude=["KEY", "CROP", "CULTIVAR", "ORG_FLAG"]),
        [
            "CITY", "YEAR", "SYN_KG", "COMPOST_LOC_KG", "COMPOST_IMP_KG", "DIGEST_KG", "DIGEST_MACHINERY_UNITS", "DIGEST_MACHINERY_NEW_UNITS", "AGROCHEM_KG",
            "AG_DIESEL_L", "AG_GASOL_L", "AG_KEROS_L", "AG_ELEC_KWH",
        ],
        0.0,
    )
    crop["CROP"] = crop.get("CROP", "UNKNOWN").astype(str).str.upper().str.strip()
    crop = add_incremental_digest_machinery_units(crop)
    crop["CITY"] = to_num(crop["CITY"], 0).astype(int)
    crop["YEAR"] = to_num(crop["YEAR"], 0).astype(int)
    crop["REGION"] = crop["CITY"].map(city_to_region)
    crop = crop.merge(price_y, on="YEAR", how="left")

    agrochem_2024 = 0.0
    dav2500_2024 = 0.0
    hist = pd.read_csv(os.path.join(INPUT_DIR, "00_cost.csv"))
    if "AGROCHEM" in hist.columns and (hist["YEAR"] == 2024).any():
        agrochem_2024 = float(to_num(hist.loc[hist["YEAR"] == 2024, "AGROCHEM"], 0.0).iloc[0])
    if "DAV2500" in hist.columns and (hist["YEAR"] == 2024).any():
        dav2500_2024 = float(to_num(hist.loc[hist["YEAR"] == 2024, "DAV2500"], 0.0).iloc[0])

    out = crop[["REGION", "CITY", "YEAR", "CROP"]].copy()
    out["SYN_PURCHASE"] = crop["SYN_KG"] * crop.get("SYN", 0.0)
    out["SYN_TRANS"] = crop["SYN_KG"] * 0.001 / (10 * 0.75) * fare_fn("LARGE", crop.get("SYN_KM", 0.0)) * crop.get("PUBLIC_INFL_2024", 1.0)
    out["COMPOST_IMPORT_PURCHASE"] = crop["COMPOST_IMP_KG"] * crop.get("ORG", 0.0)
    out["COMPOST_LOC_TRANS"] = crop["COMPOST_LOC_KG"] * 0.001 / (10 * 0.75) * fare_fn("LARGE", crop.get("COMPOST_LOC_KM", 0.0)) * crop.get("PUBLIC_INFL_2024", 1.0)
    out["COMPOST_IMPORT_TRANS"] = crop["COMPOST_IMP_KG"] * 0.001 / (10 * 0.75) * fare_fn("LARGE", crop.get("COMPOST_IMP_KM", 0.0)) * crop.get("PUBLIC_INFL_2024", 1.0)
    out["DIGEST_TRANS"] = crop["DIGEST_KG"] * 0.001 / (20 * 0.75) * fare_fn("TANKER", crop.get("DIGEST_KM", 0.0)) * crop.get("PUBLIC_INFL_2024", 1.0)
    out["DIGEST_MACHINERY"] = crop.get("DIGEST_MACHINERY_NEW_UNITS", 0.0) * dav2500_2024 * crop.get("MACHINERY_INFL_2024", 1.0)
    out["AGROCHEM_PURCHASE"] = crop["AGROCHEM_KG"] * agrochem_2024 * crop.get("CHEM_INFL_2024", 1.0)
    out["AG_DIESEL"] = crop["AG_DIESEL_L"] * crop.get("DIESEL", 0.0)
    out["AG_GASOL"] = crop["AG_GASOL_L"] * crop.get("GASOL", 0.0)
    out["AG_KEROS"] = crop["AG_KEROS_L"] * crop.get("KEROS", 0.0)
    out["AG_ELEC"] = crop["AG_ELEC_KWH"] * crop.get("ELEC_DEFAULT", 0.0)
    return out.groupby(["REGION", "CITY", "YEAR", "CROP"], as_index=False).sum(numeric_only=True)

def calc_crop_cost_wide(crop_long: pd.DataFrame) -> pd.DataFrame:
    keys = ["REGION", "CITY", "YEAR"]
    base = crop_long[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    if base.empty:
        return base

    # Build a full CITY-YEAR-CROP grid so sparse crops like TEA and VEG
    # are retained with explicit zero values rather than disappearing.
    grid = (
        base.assign(_tmp_key=1)
        .merge(pd.DataFrame({"CROP": CROPS, "_tmp_key": 1}), on="_tmp_key", how="inner")
        .drop(columns=["_tmp_key"])
    )

    merged = grid.merge(
        crop_long,
        on=["REGION", "CITY", "YEAR", "CROP"],
        how="left",
    )

    for col in CROP_BASES:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = to_num(merged[col], 0.0)

    out = base.copy()
    for crop_name in CROPS:
        sub = merged[merged["CROP"] == crop_name][keys + CROP_BASES].copy()
        sub = sub.rename(columns={col: f"{col}_{crop_name}" for col in CROP_BASES})
        out = out.merge(sub, on=keys, how="left")

    for crop_name in CROPS:
        for col in CROP_BASES:
            wide_col = f"{col}_{crop_name}"
            if wide_col not in out.columns:
                out[wide_col] = 0.0
            out[wide_col] = to_num(out[wide_col], 0.0)

    return out.fillna(0.0)

def load_conv_mwad_capacity_map():
    path = os.path.join(INPUT_DIR, "00_manurefac.csv")
    if not os.path.exists(path):
        return {}
    mf = pd.read_csv(path)
    if "TYPE" not in mf.columns or "FACILITY" not in mf.columns or "CAPACITY" not in mf.columns:
        return {}
    mf["TYPE"] = mf["TYPE"].astype(str).str.upper().str.strip()
    mf["FACILITY"] = mf["FACILITY"].astype(str)
    mf["CAPACITY"] = to_num(mf["CAPACITY"], 0.0)
    mf = mf[mf["TYPE"] == "MWAD"].copy()
    if mf.empty:
        return {}
    return mf.groupby("FACILITY", as_index=False)["CAPACITY"].sum().set_index("FACILITY")["CAPACITY"].to_dict()

def calc_manure_cost(manure: pd.DataFrame, price_y: pd.DataFrame, fare_fn) -> pd.DataFrame:
    manure = standardize_units(manure, "02_inventory_manure.csv")
    require_nonnegative(manure, "02_inventory_manure.csv")
    manure = ensure_cols(
        prep_numeric(manure, exclude=["FACILITY", "TYPE"]),
        [
            "FACILITY", "YEAR", "MANURE_KG", "RESIDUE_SUPPLY_KG", "RESIDUE_IMPORT_KG",
            "MANURE_KM", "RESIDUE_SUPPLIED_KM", "RESIDUE_IMPORT_KM",
        ],
        0.0,
    )
    manure["FACILITY"] = manure["FACILITY"].astype(str)
    manure["TYPE"] = manure.get("TYPE", "").astype(str).str.upper().str.strip()
    # Defensive cleanup for old inventories: MWAD must not carry residue transport.
    mwad_mask = manure["TYPE"].eq("MWAD")
    for col in ["RESIDUE_SUPPLY_KG", "RESIDUE_IMPORT_KG", "RESIDUE_SUPPLIED_KM", "RESIDUE_IMPORT_KM"]:
        if col in manure.columns:
            manure.loc[mwad_mask, col] = 0.0
    manure["YEAR"] = to_num(manure["YEAR"], 0).astype(int)
    manure["CITY"] = manure["FACILITY"].map(facility_to_city).astype(int)
    manure["REGION"] = manure["CITY"].map(city_to_region)
    manure = manure.merge(price_y[["YEAR", "PUBLIC_INFL_2024", "PUBLIC_INFL_2015"]], on="YEAR", how="left")

    out = manure[["REGION", "CITY", "YEAR"]].copy()
    out["MANURE_TRANS"] = manure["MANURE_KG"] * 0.001 / (20 * 0.75) * fare_fn("TANKER", manure.get("MANURE_KM", 0.0)) * manure.get("PUBLIC_INFL_2024", 1.0)
    out["RESIDUE_TRANS"] = (
        manure["RESIDUE_SUPPLY_KG"] * 0.001 / (10 * 0.75) * fare_fn("LARGE", manure.get("RESIDUE_SUPPLIED_KM", 0.0))
        + manure["RESIDUE_IMPORT_KG"] * 0.001 / (10 * 0.75) * fare_fn("LARGE", manure.get("RESIDUE_IMPORT_KM", 0.0))
    ) * manure.get("PUBLIC_INFL_2024", 1.0)
    out["COMPOST_PROCESS"] = np.where(
        manure["TYPE"] == "COMPOST",
        43.748 * np.power(np.maximum(manure["MANURE_KG"] * 0.001 / 365.0, 0.0), 0.0138) * (10 ** 6) * manure.get("PUBLIC_INFL_2015", 1.0),
        0.0,
    )

    manure["MWAD_CAPACITY_REF"] = 0.0
    if SCENARIO == "CONV":
        cap_map = load_conv_mwad_capacity_map()
        manure["MWAD_CAPACITY_REF"] = np.where(manure["TYPE"] == "MWAD", manure["FACILITY"].map(cap_map).fillna(0.0), 0.0)
    else:
        if "CAPACITY" in manure.columns:
            manure["MWAD_CAPACITY_REF"] = np.where(manure["TYPE"] == "MWAD", to_num(manure["CAPACITY"], 0.0), 0.0)

    out["MANURE_MWAD_PROCESS"] = np.where(
        manure["TYPE"] == "MWAD",
        5.5676 * np.power(np.maximum(manure["MWAD_CAPACITY_REF"] * 0.001 / 365.0, 0.0), 0.4674) * (10 ** 6) * manure.get("PUBLIC_INFL_2015", 1.0),
        0.0,
    )
    return out.groupby(["REGION", "CITY", "YEAR"], as_index=False).sum(numeric_only=True)

def calc_waste_cost(waste: pd.DataFrame, crop: pd.DataFrame, price_y: pd.DataFrame, fare_fn, sewage_fn) -> pd.DataFrame:
    waste = standardize_units(waste, "02_inventory_waste.csv")
    crop = standardize_units(crop, "02_inventory_crop.csv")
    require_nonnegative(waste, "02_inventory_waste.csv")
    waste = ensure_cols(
        prep_numeric(waste, exclude=["FACILITY", "TYPE", "EVENT", "ENERGY_USE"]),
        [
            "FACILITY", "YEAR", "TYPE", "CAPACITY", "WASTE_KG", "FOOD_KG",
            "INC_INPUT_KG", "ASH_KG", "ASH_KM",
            "TDAD_REFUSE_KG", "TDAD_REFUSE_KM", "TDAD_DIGSOLID_KG", "TDAD_DIGEST_KG",
            "RDF_KG", "RDF_KM", "RDF_REFUSE_KG", "RDF_REFUSE_KM",
            "MWAD_REFUSE_KG", "MWAD_REFUSE_KM", "MWAD_DIGEST_KG",
            "INC_CAPEX", "TDAD_CAPEX", "RDF_CAPEX", "MWAD_CAPEX",
        ],
        0.0,
    )
    waste["YEAR"] = to_num(waste["YEAR"], 0).astype(int)
    waste["CITY"] = waste["FACILITY"].map(facility_to_city).astype(int)
    waste["REGION"] = waste["CITY"].map(city_to_region)
    waste["TYPE"] = waste["TYPE"].astype(str).str.upper()
    waste = waste.merge(price_y[["YEAR", "PUBLIC_INFL_2024", "PUBLIC_INFL_2020", "PUBLIC_INFL_2015"]], on="YEAR", how="left")

    crop_used = ensure_cols(prep_numeric(crop, exclude=["KEY", "CROP", "CULTIVAR", "ORG_FLAG"]), ["CITY", "YEAR", "DIGEST_KG"], 0.0)
    crop_used["CITY"] = to_num(crop_used["CITY"], 0).astype(int)
    crop_used["YEAR"] = to_num(crop_used["YEAR"], 0).astype(int)
    digest_used = crop_used.groupby(["CITY", "YEAR"], as_index=False)["DIGEST_KG"].sum().rename(columns={"DIGEST_KG": "DIGEST_USED_KG"})
    waste = waste.merge(digest_used, on=["CITY", "YEAR"], how="left")
    waste["DIGEST_USED_KG"] = to_num(waste.get("DIGEST_USED_KG", 0.0), 0.0)

    if "DIGEST_UNUSED_KG" not in waste.columns:
        waste["DIGEST_UNUSED_KG"] = np.maximum(waste.get("MWAD_DIGEST_KG", 0.0) - waste["DIGEST_USED_KG"], 0.0)
    else:
        waste["DIGEST_UNUSED_KG"] = to_num(waste["DIGEST_UNUSED_KG"], 0.0)

    mwad_fac_counts = (
        waste.assign(IS_MWAD=(waste["TYPE"] == "MWAD"))
        .groupby(["CITY", "YEAR"], as_index=False)["IS_MWAD"]
        .sum()
        .rename(columns={"IS_MWAD": "MWAD_FAC_COUNT"})
    )
    waste = waste.merge(mwad_fac_counts, on=["CITY", "YEAR"], how="left")
    waste["MWAD_FAC_COUNT"] = to_num(waste["MWAD_FAC_COUNT"], 0.0)

    out = waste[["REGION", "CITY", "YEAR"]].copy()
    is_inc = waste["TYPE"] == "INC"
    is_tdad = waste["TYPE"] == "TDAD"
    is_rdf = waste["TYPE"] == "RDF"
    is_mwad = waste["TYPE"] == "MWAD"
    is_compost = waste["TYPE"] == "COMPOST"

    out["WASTE_COLLECT"] = waste["WASTE_KG"] * 0.001 / 2.0 * fare_fn("SMALL", waste.get("WASTE_KM", 0.0)) * waste.get("PUBLIC_INFL_2024", 1.0)
    out["INC_PROCESS"] = np.where(is_inc, ((6971.033 * (waste["INC_INPUT_KG"] * 0.001 / 300.0)) + 655042.918) * 300.0 * waste.get("PUBLIC_INFL_2020", 1.0), 0.0)
    out["COMPOST_FOOD_PROCESS"] = np.where(
        is_compost,
        43.748 * np.power(np.maximum(waste["FOOD_KG"] * 0.001 / 365.0, 0.0), 0.0138) * (10 ** 6) * waste.get("PUBLIC_INFL_2015", 1.0),
        0.0,
    )
    out["ASH_TRANS"] = np.where(is_inc, waste["ASH_KG"] * 0.001 / (10 * 0.75) * fare_fn("LARGE", waste.get("ASH_KM", 0.0)) * waste.get("PUBLIC_INFL_2024", 1.0), 0.0)
    out["TDAD_PROCESS"] = np.where(is_tdad, 5.4353 * np.power(np.maximum(waste["CAPACITY"] * 0.001 / 365.0, 0.0), 0.7232) * (10 ** 6) * waste.get("PUBLIC_INFL_2015", 1.0), 0.0)
    out["TDAD_REF_TRANS"] = np.where(is_tdad, waste["TDAD_REFUSE_KG"] * 0.001 / (10 * 0.75) * fare_fn("LARGE", waste.get("TDAD_REFUSE_KM", 0.0)) * waste.get("PUBLIC_INFL_2024", 1.0), 0.0)
    out["TDAD_DIGSOLID_TRANS"] = np.where(is_tdad, waste["TDAD_DIGSOLID_KG"] * 0.001 / (10 * 0.75) * fare_fn("LARGE", waste.get("TDAD_REFUSE_KM", 0.0)) * waste.get("PUBLIC_INFL_2024", 1.0), 0.0)
    tdad_vol_2mo = waste["TDAD_DIGEST_KG"] * 0.001 / 6.0
    tdad_fee_yen_per_t = sewage_fn(tdad_vol_2mo)
    out["TDAD_DIGWW"] = np.where(
        is_tdad,
        tdad_vol_2mo * tdad_fee_yen_per_t * 6.0 * waste.get("PUBLIC_INFL_2024", 1.0),
        0.0,
    )
    out["RDF_PROCESS"] = np.where(is_rdf, 7.644 * np.power(np.maximum(waste["CAPACITY"] * 0.001 / 300.0, 0.0), 0.8533) * (10 ** 6) * waste.get("PUBLIC_INFL_2015", 1.0), 0.0)
    out["RDF_TRANS"] = np.where(is_rdf, waste["RDF_KG"] * 0.001 / (10 * 0.75) * fare_fn("LARGE", waste.get("RDF_KM", 0.0)) * waste.get("PUBLIC_INFL_2024", 1.0), 0.0)
    out["RDF_REF_TRANS"] = np.where(is_rdf, waste["RDF_REFUSE_KG"] * 0.001 / (10 * 0.75) * fare_fn("LARGE", waste.get("RDF_REFUSE_KM", 0.0)) * waste.get("PUBLIC_INFL_2024", 1.0), 0.0)
    out["MWAD_PROCESS"] = np.where(is_mwad, 5.5676 * np.power(np.maximum(waste["CAPACITY"] * 0.001 / 365.0, 0.0), 0.4674) * (10 ** 6) * waste.get("PUBLIC_INFL_2015", 1.0), 0.0)

    vol_2mo = waste["DIGEST_UNUSED_KG"] * 0.001 / 6.0
    fee_yen_per_t = sewage_fn(vol_2mo)
    annual_city_charge = vol_2mo * fee_yen_per_t * 6.0 * waste.get("PUBLIC_INFL_2024", 1.0)
    alloc_charge = np.where(waste["MWAD_FAC_COUNT"] > 0, annual_city_charge / waste["MWAD_FAC_COUNT"], 0.0)
    out["MWAD_DIGWW"] = np.where(is_mwad, alloc_charge, 0.0)
    out["MWAD_REF_TRANS"] = np.where(is_mwad, waste["MWAD_REFUSE_KG"] * 0.001 / (10 * 0.75) * fare_fn("LARGE", waste.get("MWAD_REFUSE_KM", 0.0)) * waste.get("PUBLIC_INFL_2024", 1.0), 0.0)
    out["INC_CAPEX"] = np.where(is_inc, waste.get("INC_CAPEX", 0.0), 0.0)
    out["TDAD_CAPEX"] = np.where(is_tdad, waste.get("TDAD_CAPEX", 0.0), 0.0)
    out["RDF_CAPEX"] = np.where(is_rdf, waste.get("RDF_CAPEX", 0.0), 0.0)
    out["MWAD_CAPEX"] = np.where(is_mwad, waste.get("MWAD_CAPEX", 0.0), 0.0)

    return out.groupby(["REGION", "CITY", "YEAR"], as_index=False).sum(numeric_only=True)

def ordered_cost_columns():
    cols = ["REGION", "CITY", "YEAR"]
    for crop_name in CROPS:
        for base in CROP_BASES:
            cols.append(f"{base}_{crop_name}")
    cols += TOTAL_BASES
    return cols

def main():
    cost_path = os.path.join(INPUT_DIR, "00_cost.csv")
    fare_path = os.path.join(INPUT_DIR, "02_fare_transport.csv")
    sewage_path = os.path.join(INPUT_DIR, "02_fare_sewage.csv")
    crop_path = os.path.join(INPUT_DIR, "02_inventory_crop.csv")
    manure_path = os.path.join(INPUT_DIR, "02_inventory_manure.csv")
    waste_path = os.path.join(INPUT_DIR, "02_inventory_waste.csv")

    price_y = build_price_table(cost_path)
    fare_fn = fare_interp(pd.read_csv(fare_path))
    sewage_fn = sewage_fare_interp(pd.read_csv(sewage_path))

    crop = pd.read_csv(crop_path)
    manure = pd.read_csv(manure_path)
    waste = pd.read_csv(waste_path)

    crop_cost_long = calc_crop_cost_long(crop, price_y, fare_fn)
    crop_cost_wide = calc_crop_cost_wide(crop_cost_long)
    manure_cost = calc_manure_cost(manure, price_y, fare_fn)
    waste_cost = calc_waste_cost(waste, crop, price_y, fare_fn, sewage_fn)

    out = crop_cost_wide.merge(manure_cost, on=["REGION", "CITY", "YEAR"], how="outer")
    out = out.merge(waste_cost, on=["REGION", "CITY", "YEAR"], how="outer").fillna(0.0)

    desired = ordered_cost_columns()
    for c in desired:
        if c not in out.columns:
            out[c] = 0.0
    out = out[desired].sort_values(["REGION", "CITY", "YEAR"]).reset_index(drop=True)
    if "YEAR" in out.columns:
        out = out[pd.to_numeric(out["YEAR"], errors="coerce").fillna(0).astype(int) >= 2030].copy()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_total_csv = os.path.join(OUTPUT_DIR, f"04_cost_{SCENARIO}.csv")
    out.to_csv(out_total_csv, index=False)
    print(f"Done. Wrote {out_total_csv}")

if __name__ == "__main__":
    main()
