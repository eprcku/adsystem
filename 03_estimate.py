
import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FACTORS = ["GWP", "MEP", "FEP", "TEP", "WATER"]
CROPS = ["RICE", "TEA", "VEG"]
P2O5_TO_P = 62.0 / 142.0
K2O_TO_K = 78.2 / 94.2


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

NUMERIC_EXCLUDE = {"CROP", "CULTIVAR", "FACILITY", "TYPE", "EVENT", "ENERGY_USE"}


# 単位整合性チェック

UNIT_FACTORS = {
    ("kg", "kg"): 1.0, ("kilogram", "kg"): 1.0, ("kilograms", "kg"): 1.0,
    ("t", "kg"): 1000.0, ("ton", "kg"): 1000.0, ("tons", "kg"): 1000.0, ("tonne", "kg"): 1000.0, ("tonnes", "kg"): 1000.0,
    ("g", "kg"): 0.001,
    ("km", "km"): 1.0, ("m", "km"): 0.001,
    ("kwh", "kwh"): 1.0, ("wh", "kwh"): 0.001, ("mwh", "kwh"): 1000.0,
    ("mj", "mj"): 1.0, ("gj", "mj"): 1000.0, ("kj", "mj"): 0.001,
    ("l", "l"): 1.0, ("liter", "l"): 1.0, ("litre", "l"): 1.0, ("m3", "m3"): 1.0, ("m^3", "m3"): 1.0,
    ("nm3", "nm3"): 1.0, ("nm3-ch4", "nm3"): 1.0, ("m3-ch4", "nm3"): 1.0,
    ("ha", "ha"): 1.0, ("a", "ha"): 0.01, ("are", "ha"): 0.01, ("m2", "ha"): 0.0001,
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
    for unit in units.dropna().unique():
        unit_n = _norm_unit(unit)
        factor = UNIT_FACTORS.get((unit_n, to_unit_n))
        if factor is None:
            raise ValueError(f"Unsupported unit conversion for {column_name}: {unit!r} -> {to_unit!r}")
        out.loc[units.map(_norm_unit) == unit_n] = vals.loc[units.map(_norm_unit) == unit_n].astype(float) * factor
    return out

UNIT_SCHEMA = {
    "02_inventory_crop.csv": {
        "EXTENT_HA": "ha",
        "SYN_KG": "kg", "SYN_N_KG": "kg", "SYN_P_KG": "kg", "SYN_K_KG": "kg",
        "SYN_P2O5_KG": "kg", "SYN_K2O_KG": "kg", "SYN_P_ELEM_KG": "kg", "SYN_K_ELEM_KG": "kg",
        "COMPOST_IMP_KG": "kg", "COMPOST_LOC_KG": "kg", "COMPOST_N_KG": "kg", "COMPOST_P_KG": "kg",
        "COMPOST_K_KG": "kg", "COMPOST_P2O5_KG": "kg", "COMPOST_K2O_KG": "kg", "COMPOST_P_ELEM_KG": "kg", "COMPOST_K_ELEM_KG": "kg",
        "DIGEST_KG": "kg", "DIGEST_N_KG": "kg", "DIGEST_P_KG": "kg", "DIGEST_C_KG": "kg",
        "DIGEST_K_KG": "kg", "DIGEST_P2O5_KG": "kg", "DIGEST_K2O_KG": "kg", "DIGEST_P_ELEM_KG": "kg", "DIGEST_K_ELEM_KG": "kg",
        "AGROCHEM_KG": "kg", "AG_DIESEL_MJ": "mj", "AG_GASOL_MJ": "mj", "AG_KEROS_MJ": "mj",
        "AG_ELEC_KWH": "kwh", "NH3_KG": "kg", "NO_KG": "kg", "NRUNOFF_KG": "kg", "NLEACH_KG": "kg",
        "RESIDUE_C_KG": "kg", "SYN_KM": "km", "COMPOST_IMP_KM": "km", "COMPOST_LOC_KM": "km",
        "DIGEST_KM": "km", "AGROCHEM_KM": "km",
        "DIGEST_MACHINERY_UNITS": "fraction",
        "DIGEST_MACHINERY_NEW_UNITS": "fraction",
    },
    "02_inventory_manure.csv": {
        "MANURE_KG": "kg", "DAIRY_MANURE_KG": "kg", "CATTLE_MANURE_KG": "kg", "SWINE_MANURE_KG": "kg",
        "CHICKEN_MANURE_KG": "kg", "BROILER_MANURE_KG": "kg", "COMPOST_N": "kg", "COMPOST_P2O5": "kg",
        "DIGEST_N": "kg", "DIGEST_P2O5": "kg", "RESIDUE_SUPPLY_KG": "kg", "RESIDUE_IMPORT_KG": "kg",
        "BIOGAS_CH4_NM3": "nm3", "MWAD_ELEC_USE_KWH": "kwh", "MWAD_WATER_USE_M3": "m3",
        "MANURE_KM": "km", "RESIDUE_SUPPLIED_KM": "km", "RESIDUE_IMPORT_KM": "km",
    },
    "02_inventory_waste.csv": {
        "CAPACITY": "kg", "WASTE_KG": "kg", "FOOD_KG": "kg", "MANURE_KG": "kg",
        "DAIRY_MANURE_KG": "kg", "CATTLE_MANURE_KG": "kg", "SWINE_MANURE_KG": "kg",
        "CHICKEN_MANURE_KG": "kg", "BROILER_MANURE_KG": "kg",
        "PLASTIC_KG": "kg", "SYNTEX_KG": "kg", "INC_INPUT_KG": "kg", "INC_KEROS_L": "l",
        "INC_ELEC_KWH": "kwh", "ASH_KG": "kg", "TDAD_ELEC_KWH": "kwh", "TDAD_BIOGAS_NM3": "nm3",
        "TDAD_DIGEST_KG": "kg", "TDAD_REFUSE_KG": "kg", "TDAD_DIGSOLID_KG": "kg",
        "RDF_KG": "kg", "RDF_REFUSE_KG": "kg", "MWAD_ELEC_KWH": "kwh", "MWAD_WATER_M3": "m3",
        "MWAD_BIOGAS_NM3": "nm3", "MWAD_DIGEST_KG": "kg", "DIGEST_UNUSED_KG": "kg",
        "MWAD_DIGSOLID_KG": "kg", "MWAD_REFUSE_KG": "kg",
        "WASTE_KM": "km", "MANURE_KM": "km", "ASH_KM": "km", "TDAD_REFUSE_KM": "km",
        "RDF_KM": "km", "RDF_REFUSE_KM": "km", "MWAD_REFUSE_KM": "km",
        "INC_CAPEX": "yen", "TDAD_CAPEX": "yen", "RDF_CAPEX": "yen", "MWAD_CAPEX": "yen",
        "ELEC_INC_KWH": "kwh", "ELEC_TDAD_KWH": "kwh", "ELEC_RDF_KWH": "kwh", "ELEC_MWAD_KWH": "kwh",
    },
    "02_energy.csv": {
        "COAL": "fraction", "PETRO": "fraction", "LNG": "fraction", "NUC": "fraction",
        "HYDRO": "fraction", "SOLAR": "fraction", "WIND": "fraction", "BIO": "fraction",
    },
}

def standardize_units(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = df.copy()
    schema = UNIT_SCHEMA.get(dataset_name, {})
    for col, target_unit in schema.items():
        if col not in out.columns:
            continue
        unit_col = f"{col}_UNIT"
        out[col] = convert_series_units(out[col], out[unit_col] if unit_col in out.columns else None, target_unit, col)
    return out

def require_nonnegative(df: pd.DataFrame, dataset_name: str, columns: Iterable[str] | None = None) -> None:
    cols = list(columns) if columns is not None else list(UNIT_SCHEMA.get(dataset_name, {}).keys())
    bad = []
    for col in cols:
        if col in df.columns:
            s = to_num(df[col], 0.0)
            if isinstance(s, pd.Series) and (s < -1e-9).any():
                bad.append(col)
    if bad:
        raise ValueError(f"{dataset_name} has negative values in nonnegative unit columns: {bad}")


def to_num(s, default=0.0):
    out = pd.to_numeric(s, errors="coerce")
    if isinstance(out, pd.Series):
        return out.fillna(default)
    return default if pd.isna(out) else float(out)


def ensure_columns(df: pd.DataFrame, cols: Iterable[str], default=0.0) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = default
    return out


def series_or_zero(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return to_num(df[col], 0.0)
    return pd.Series(0.0, index=df.index)



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


def ef_value(ef: pd.DataFrame, factor: str, col: str) -> float:
    sub = ef[ef["FACTOR"].astype(str).str.upper() == str(factor).upper()]
    if sub.empty or col not in sub.columns:
        return 0.0
    return float(to_num(sub.iloc[0][col], 0.0))


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


def build_elec_ef(energy: pd.DataFrame, ef: pd.DataFrame) -> pd.DataFrame:
    energy = standardize_units(energy.copy(), "02_energy.csv")
    energy["YEAR"] = to_num(energy["YEAR"]).astype(int)
    source_cols = ["COAL", "PETRO", "LNG", "NUC", "HYDRO", "SOLAR", "WIND", "BIO"]
    energy = ensure_columns(energy, source_cols, 0.0)
    for c in source_cols:
        energy[c] = to_num(energy[c], 0.0)

    out = energy[["YEAR"]].copy()
    for fac in FACTORS:
        out[f"ELEC_{fac}"] = (
            energy["COAL"] * ef_value(ef, fac, "ELEC_COAL")
            + energy["PETRO"] * ef_value(ef, fac, "ELEC_PETRO")
            + energy["LNG"] * ef_value(ef, fac, "ELEC_LNG")
            + energy["NUC"] * ef_value(ef, fac, "ELEC_NUC")
            + energy["HYDRO"] * ef_value(ef, fac, "ELEC_HYDRO")
            + energy["SOLAR"] * ef_value(ef, fac, "ELEC_SOLAR")
            + energy["WIND"] * ef_value(ef, fac, "ELEC_WIND")
            + energy["BIO"] * ef_value(ef, fac, "ELEC_BIO")
        )
    return out


def prep_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in NUMERIC_EXCLUDE:
            continue
        out[c] = to_num(out[c], 0.0)
    return out


def calc_crop(crop: pd.DataFrame, elec_ef: pd.DataFrame, ef: pd.DataFrame) -> pd.DataFrame:
    crop = standardize_units(crop, "02_inventory_crop.csv")
    require_nonnegative(crop, "02_inventory_crop.csv")
    required = [
        "CITY", "YEAR", "CROP", "EXTENT_HA", "SYN_KG", "SYN_KM",
        "SYN_N_KG", "SYN_P_KG", "SYN_K_KG",
        "COMPOST_IMP_KG", "COMPOST_IMP_KM",
        "COMPOST_LOC_KG", "COMPOST_LOC_KM",
        "DIGEST_KG", "DIGEST_KM", "DIGEST_MACHINERY_UNITS", "DIGEST_MACHINERY_NEW_UNITS",
        "AGROCHEM_KG", "AGROCHEM_KM",
        "AG_DIESEL_MJ", "AG_GASOL_MJ", "AG_KEROS_MJ", "AG_ELEC_KWH",
        "COMPOST_N_KG", "COMPOST_P_KG", "DIGEST_N_KG", "DIGEST_P_KG", "DIGEST_C_KG",
        "NH3_KG", "NO_KG", "NRUNOFF_KG", "NLEACH_KG",
        "RESIDUE_C_KG", "SOILTEMP", "SAND", "IRRIG_TERM",
    ]
    df = ensure_columns(prep_numeric(crop), required, 0.0)
    df = add_incremental_digest_machinery_units(df)
    df["YEAR"] = to_num(df["YEAR"]).astype(int)
    df["CITY"] = to_num(df["CITY"]).astype(int)
    df["CROP"] = df["CROP"].astype(str).str.upper()
    df = df.merge(elec_ef, on="YEAR", how="left")
    df["REGION"] = df["CITY"].map(city_to_region)


    if "SYN_P2O5_KG" not in df.columns:
        df["SYN_P2O5_KG"] = df["SYN_P_KG"]
    if "SYN_K2O_KG" not in df.columns:
        df["SYN_K2O_KG"] = df["SYN_K_KG"]
    if "COMPOST_P2O5_KG" not in df.columns:
        df["COMPOST_P2O5_KG"] = df["COMPOST_P_KG"]
    if "DIGEST_P2O5_KG" not in df.columns:
        df["DIGEST_P2O5_KG"] = df["DIGEST_P_KG"]
    if "SYN_P_ELEM_KG" not in df.columns:
        df["SYN_P_ELEM_KG"] = df["SYN_P2O5_KG"] * P2O5_TO_P
    if "COMPOST_P_ELEM_KG" not in df.columns:
        df["COMPOST_P_ELEM_KG"] = df["COMPOST_P2O5_KG"] * P2O5_TO_P
    if "DIGEST_P_ELEM_KG" not in df.columns:
        df["DIGEST_P_ELEM_KG"] = df["DIGEST_P2O5_KG"] * P2O5_TO_P

    results = []
    for crop_name in CROPS:
        sub = df[df["CROP"] == crop_name].copy()
        if sub.empty:
            continue

        extent_safe = np.where(sub["EXTENT_HA"] == 0, np.nan, sub["EXTENT_HA"])
        out = sub[["REGION", "CITY", "YEAR"]].copy()

        for fac in FACTORS:
            out[f"SYN_PROD_{crop_name}_{fac}"] = (
                sub["SYN_N_KG"] * ef_value(ef, fac, "SYN_N")
                + sub["SYN_P2O5_KG"] * ef_value(ef, fac, "SYN_P")
                + sub["SYN_K2O_KG"] * ef_value(ef, fac, "SYN_K")
            )
            out[f"SYN_TRANS_{crop_name}_{fac}"] = sub["SYN_KG"] * 0.001 * sub["SYN_KM"] * ef_value(ef, fac, "SYN_TRANS")
            out[f"COMPOST_IMP_PROD_{crop_name}_{fac}"] = sub["COMPOST_IMP_KG"] * ef_value(ef, fac, "ORG")
            out[f"COMPOST_IMP_TRANS_{crop_name}_{fac}"] = sub["COMPOST_IMP_KG"] * 0.001 * sub["COMPOST_IMP_KM"] * ef_value(ef, fac, "ORG_TRANS")
            out[f"COMPOST_LOC_TRANS_{crop_name}_{fac}"] = sub["COMPOST_LOC_KG"] * 0.001 * sub["COMPOST_LOC_KM"] * ef_value(ef, fac, "ORG_TRANS")
            out[f"DIGEST_TRANS_{crop_name}_{fac}"] = sub["DIGEST_KG"] * 0.001 * sub["DIGEST_KM"] * ef_value(ef, fac, "DIG_TRANS")
            out[f"DIGEST_MACHINERY_{crop_name}_{fac}"] = sub["DIGEST_MACHINERY_NEW_UNITS"] * ef_value(ef, fac, "MACHINERY")
            out[f"AGROCHEM_PROD_{crop_name}_{fac}"] = sub["AGROCHEM_KG"] * ef_value(ef, fac, "AGROCHEM")
            out[f"AGROCHEM_TRANS_{crop_name}_{fac}"] = sub["AGROCHEM_KG"] * 0.001 * (300.0 + sub["AGROCHEM_KM"]) * ef_value(ef, fac, "AGROCHEM_TRANS")
            out[f"AG_DIESEL_USE_{crop_name}_{fac}"] = sub["AG_DIESEL_MJ"] * ef_value(ef, fac, "DIESEL_E")
            out[f"AG_GASOL_USE_{crop_name}_{fac}"] = sub["AG_GASOL_MJ"] * ef_value(ef, fac, "GASOL_E")
            out[f"AG_KEROS_USE_{crop_name}_{fac}"] = sub["AG_KEROS_MJ"] * ef_value(ef, fac, "KEROS_E")
            out[f"AG_ELEC_USE_{crop_name}_{fac}"] = sub["AG_ELEC_KWH"] * sub[f"ELEC_{fac}"]

        if crop_name == "RICE":
            rice_ch4 = (
                (0.09509 * ((sub["COMPOST_N_KG"] * 25.0 + sub["RESIDUE_C_KG"] + sub["DIGEST_C_KG"]) / extent_safe) + 6.4)
                * sub["EXTENT_HA"] * (16.0 / 12.0)
            )
            rice_n2o = (sub["SYN_N_KG"] * 0.0031 + sub["COMPOST_N_KG"] * 0.0116 + sub["DIGEST_N_KG"] * 0.0038) * (44.0 / 28.0)
            out["FIELD_DIRECT_RICE_GWP"] = np.nan_to_num(rice_ch4 * 27.2 + rice_n2o * 273.0, nan=0.0)
        elif crop_name == "TEA":
            tea_n2o = (sub["SYN_N_KG"] * 0.021 + sub["COMPOST_N_KG"] * 0.0234375 + sub["DIGEST_N_KG"] * 0.0017) * (44.0 / 28.0)
            out["FIELD_DIRECT_TEA_GWP"] = tea_n2o * 273.0
        elif crop_name == "VEG":
            veg_n2o = (sub["SYN_N_KG"] * 0.0046 + sub["COMPOST_N_KG"] * 0.015 + sub["DIGEST_N_KG"] * 0.0017) * (44.0 / 28.0)
            out["FIELD_DIRECT_VEG_GWP"] = veg_n2o * 273.0

        out[f"FIELD_INDIRECT_{crop_name}_GWP"] = ((sub["NH3_KG"] + sub["NO_KG"]) * 0.014 + (sub["NLEACH_KG"] + sub["NRUNOFF_KG"]) * 0.011) * (44.0 / 28.0) * 273.0
        out[f"FIELD_{crop_name}_MEP"] = sub["NRUNOFF_KG"] * ef_value(ef, "MEP", "FIELD_MEP")
        if crop_name == "RICE":
            out["FIELD_RICE_FEP"] = sub["EXTENT_HA"] * np.exp(-30.54 + 1.89 * sub["SOILTEMP"] + 0.72 * sub["SAND"] + 3.11e-04 * sub["IRRIG_TERM"])
        elif crop_name == "TEA":
            out["FIELD_TEA_FEP"] = (sub["SYN_P_ELEM_KG"] + sub["COMPOST_P_ELEM_KG"] + sub["DIGEST_P_ELEM_KG"]) * 0.0043
        elif crop_name == "VEG":
            out["FIELD_VEG_FEP"] = (sub["SYN_P_ELEM_KG"] + sub["COMPOST_P_ELEM_KG"] + sub["DIGEST_P_ELEM_KG"]) * (0.0257 * 0.45 + 0.0395 * 0.29 + 0.0007 * 0.26)
        results.append(out.groupby(["REGION", "CITY", "YEAR"], as_index=False).sum(numeric_only=True))

    if not results:
        return pd.DataFrame(columns=["REGION", "CITY", "YEAR"])

    res = results[0]
    for block in results[1:]:
        res = res.merge(block, on=["REGION", "CITY", "YEAR"], how="outer")
    return res.fillna(0.0)


def calc_manure(
    manure: pd.DataFrame, elec_ef: pd.DataFrame, ef: pd.DataFrame, scenario: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manure = standardize_units(manure, "02_inventory_manure.csv")
    require_nonnegative(manure, "02_inventory_manure.csv")
    required = [
        "FACILITY", "YEAR", "MANURE_KG", "MANURE_KM",
        "DAIRY_MANURE_KG", "CATTLE_MANURE_KG", "SWINE_MANURE_KG", "CHICKEN_MANURE_KG", "BROILER_MANURE_KG",
        "COMPOST_N", "COMPOST_P2O5", "DIGEST_N", "DIGEST_P2O5",
        "RESIDUE_SUPPLY_KG", "RESIDUE_IMPORT_KG", "RESIDUE_SUPPLIED_KM", "RESIDUE_IMPORT_KM",
        "BIOGAS_CH4_NM3", "MWAD_ELEC_USE_KWH", "MWAD_WATER_USE_M3",
    ]
    df = ensure_columns(prep_numeric(manure), required, 0.0)
    if "TYPE" not in df.columns:
        df["TYPE"] = ""
    df["TYPE"] = df["TYPE"].astype(str).str.upper().str.strip()

    mwad_mask = df["TYPE"].eq("MWAD")
    for col in ["RESIDUE_SUPPLY_KG", "RESIDUE_IMPORT_KG", "RESIDUE_SUPPLIED_KM", "RESIDUE_IMPORT_KM", "COMPOST_N", "COMPOST_P2O5"]:
        if col in df.columns:
            df.loc[mwad_mask, col] = 0.0
    df["YEAR"] = to_num(df["YEAR"]).astype(int)
    df["CITY"] = df["FACILITY"].map(facility_to_city).astype(int)
    df["REGION"] = df["CITY"].map(city_to_region)
    df = df.merge(elec_ef, on="YEAR", how="left")

    species_cols = {
        "DAIRY": "DAIRY_MANURE_KG",
        "CATTLE": "CATTLE_MANURE_KG",
        "SWINE": "SWINE_MANURE_KG",
        "CHICKEN": "CHICKEN_MANURE_KG",
        "BROILER": "BROILER_MANURE_KG",
    }

    out = df[["REGION", "CITY", "YEAR"]].copy()
    for fac in FACTORS:
        manure_trans = pd.Series(0.0, index=df.index)
        compost_prod = pd.Series(0.0, index=df.index)
        for species, manure_col in species_cols.items():
            manure_kg_species = series_or_zero(df, manure_col)
            manure_trans = manure_trans + manure_kg_species * 0.001 * df["MANURE_KM"] * ef_value(ef, fac, f"MANURE_{species}_TRANS")
            compost_prod = compost_prod + manure_kg_species * ef_value(ef, fac, f"COMPOST_{species}")

        out[f"MANURE_TRANS_{fac}"] = manure_trans
        out[f"RESIDUE_TRANS_{fac}"] = (
            df["RESIDUE_SUPPLY_KG"] * 0.001 * df["RESIDUE_SUPPLIED_KM"] * ef_value(ef, fac, "TDAD_REF_TRANS")
            + df["RESIDUE_IMPORT_KG"] * 0.001 * df["RESIDUE_IMPORT_KM"] * ef_value(ef, fac, "TDAD_REF_TRANS")
        )
        out[f"COMPOST_LOC_PROD_{fac}"] = compost_prod
        out[f"DIGEST_MANURE_PROD_{fac}"] = np.where(
            scenario.upper() == "CONV",
            df["BIOGAS_CH4_NM3"] * 33.906 * ef_value(ef, fac, "MWAD_COMBUST")
            + df["MWAD_ELEC_USE_KWH"] * df[f"ELEC_{fac}"]
            + df["MWAD_WATER_USE_M3"] * ef_value(ef, fac, "MWAD_WATER"),
            0.0,
        )

    manure_res = out.groupby(["REGION", "CITY", "YEAR"], as_index=False).sum(numeric_only=True)
    manure_energy = df.groupby(["CITY", "YEAR"], as_index=False).agg(
        MANURE_BIOGAS_CH4_NM3=("BIOGAS_CH4_NM3", "sum")
    )
    return manure_res, manure_energy



def calc_waste(
    waste: pd.DataFrame,
    crop: pd.DataFrame,
    manure_energy: pd.DataFrame,
    elec_ef: pd.DataFrame,
    ef: pd.DataFrame,
    scenario: str,
    enable_gridgas: bool = False,
) -> pd.DataFrame:
    waste = standardize_units(waste, "02_inventory_waste.csv")
    crop = standardize_units(crop, "02_inventory_crop.csv")
    require_nonnegative(waste, "02_inventory_waste.csv")
    required = [
        "FACILITY", "YEAR", "TYPE", "EVENT", "ENERGY_USE",
        "WASTE_KG", "FOOD_KG", "MANURE_KG", "DAIRY_MANURE_KG", "CATTLE_MANURE_KG", "SWINE_MANURE_KG", "CHICKEN_MANURE_KG", "BROILER_MANURE_KG", "WASTE_KM", "MANURE_KM", "PLASTIC_KG", "SYNTEX_KG",
        "INC_INPUT_KG", "INC_KEROS_L", "INC_ELEC_KWH", "ASH_KG", "ASH_KM",
        "TDAD_ELEC_KWH", "TDAD_BIOGAS_NM3", "TDAD_DIGEST_KG", "TDAD_REFUSE_KG", "TDAD_REFUSE_KM", "TDAD_DIGSOLID_KG",
        "RDF_KG", "RDF_KM", "RDF_REFUSE_KG", "RDF_REFUSE_KM",
        "MWAD_ELEC_KWH", "MWAD_WATER_M3", "MWAD_BIOGAS_NM3", "MWAD_DIGEST_KG", "DIGEST_UNUSED_KG", "MWAD_DIGSOLID_KG", "MWAD_REFUSE_KG", "MWAD_REFUSE_KM",
        "INC_CAPEX", "TDAD_CAPEX", "RDF_CAPEX", "MWAD_CAPEX",
        "ELEC_INC_KWH", "ELEC_TDAD_KWH", "ELEC_RDF_KWH", "ELEC_MWAD_KWH",
    ]
    df = ensure_columns(prep_numeric(waste), required, 0.0)
    df["YEAR"] = to_num(df["YEAR"]).astype(int)
    df["CITY"] = df["FACILITY"].map(facility_to_city).astype(int)
    df["REGION"] = df["CITY"].map(city_to_region)
    df["TYPE"] = df["TYPE"].astype(str).str.upper()
    df["ENERGY_USE"] = df["ENERGY_USE"].astype(str).str.upper()
    df["EVENT"] = df["EVENT"].astype(str).replace("nan", "").str.upper().str.strip()

    df = df.sort_values(["FACILITY", "TYPE", "YEAR"]).copy()
    df["EVENT_CARRY"] = (
        df.groupby(["FACILITY", "TYPE"], sort=False)["EVENT"]
        .transform(lambda s: s.replace("", np.nan).ffill())
        .fillna("")
    )

    df = df.merge(elec_ef, on="YEAR", how="left")

    crop_used = ensure_columns(prep_numeric(crop), ["CITY", "YEAR", "DIGEST_KG"], 0.0)
    crop_used["CITY"] = to_num(crop_used["CITY"]).astype(int)
    crop_used["YEAR"] = to_num(crop_used["YEAR"]).astype(int)
    digest_used = crop_used.groupby(["CITY", "YEAR"], as_index=False)["DIGEST_KG"].sum().rename(columns={"DIGEST_KG": "MWAD_USED"})
    df = df.merge(digest_used, on=["CITY", "YEAR"], how="left")
    df["MWAD_USED"] = to_num(df["MWAD_USED"], 0.0)

    manure_energy = manure_energy.copy()
    manure_energy["CITY"] = to_num(manure_energy["CITY"]).astype(int)
    manure_energy["YEAR"] = to_num(manure_energy["YEAR"]).astype(int)
    manure_energy["MANURE_BIOGAS_CH4_NM3"] = to_num(manure_energy.get("MANURE_BIOGAS_CH4_NM3", 0.0), 0.0)

    base_keys = pd.concat(
        [
            df[["CITY", "YEAR"]].drop_duplicates(),
            manure_energy[["CITY", "YEAR"]].drop_duplicates(),
        ],
        ignore_index=True,
    ).drop_duplicates()
    base_keys["REGION"] = base_keys["CITY"].map(city_to_region)
    base_keys = base_keys.merge(elec_ef, on="YEAR", how="left")
    base_keys = base_keys.merge(manure_energy, on=["CITY", "YEAR"], how="left")
    base_keys["MANURE_BIOGAS_CH4_NM3"] = to_num(base_keys["MANURE_BIOGAS_CH4_NM3"], 0.0)

    is_inc = df["TYPE"].eq("INC")
    is_tdad = df["TYPE"].eq("TDAD")
    is_rdf = df["TYPE"].eq("RDF")
    is_mwad = df["TYPE"].eq("MWAD")
    is_compost = df["TYPE"].eq("COMPOST")

    is_rebuild = df["EVENT_CARRY"].isin(["NEW", "REBUILD"])
    is_renew = df["EVENT_CARRY"].isin(["RENEW", "RENOV", "RENOVATE"])

    factor_blocks = []
    for fac in FACTORS:
        block = df[["REGION", "CITY", "YEAR"]].copy()

        manure_trans_collect = (
            series_or_zero(df, "DAIRY_MANURE_KG") * 0.001 * df["MANURE_KM"] * ef_value(ef, fac, "MANURE_DAIRY_TRANS")
            + series_or_zero(df, "CATTLE_MANURE_KG") * 0.001 * df["MANURE_KM"] * ef_value(ef, fac, "MANURE_CATTLE_TRANS")
            + series_or_zero(df, "SWINE_MANURE_KG") * 0.001 * df["MANURE_KM"] * ef_value(ef, fac, "MANURE_SWINE_TRANS")
            + series_or_zero(df, "CHICKEN_MANURE_KG") * 0.001 * df["MANURE_KM"] * ef_value(ef, fac, "MANURE_CHICKEN_TRANS")
            + series_or_zero(df, "BROILER_MANURE_KG") * 0.001 * df["MANURE_KM"] * ef_value(ef, fac, "MANURE_BROILER_TRANS")
        )
        block[f"WASTE_COLLECT_{fac}"] = (
            df["WASTE_KG"] * 0.001 * df["WASTE_KM"] * ef_value(ef, fac, "WASTE_TRANS")
            + manure_trans_collect
        )

        inc_process = np.where(
            df["ENERGY_USE"].eq("ELEC"),
            df["INC_INPUT_KG"] * ef_value(ef, fac, "INC_ELEC_GEN"),
            df["INC_INPUT_KG"] * ef_value(ef, fac, "INC_NO_GEN"),
        )
        waste_fossilc_gwp = (
            (df["PLASTIC_KG"] * 0.9054 * 0.757 + df["SYNTEX_KG"] * 0.931 * 0.5321) * (44.0 / 12.0)
            if fac == "GWP"
            else 0.0
        )
        block[f"INC_{fac}"] = np.where(
            is_inc,
            waste_fossilc_gwp
            + (0.0 if fac == "GWP" else inc_process)
            + (df["INC_KEROS_L"] / 0.0274013757) * ef_value(ef, fac, "INC_KEROS_E")
            + df["INC_ELEC_KWH"] * df[f"ELEC_{fac}"]
            + df["ASH_KG"] * 0.001 * df["ASH_KM"] * ef_value(ef, fac, "ASH_TRANS")
            + df["ASH_KG"] * ef_value(ef, fac, "ASH_LANDFILL"),
            0.0,
        )

        block[f"TDAD_{fac}"] = np.where(
            is_tdad,
            df["TDAD_BIOGAS_NM3"] * 33.906 * ef_value(ef, fac, "TDAD_COMBUST")
            + df["TDAD_ELEC_KWH"] * df[f"ELEC_{fac}"]
            + df["TDAD_DIGEST_KG"] * 0.001 * ef_value(ef, fac, "TDAD_DIGWW")
            + df["TDAD_DIGSOLID_KG"] * 0.001 * ef_value(ef, fac, "TDAD_SLUDGE")
            + df["TDAD_DIGSOLID_KG"] * 0.001 * df["TDAD_REFUSE_KM"] * ef_value(ef, fac, "TDAD_REF_TRANS")
            + df["TDAD_REFUSE_KG"] * 0.001 * df["TDAD_REFUSE_KM"] * ef_value(ef, fac, "TDAD_REF_TRANS"),
            0.0,
        )

        block[f"RDF_{fac}"] = np.where(
            is_rdf,
            df["RDF_KG"] * ef_value(ef, fac, "RDF_PROD")
            + df["RDF_KG"] * 0.001 * df["RDF_KM"] * ef_value(ef, fac, "RDF_TRANS")
            + df["RDF_KG"] * 0.056 * ef_value(ef, fac, "RDF_COMBUST")
            + df["RDF_REFUSE_KG"] * 0.001 * df["RDF_REFUSE_KM"] * ef_value(ef, fac, "RDF_REF_TRANS"),
            0.0,
        )

        block[f"COMPOST_FOOD_PROC_{fac}"] = np.where(
            is_compost,
            df["FOOD_KG"] * ef_value(ef, fac, "COMPOST_FOOD"),
            0.0,
        )

        block[f"MWAD_{fac}"] = np.where(
            (scenario.upper() != "CONV") & is_mwad,
            df["MWAD_BIOGAS_NM3"] * 33.906 * ef_value(ef, fac, "MWAD_COMBUST")
            + df["MWAD_ELEC_KWH"] * df[f"ELEC_{fac}"]
            + df["MWAD_WATER_M3"] * ef_value(ef, fac, "MWAD_WATER")
            + df["DIGEST_UNUSED_KG"] * 0.001 * ef_value(ef, fac, "MWAD_DIGWW")
            + df["MWAD_DIGSOLID_KG"] * 0.001 * ef_value(ef, fac, "MWAD_SLUDGE")
            + df["MWAD_DIGSOLID_KG"] * 0.001 * df["MWAD_REFUSE_KM"] * ef_value(ef, fac, "MWAD_REF_TRANS")
            + df["MWAD_REFUSE_KG"] * 0.001 * df["MWAD_REFUSE_KM"] * ef_value(ef, fac, "MWAD_REF_TRANS"),
            0.0,
        )

        block[f"CONSTRUCT_INC_{fac}"] = np.where(
            is_inc & is_rebuild,
            df["INC_CAPEX"] * ef_value(ef, fac, "WASTE_REBUILD"),
            np.where(is_inc & is_renew, df["INC_CAPEX"] * ef_value(ef, fac, "WASTE_RENOV"), 0.0),
        )
        block[f"CONSTRUCT_TDAD_{fac}"] = np.where(
            is_tdad & is_rebuild,
            df["TDAD_CAPEX"] * ef_value(ef, fac, "WASTE_REBUILD"),
            np.where(is_tdad & is_renew, df["TDAD_CAPEX"] * ef_value(ef, fac, "WASTE_RENOV"), 0.0),
        )
        block[f"CONSTRUCT_RDF_{fac}"] = np.where(
            is_rdf & is_rebuild,
            df["RDF_CAPEX"] * ef_value(ef, fac, "WASTE_REBUILD"),
            np.where(is_rdf & is_renew, df["RDF_CAPEX"] * ef_value(ef, fac, "WASTE_RENOV"), 0.0),
        )
        block[f"CONSTRUCT_MWAD_{fac}"] = np.where(
            is_mwad & is_rebuild,
            df["MWAD_CAPEX"] * ef_value(ef, fac, "WASTE_REBUILD"),
            np.where(is_mwad & is_renew, df["MWAD_CAPEX"] * ef_value(ef, fac, "WASTE_RENOV"), 0.0),
        )

        block[f"ELEC_INC_{fac}"] = np.where(is_inc, df["ELEC_INC_KWH"] * df[f"ELEC_{fac}"], 0.0)
        block[f"ELEC_TDAD_{fac}"] = np.where(is_tdad, df["ELEC_TDAD_KWH"] * df[f"ELEC_{fac}"], 0.0)
        block[f"ELEC_RDF_{fac}"] = np.where(is_rdf, df["ELEC_RDF_KWH"] * df[f"ELEC_{fac}"], 0.0)

        agg = block.groupby(["REGION", "CITY", "YEAR"], as_index=False).sum(numeric_only=True)

        mwad_kwh_from_manure = base_keys["MANURE_BIOGAS_CH4_NM3"] * 33906.0 * 0.35 / 3600.0
        extra = base_keys[["REGION", "CITY", "YEAR"]].copy()
        extra[f"ELEC_MWAD_{fac}"] = np.where(
            enable_gridgas and scenario.upper() == "OPT",
            0.0,
            mwad_kwh_from_manure * base_keys[f"ELEC_{fac}"],
        )
        extra[f"GRIDGAS_MWAD_{fac}"] = np.where(
            enable_gridgas and scenario.upper() == "OPT",
            -base_keys["MANURE_BIOGAS_CH4_NM3"] * ef_value(ef, fac, "GRID_GAS"),
            0.0,
        )

        waste_side = df[["REGION", "CITY", "YEAR"]].copy()
        waste_side[f"ELEC_MWAD_{fac}"] = df["ELEC_MWAD_KWH"] * df[f"ELEC_{fac}"]
        waste_side[f"GRIDGAS_TDAD_{fac}"] = np.where(
            enable_gridgas and scenario.upper() == "OPT",
            np.where(is_tdad & df["ENERGY_USE"].eq("GRIDGAS"), -df["TDAD_BIOGAS_NM3"] * ef_value(ef, fac, "GRID_GAS"), 0.0),
            0.0,
        )
        waste_side[f"GRIDGAS_MWAD_{fac}"] = np.where(
            enable_gridgas and scenario.upper() == "OPT",
            np.where(is_mwad & df["ENERGY_USE"].eq("GRIDGAS"), -df["MWAD_BIOGAS_NM3"] * ef_value(ef, fac, "GRID_GAS"), 0.0),
            0.0,
        )
        waste_side = waste_side.groupby(["REGION", "CITY", "YEAR"], as_index=False).sum(numeric_only=True)

        agg = agg.merge(waste_side, on=["REGION", "CITY", "YEAR"], how="outer")
        agg = agg.merge(extra, on=["REGION", "CITY", "YEAR"], how="outer", suffixes=("", "_MANURE"))

        if f"ELEC_MWAD_{fac}_MANURE" in agg.columns:
            agg[f"ELEC_MWAD_{fac}"] = to_num(agg.get(f"ELEC_MWAD_{fac}", 0.0), 0.0) + to_num(agg[f"ELEC_MWAD_{fac}_MANURE"], 0.0)
            agg = agg.drop(columns=[f"ELEC_MWAD_{fac}_MANURE"])
        if f"GRIDGAS_MWAD_{fac}_MANURE" in agg.columns:
            agg[f"GRIDGAS_MWAD_{fac}"] = to_num(agg.get(f"GRIDGAS_MWAD_{fac}", 0.0), 0.0) + to_num(agg[f"GRIDGAS_MWAD_{fac}_MANURE"], 0.0)
            agg = agg.drop(columns=[f"GRIDGAS_MWAD_{fac}_MANURE"])

        factor_blocks.append(agg)

    if not factor_blocks:
        return pd.DataFrame(columns=["REGION", "CITY", "YEAR"])

    out = factor_blocks[0]
    for block in factor_blocks[1:]:
        out = out.merge(block, on=["REGION", "CITY", "YEAR"], how="outer")

    return out.groupby(["REGION", "CITY", "YEAR"], as_index=False).sum(numeric_only=True)
def merge_blocks(crop_res: pd.DataFrame, manure_res: pd.DataFrame, waste_res: pd.DataFrame) -> pd.DataFrame:
    out = crop_res.merge(manure_res, on=["REGION", "CITY", "YEAR"], how="outer")
    out = out.merge(waste_res, on=["REGION", "CITY", "YEAR"], how="outer")
    return out.fillna(0.0).sort_values(["REGION", "CITY", "YEAR"]).reset_index(drop=True)


def factor_columns(factor: str) -> list[str]:
    cols = ["REGION", "CITY", "YEAR"]
    crop_bases = [
        "SYN_PROD", "SYN_TRANS", "COMPOST_IMP_PROD", "COMPOST_IMP_TRANS", "COMPOST_LOC_TRANS",
        "DIGEST_TRANS", "DIGEST_MACHINERY", "AGROCHEM_PROD", "AGROCHEM_TRANS",
        "AG_DIESEL_USE", "AG_GASOL_USE", "AG_KEROS_USE", "AG_ELEC_USE",
    ]
    for base in crop_bases:
        for crop_name in CROPS:
            cols.append(f"{base}_{crop_name}_{factor}")

    if factor == "GWP":
        cols += [
            "FIELD_DIRECT_RICE_GWP", "FIELD_DIRECT_TEA_GWP", "FIELD_DIRECT_VEG_GWP",
            "FIELD_INDIRECT_RICE_GWP", "FIELD_INDIRECT_TEA_GWP", "FIELD_INDIRECT_VEG_GWP",
        ]
    if factor == "MEP":
        cols += [
            "FIELD_RICE_MEP", "FIELD_TEA_MEP", "FIELD_VEG_MEP",
        ]
    if factor == "FEP":
        cols += [
            "FIELD_RICE_FEP", "FIELD_TEA_FEP", "FIELD_VEG_FEP",
        ]

    cols += [
        f"MANURE_TRANS_{factor}", f"RESIDUE_TRANS_{factor}", f"COMPOST_LOC_PROD_{factor}", f"DIGEST_MANURE_PROD_{factor}",
        f"WASTE_COLLECT_{factor}", f"INC_{factor}", f"COMPOST_FOOD_PROC_{factor}", f"TDAD_{factor}", f"RDF_{factor}", f"MWAD_{factor}",
        f"CONSTRUCT_INC_{factor}", f"CONSTRUCT_TDAD_{factor}", f"CONSTRUCT_RDF_{factor}", f"CONSTRUCT_MWAD_{factor}",
        f"ELEC_INC_{factor}", f"ELEC_TDAD_{factor}", f"ELEC_MWAD_{factor}", f"ELEC_RDF_{factor}",
        f"GRIDGAS_TDAD_{factor}", f"GRIDGAS_MWAD_{factor}",
    ]
    return cols


def write_outputs(res: pd.DataFrame, out_dir: Path, scenario: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fac in FACTORS:
        desired = factor_columns(fac)
        df = res.copy()
        for c in desired:
            if c not in df.columns:
                df[c] = 0.0
        df = df[desired]
        df.to_csv(out_dir / f"04_{fac}_{scenario.upper()}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="CONV", choices=["CONV", "ALT", "OPT"])
    ap.add_argument("--input-dir", default=".")
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--output-dir", default=".")
    ap.add_argument("--enable-gridgas", action="store_true")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir is not None else input_dir
    output_dir = Path(args.output_dir)

    required_dataset = ["02_EF.csv", "02_energy.csv"]
    required_inventory = ["02_inventory_crop.csv", "02_inventory_manure.csv", "02_inventory_waste.csv"]

    missing_dataset = [name for name in required_dataset if not (dataset_dir / name).exists()]
    missing_inventory = [name for name in required_inventory if not (input_dir / name).exists()]
    if missing_dataset or missing_inventory:
        messages = []
        if missing_dataset:
            messages.append(f"dataset-dir missing: {missing_dataset} (dataset_dir={dataset_dir})")
        if missing_inventory:
            messages.append(f"input-dir missing: {missing_inventory} (input_dir={input_dir})")
        raise FileNotFoundError("; ".join(messages))

    ef = pd.read_csv(dataset_dir / "02_EF.csv")
    energy = pd.read_csv(dataset_dir / "02_energy.csv")
    crop = pd.read_csv(input_dir / "02_inventory_crop.csv")
    manure = pd.read_csv(input_dir / "02_inventory_manure.csv")
    waste = pd.read_csv(input_dir / "02_inventory_waste.csv")

    elec_ef = build_elec_ef(energy, ef)
    crop_res = calc_crop(crop, elec_ef, ef)
    manure_res, manure_energy = calc_manure(manure, elec_ef, ef, args.scenario)
    waste_res = calc_waste(waste, crop, manure_energy, elec_ef, ef, args.scenario, enable_gridgas=args.enable_gridgas)
    res = merge_blocks(crop_res, manure_res, waste_res)
    write_outputs(res, output_dir, args.scenario)
    print(f"Done. Wrote factor files for scenario={args.scenario} to {output_dir} (dataset_dir={dataset_dir}, input_dir={input_dir})")


if __name__ == "__main__":
    main()