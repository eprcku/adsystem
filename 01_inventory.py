"インベントリ構築（作物栽培・廃棄物処理）"

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except Exception as exc:
    raise ImportError("geopandas is required for 01_inventory.py") from exc

try:
    import requests
except Exception as exc:
    raise ImportError("requests is required for 01_inventory.py") from exc

YEARS: list[int] = list(range(2029, 2051))
OUTPUT_START_YEAR = 2030
CROP_CARRYOVER_START_YEAR = 2029 # 水稲残渣は1年前のものを取るので
DIESEL_L_PER_MJ = 0.0262868597 # L/MJ
GASOL_L_PER_MJ = 0.0299726263 # L/MJ
KEROS_L_PER_MJ = 0.0274013757 # L/MJ
DEFAULT_OSRM_URL = "http://router.project-osrm.org/route/v1/driving" # 運転距離
KYOTO_CITIES = {101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111} # 京都市内区
KYOTO_CHOICES = ["KYOTO_NORTH", "KYOTO_NE", "KYOTO_SOUTH"]
CITY_TO_FAC = {
    "FUKUCHIYAMA": [201], "MAIZURU": [202], "AYABE": [203],
    "SAKURAZUKA": [206, 213, 407], "KYOTANABE": [211], "MINEYAMA": [212],
    "ORII": [204, 207, 210, 322, 343, 344, 364, 365, 367],
    "HASEYAMA": [204, 207, 210, 322, 343, 344, 364, 365, 367],
    "KIZUGAWA": [214, 366], "OTOKUNI": [208, 209, 303],
    "MIYAZUYOSA": [205, 463, 465],
}
GROUP_A = {101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 206, 213, 407}
GROUP_B = {201, 202, 203, 205, 212, 463, 465}
GROUP_C = {204, 207, 208, 209, 210, 211, 214, 303, 322, 343, 344, 364, 365, 366, 367}
BOUNDARY = {
    "KYOTO_NORTH": KYOTO_CITIES, "KYOTO_NE": KYOTO_CITIES, "KYOTO_SOUTH": KYOTO_CITIES,
    "FUKUCHIYAMA": {201}, "MAIZURU": {202}, "AYABE": {203},
    "SAKURAZUKA": {206, 213, 407}, "KYOTANABE": {211}, "MINEYAMA": {212},
    "ORII": {204, 207, 210, 322, 343, 344, 364, 365, 367},
    "HASEYAMA": {204, 207, 210, 322, 343, 344, 364, 365, 367},
    "KIZUGAWA": {214, 366}, "OTOKUNI": {208, 209, 303},
    "MIYAZUYOSA": {205, 463, 465},
}
RDF_CONV = 2.404
INC_ENERGY_L_PER_KG = 0.00373
INC_ELEC_KWH_PER_KG = 0.0264
DEFAULT_ASH_RATIO = 0.10
AIRTEMP_FALLBACK = 15.0

NH3_AIRTEMP_MONTH_COLS = ["AIRTEMP_M03", "AIRTEMP_M05", "AIRTEMP_M06", "AIRTEMP_M09", "AIRTEMP_M11"]
NH3_WIND_MONTH_COLS = ["WIND_M03", "WIND_M05", "WIND_M06", "WIND_M09", "WIND_M11"]
NH3_ENV_MONTH_COLS = NH3_AIRTEMP_MONTH_COLS + NH3_WIND_MONTH_COLS
KJ_PER_KWH = 3600.0
BIOGAS_HV = 33906.0
RDF_HV = 55.56
MWAD_DIGSOLID_TS_RECOVERY = 0.95
MWAD_DIGSOLID_WATER = 0.75
TDAD_DIGSOLID_TS_RECOVERY = 0.95
TDAD_DIGSOLID_WATER = 0.75
POULTRY = {"CHICKEN", "BROILER"}
R_MIN = -0.03
R_MAX = 0.03

SPECIES_RATIO_MAP = {
    "DAIRY": "DAIRY_RATIO",
    "CATTLE": "CATTLE_RATIO",
    "SWINE": "SWINE_RATIO",
    "CHICKEN": "CHICKEN_RATIO",
    "BROILER": "BROILER_RATIO",
}

MANURE_UNIT_RATES = {
    "DAIRY": {"F": 47.0, "U": 17.0},
    "CATTLE": {"F": 10.4, "U": 5.5},
    "SWINE": {"F": 1.9, "U": 3.8},
    "CHICKEN": {"E": 0.086},
    "BROILER": {"E": 0.082},
}

RESIDUE_REQ_COEF = {
    "DAIRY": 0.38,
    "CATTLE": 0.38,
    "SWINE": 0.132,
    "CHICKEN": 0.096,
    "BROILER": 0.096,
}

COMPOST_COEF = {
    "DAIRY": 0.665,
    "CATTLE": 0.665,
    "SWINE": 0.415,
    "CHICKEN": 0.327,
    "BROILER": 0.327,
}

ORG_SHARE_2020 = {"RICE": 0.000931469, "TEA": 0.037230458, "VEG": 0.013137622}

VEG_CULTIVAR_RATIOS_KYOTO = {
    "spring": 0.380259424,
    "summer": 0.418565059,
    "autumn": 0.50336441,
    "winter": 0.192987434,
}

VEG_CULTIVAR_RATIOS_OTHER = {
    "spring": 0.406757018,
    "summer": 0.527666989,
    "autumn": 0.539438529,
    "winter": 0.23018393,
}

ORG_SHARE_2050 = 0.25
ORG_PRIORITY_CITIES = {"RICE": {206, 212}, "VEG": {206, 212}, "TEA": {344, 365, 367}}
MANURE_OUTPUT_COLUMNS = [
    "FACILITY",
    "YEAR",
    "TYPE",
    "MANURE_KG",
    "MANURE_N",
    "DAIRY_MANURE_KG",
    "CATTLE_MANURE_KG",
    "SWINE_MANURE_KG",
    "CHICKEN_MANURE_KG",
    "BROILER_MANURE_KG",
    "MANURE_KM",
    "RESIDUE_REQ_KG",
    "RESIDUE_SUPPLY_KG",
    "RESIDUE_IMPORT_KG",
    "RESIDUE_SUPPLIED_KM",
    "RESIDUE_IMPORT_KM",
    "COMPOST_KG",
    "COMPOST_USED_KG",
    "COMPOST_N",
    "COMPOST_P2O5",
    "DIGEST_KG",
    "DIGEST_C",
    "DIGEST_N",
    "DIGEST_P2O5",
    "BIOGAS_CH4_NM3",
    "MWAD_ELEC_USE_KWH",
    "MWAD_HEAT_USE_MJ",
    "MWAD_WATER_USE_M3",
    "MWAD_REFUSED_KG",
]

CROP_OUTPUT_COLUMNS = [
    "KEY",
    "CITY",
    "YEAR",
    "CROP",
    "CULTIVAR",
    "EXTENT_HA",
    "ORG_FLAG",
    "SYN_KG",
    "SYN_KM",
    "SYN_P_KG",
    "SYN_K_KG",
    "COMPOST_KG",
    "COMPOST_LOC_KG",
    "COMPOST_IMP_KG",
    "COMPOST_LOC_KM",
    "COMPOST_IMP_KM",
    "DIGEST_KG",
    "DIGEST_KM",
    "DIGEST_MACHINERY_UNITS",
    "DIGEST_MACHINERY_NEW_UNITS",
    "AGROCHEM_KG",
    "AGROCHEM_KM",
    "AG_DIESEL_L",
    "AG_DIESEL_MJ",
    "AG_GASOL_L",
    "AG_GASOL_MJ",
    "AG_KEROS_L",
    "AG_KEROS_MJ",
    "AG_ELEC_KWH",
    "SYN_N_KG",
    "COMPOST_N_KG",
    "COMPOST_P_KG",
    "DIGEST_N_KG",
    "DIGEST_P_KG",
    "DIGEST_C_KG",
    "NH3_KG",
    "NO_KG",
    "NRUNOFF_KG",
    "NLEACH_KG",
    "SAND",
    "IRRIG_TERM",
    "RESIDUE_KG",
    "RESIDUE_C_KG",
    "RESIDUE_N_KG",
    "MANURE_DAIRY",
    "MANURE_CATTLE",
    "MANURE_SWINE",
    "MANURE_CHICKEN",
    "MANURE_BROILER",
]

WASTE_OUTPUT_COLUMNS = [
    "FACILITY",
    "YEAR",
    "TYPE",
    "CAPACITY",
    "EVENT",
    "ENERGY_USE",
    "WASTE_KG",
    "MANURE_KG",
    "DAIRY_MANURE_KG",
    "CATTLE_MANURE_KG",
    "SWINE_MANURE_KG",
    "CHICKEN_MANURE_KG",
    "BROILER_MANURE_KG",
    "WASTE_KM",
    "MANURE_KM",
    "PLASTIC_KG",
    "SYNTEX_KG",
    "FOOD_KG",
    "PAPER_KG",
    "NATTEX_KG",
    "WOOD_KG",
    "INC_INPUT_KG",
    "INC_KEROS_L",
    "INC_ELEC_KWH",
    "ASH_KG",
    "ASH_KM",
    "TDAD_ELEC_KWH",
    "TDAD_HEAT_MJ",
    "TDAD_BIOGAS_NM3",
    "TDAD_DIGEST_KG",
    "TDAD_REFUSE_KG",
    "TDAD_REFUSE_KM",
    "TDAD_DIGSOLID_KG",
    "RDF_KG",
    "RDF_KM",
    "RDF_REFUSE_KG",
    "RDF_REFUSE_KM",
    "MWAD_ELEC_KWH",
    "MWAD_HEAT_MJ",
    "MWAD_WATER_M3",
    "MWAD_BIOGAS_NM3",
    "MWAD_DIGEST_KG",
    "DIGEST_UNUSED_KG",
    "MWAD_DIGSOLID_KG",
    "MWAD_DIGEST_ORGC_KG",
    "MWAD_DIGEST_N_KG",
    "MWAD_DIGEST_P2O5_KG",
    "MWAD_DIGEST_K2O_KG",
    "MWAD_REFUSE_KG",
    "MWAD_REFUSE_KM",
    "INC_CAPEX",
    "TDAD_CAPEX",
    "RDF_CAPEX",
    "MWAD_CAPEX",
    "ELEC_INC_KWH",
    "ELEC_TDAD_KWH",
    "ELEC_RDF_KWH",
    "ELEC_MWAD_KWH",
]




WASTE_LINK_COLUMNS = [
    "YEAR", "FROM_KEY", "FROM_CITY", "TO_FACILITY", "TYPE",
    "FOOD_N_KG", "PAPER_N_KG", "WOOD_N_KG",
    "DAIRY_N_KG", "CATTLE_N_KG", "SWINE_N_KG", "CHICKEN_N_KG", "BROILER_N_KG",
    "RESIDUE_N_KG",
]

FERTILIZER_LINK_COLUMNS = [
    "YEAR", "FROM_FACILITY", "TO_KEY", "TO_CITY", "CROP", "CULTIVAR",
    "SYN_N_KG", "COMPOST_N_KG", "DIGEST_N_KG", "RESIDUE_N_KG",
]

RESIDUE_N_PER_KG = (0.835 / 1.075 * 0.00541 + 0.240 / 0.835 * 0.009)

@dataclass(slots=True)
class ScenarioSettings:
    scenario_name: str = "BASE"
    active_organic_n_source: str = "COMPOST"
    use_alt_adfac: bool = False
    use_osrm: bool = True
    facility_plan_path: Path | None = None
    digestate_links_path: Path | None = None
    fertilizer_links_path: Path | None = None
    compost_supply_path: Path | None = None
    fallback_flows_path: Path | None = None
    org_mode: str = "baseline"
    link_mode: str = "single"
    disable_crop_digestate: bool = False


@dataclass(slots=True)
class InventoryPaths:
    dataset_dir: Path
    gis_dir: Path
    output_dir: Path
    cache_dir: Path
    script_dir: Path | None = None

    @property
    def rcom_csv(self) -> Path:
        return self.dataset_dir / "00_rcom.csv"

    @property
    def rcom_shp(self) -> Path:
        return self.gis_dir / "00_rcom.shp"

    @property
    def offices_shp(self) -> Path:
        return self.gis_dir / "00_offices.shp"

    @property
    def manurefac_csv(self) -> Path:
        return self.dataset_dir / "00_manurefac.csv"

    @property
    def livestock_csv(self) -> Path:
        return self.dataset_dir / "00_livestock.csv"

    @property
    def cropland_csv(self) -> Path:
        return self.dataset_dir / "00_cropland.csv"

    @property
    def agroinput_csv(self) -> Path:
        return self.dataset_dir / "00_agroinput.csv"

    @property
    def biowaste_csv(self) -> Path:
        return self.dataset_dir / "00_biowaste.csv"

    @property
    def env_csv(self) -> Path:
        return self.dataset_dir / "00_env.csv"

    @property
    def wastegen_csv(self) -> Path:
        return self.dataset_dir / "00_wastegen.csv"

    @property
    def population_csv(self) -> Path:
        return self.dataset_dir / "00_population.csv"

    @property
    def mswfac_csv(self) -> Path:
        return self.dataset_dir / "00_mswfac.csv"

    @property
    def cost_csv(self) -> Path:
        return self.dataset_dir / "00_cost.csv"


@dataclass(slots=True)
class InventoryContext:
    paths: InventoryPaths
    scenario: ScenarioSettings
    years: list[int] = field(default_factory=lambda: YEARS.copy())
    osrm_url: str = DEFAULT_OSRM_URL
    timeout_sec: int = 30
    requests_session: requests.Session | None = None
    distance_cache: dict[str, float] = field(default_factory=dict)
    shared_inputs: dict[str, pd.DataFrame] | None = None

    def get_session(self) -> requests.Session:
        if self.requests_session is None:
            self.requests_session = requests.Session()
        return self.requests_session



def log(message: str) -> None:
    print(f"[01_inventory] {message}", flush=True)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_code(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        text = text[:-2]
    return text


def normalize_species(value: Any) -> str:
    return str(value).strip().upper()


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str], default: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)
    return out


# 単位整合性
@dataclass(frozen=True, slots=True)
class UnitRule:
    column: str
    source_unit: str
    internal_unit: str
    description: str = ""


_UNIT_ALIASES = {
    "": "",
    "1": "1",
    "ratio": "1",
    "fraction": "1",
    "%": "%",
    "percent": "%",
    "kg": "kg",
    "kg/y": "kg/year",
    "kg/yr": "kg/year",
    "kg/year": "kg/year",
    "kg person-1 year-1": "kg/person/year",
    "kg/person/year": "kg/person/year",
    "kg/cap/year": "kg/person/year",
    "kg/人/年": "kg/person/year",
    "a": "a",
    "are": "a",
    "ares": "a",
    "ha": "ha",
    "m2": "m2",
    "kg/a": "kg/a",
    "kg/are": "kg/a",
    "kg/ha": "kg/ha",
    "l/a": "L/a",
    "L/a": "L/a",
    "l/are": "L/a",
    "L/are": "L/a",
    "l/ha": "L/ha",
    "L/ha": "L/ha",
    "kwh/a": "kWh/a",
    "kWh/a": "kWh/a",
    "kwh/are": "kWh/a",
    "kWh/are": "kWh/a",
    "kwh/ha": "kWh/ha",
    "kWh/ha": "kWh/ha",
    "℃": "degC",
    "°C": "degC",
    "degC": "degC",
    "c": "degC",
    "mm": "mm",
    "m/s": "m/s",
    "cmolc/kg": "cmolc/kg",
    "cmol(+)/kg": "cmolc/kg",
    "mmolc/kg": "mmolc/kg",
    "mmol(+)/kg": "mmolc/kg",
    "mg/L": "mg/L",
    "mg/l": "mg/L",
    "t": "t",
    "tonne": "t",
    "tonnes": "t",
    "Mg": "t",
    "kg/kg": "1",
}


def _norm_unit(unit: Any) -> str:
    if pd.isna(unit):
        return ""
    text = str(unit).strip()
    return _UNIT_ALIASES.get(text, _UNIT_ALIASES.get(text.lower(), text))


_UNIT_FACTORS: dict[tuple[str, str], float] = {
    ("1", "1"): 1.0,
    ("%", "1"): 0.01,
    ("1", "%"): 100.0,

    ("kg", "kg"): 1.0,
    ("t", "kg"): 1000.0,
    ("kg", "t"): 0.001,
    ("kg/year", "kg/year"): 1.0,
    ("t/year", "kg/year"): 1000.0,
    ("kg/person/year", "kg/person/year"): 1.0,

    ("a", "a"): 1.0,
    ("ha", "a"): 100.0,
    ("m2", "a"): 0.01,
    ("a", "ha"): 0.01,
    ("ha", "ha"): 1.0,

    ("kg/a", "kg/a"): 1.0,
    ("kg/ha", "kg/a"): 0.01,
    ("L/a", "L/a"): 1.0,
    ("L/ha", "L/a"): 0.01,
    ("kWh/a", "kWh/a"): 1.0,
    ("kWh/ha", "kWh/a"): 0.01,

    ("degC", "degC"): 1.0,
    ("mm", "mm"): 1.0,
    ("m/s", "m/s"): 1.0,
    ("cmolc/kg", "cmolc/kg"): 1.0,
    # mmolc/kgにCECを変換
    ("mmolc/kg", "mmolc/kg"): 1.0,
    ("cmolc/kg", "mmolc/kg"): 10.0,
    ("mg/L", "mg/L"): 1.0,
}


INPUT_UNIT_SCHEMA: dict[str, list[UnitRule]] = {
    "00_cropland.csv": [
        UnitRule("EXTENT", "a", "a", "crop area; script later multiplies by 0.01 to obtain ha"),
    ],
    "00_agroinput.csv": [
        UnitRule("NIT", "kg/a", "kg/a", "synthetic N input intensity"),
        UnitRule("PHO", "kg/a", "kg/a", "synthetic P2O5 input intensity"),
        UnitRule("KAL", "kg/a", "kg/a", "synthetic K2O input intensity"),
        UnitRule("N_REQ", "kg/a", "kg/a", "crop N demand"),
        UnitRule("AGROCHEM", "kg/a", "kg/a", "agrochemical input intensity"),
        UnitRule("DIESEL", "L/a", "L/a", "diesel input intensity"),
        UnitRule("GASOL", "L/a", "L/a", "gasoline input intensity"),
        UnitRule("KEROS", "L/a", "L/a", "kerosene input intensity"),
        UnitRule("ELEC", "kWh/a", "kWh/a", "electricity input intensity"),
    ],
    "00_biowaste.csv": [
        UnitRule("WATER", "1", "1", "wet-basis fraction"),
        UnitRule("VS", "1", "1", "volatile solids fraction"),
        UnitRule("VS_REMOVE", "1", "1", "fraction removed"),
        UnitRule("CAR", "1", "1", "C fraction"),
        UnitRule("HYD", "1", "1", "H fraction"),
        UnitRule("OXY", "1", "1", "O fraction"),
        UnitRule("NIT", "1", "1", "N fraction"),
        UnitRule("PHO", "1", "1", "P/P2O5-related fraction used by current module"),
        UnitRule("KAL", "1", "1", "K/K2O-related fraction used by current module"),
    ],
    "00_manurefac.csv": [
        UnitRule("CAPACITY", "kg/year", "kg/year", "annual facility input capacity"),
        UnitRule("COMPOST_N", "1", "1", "N concentration in compost"),
        UnitRule("COMPOST_P", "1", "1", "P2O5 concentration in compost"),
        UnitRule("COMPOST_K", "1", "1", "K2O concentration in compost"),
    ],
    "00_mswfac.csv": [
        UnitRule("CAPACITY", "kg/year", "kg/year", "annual facility input capacity"),
        UnitRule("GEN_EFF", "1", "1", "electric generation efficiency"),
        UnitRule("ASH_RATIO", "1", "1", "ash fraction"),
    ],
    "00_wastegen.csv": [
        UnitRule("UNIT_WASTE", "kg/person/year", "kg/person/year", "municipal waste generation per capita"),
        UnitRule("FW_RATIO", "1", "1", "food waste fraction"),
        UnitRule("PAPER_RATIO", "1", "1", "paper fraction"),
        UnitRule("PLASTIC_RATIO", "1", "1", "plastic fraction"),
        UnitRule("SYNTEX_RATIO", "1", "1", "synthetic textile fraction"),
        UnitRule("NATTEX_RATIO", "1", "1", "natural textile fraction"),
        UnitRule("WOOD_RATIO", "1", "1", "wood fraction"),
    ],
    "00_env.csv": [
        UnitRule("AIRTEMP", "degC", "degC", "annual mean air temperature"),
        UnitRule("PRECIP", "mm", "mm", "precipitation"),
        UnitRule("WIND", "m/s", "m/s", "wind speed"),
        UnitRule("SOILTEMP", "degC", "degC", "soil temperature"),
        UnitRule("SAND", "1", "1", "sand fraction"),
        UnitRule("CEC", "mmolc/kg", "mmolc/kg", "cation exchange capacity; script divides by 10"),
        UnitRule("TN", "mg/L", "mg/L", "irrigation/water total nitrogen concentration"),
        UnitRule("TP", "mg/L", "mg/L", "total phosphorus concentration"),
    ],
    "00_livestock.csv": [
        UnitRule("NUMBER", "1", "1", "number of animals"),
    ],
    "00_population.csv": [
        UnitRule("POPULATION", "1", "1", "number of people"),
    ],
}


def convert_units(values: Any, from_unit: str, to_unit: str) -> Any:
    """Convert numeric value(s) between supported units."""
    fu = _norm_unit(from_unit)
    tu = _norm_unit(to_unit)
    if fu == "":
        fu = tu
    factor = _UNIT_FACTORS.get((fu, tu))
    if factor is None:
        raise ValueError(f"Unsupported unit conversion: {from_unit!r} -> {to_unit!r}")
    return pd.to_numeric(values, errors="coerce") * factor


def _unit_series_for_column(df: pd.DataFrame, column: str, default_unit: str) -> pd.Series:
    for unit_col in (f"{column}_UNIT", f"UNIT_{column}"):
        if unit_col in df.columns:
            return df[unit_col].map(_norm_unit).replace("", _norm_unit(default_unit))
    if "UNIT" in df.columns:
        units = df["UNIT"].map(_norm_unit).dropna()
        unique = [u for u in units.unique().tolist() if u != ""]
        if len(unique) <= 1:
            return df["UNIT"].map(_norm_unit).replace("", _norm_unit(default_unit))
    return pd.Series([_norm_unit(default_unit)] * len(df), index=df.index)


def apply_inventory_unit_schema(
    df: pd.DataFrame,
    dataset_name: str,
    *,
    strict: bool = True,
    add_unit_metadata: bool = True,
) -> pd.DataFrame:
    """Return a copy of df converted to 01_inventory.py internal units.

    The function is intentionally conservative:
    - If no unit columns are present, current dataset units are assumed.
    - If unit columns are present, values are converted row-by-row.
    - Unknown conversions raise an error when strict=True.
    """
    out = df.copy()
    rules = INPUT_UNIT_SCHEMA.get(dataset_name, [])
    for rule in rules:
        if rule.column not in out.columns:
            continue
        target = _norm_unit(rule.internal_unit)
        source_units = _unit_series_for_column(out, rule.column, rule.source_unit)
        values = pd.to_numeric(out[rule.column], errors="coerce")
        converted = pd.Series(index=out.index, dtype="float64")
        for unit in source_units.fillna(_norm_unit(rule.source_unit)).unique():
            unit_norm = _norm_unit(unit)
            mask = source_units.eq(unit)
            try:
                converted.loc[mask] = convert_units(values.loc[mask], unit_norm, target)
            except ValueError:
                if strict:
                    raise ValueError(
                        f"{dataset_name}.{rule.column}: cannot convert unit {unit_norm!r} "
                        f"to internal unit {target!r}. Add it to _UNIT_FACTORS."
                    )
                converted.loc[mask] = values.loc[mask]
        out[rule.column] = converted
        if add_unit_metadata:
            out[f"{rule.column}__UNIT_INTERNAL"] = target
    out.attrs.setdefault("inventory_internal_units", {})
    out.attrs["inventory_internal_units"].update({
        rule.column: _norm_unit(rule.internal_unit)
        for rule in rules
        if rule.column in out.columns
    })
    return out


def validate_inventory_units(df: pd.DataFrame, dataset_name: str) -> None:
    """Lightweight range checks after unit conversion."""
    rules = INPUT_UNIT_SCHEMA.get(dataset_name, [])
    for rule in rules:
        if rule.column not in df.columns:
            continue
        vals = pd.to_numeric(df[rule.column], errors="coerce")
        if vals.dropna().empty:
            continue
        if rule.internal_unit == "1" and rule.column not in {"NUMBER", "POPULATION"}:
            bad = vals[(vals < -1e-12) | (vals > 1.0 + 1e-12)]
            if not bad.empty:
                raise ValueError(
                    f"{dataset_name}.{rule.column} is expected as a fraction [0,1] "
                    f"after conversion, but observed range {vals.min()}–{vals.max()}."
                )
        if rule.internal_unit in {"kg/year", "kg/person/year", "a", "kg/a", "L/a", "kWh/a"}:
            bad = vals[vals < -1e-12]
            if not bad.empty:
                raise ValueError(
                    f"{dataset_name}.{rule.column} should be non-negative after conversion, "
                    f"but observed minimum {vals.min()}."
                )


def standardize_inventory_input(df: pd.DataFrame, dataset_name: str, *, strict: bool = True) -> pd.DataFrame:
    out = apply_inventory_unit_schema(df, dataset_name, strict=strict)
    validate_inventory_units(out, dataset_name)
    return out


def finalize_output_schema(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[columns]


def clean_small_numeric(df: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        vals = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        out[col] = np.where(np.abs(vals) < tol, 0.0, vals)
    return out


def zero_mwad_compost_residue_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with compost/residue fields forced to zero for MWAD rows.

    MWAD produces digestate/biogas, not compost, and rice residue is a
    composting auxiliary material. This sanitizer prevents mixed
    facility-year rows from reporting COMPOST_* or RESIDUE_* under TYPE=MWAD.
    """
    out = df.copy()
    if out.empty or "TYPE" not in out.columns:
        return out
    mwad_mask = out["TYPE"].astype(str).str.upper().str.strip().eq("MWAD")
    if not mwad_mask.any():
        return out
    zero_cols = [
        "RESIDUE_REQ_KG", "RESIDUE_SUPPLY_KG", "RESIDUE_IMPORT_KG",
        "RESIDUE_SUPPLIED_KM", "RESIDUE_IMPORT_KM",
        "COMPOST_KG", "COMPOST_USED_KG", "COMPOST_N", "COMPOST_P2O5",
        "COMPOST_P_KG", "COMPOST_K_KG", "COMPOST_N_KG",
    ]
    for col in zero_cols:
        if col in out.columns:
            out.loc[mwad_mask, col] = 0.0
    return out


MANURE_SPECIES_KG_COLUMNS = [
    "DAIRY_MANURE_KG",
    "CATTLE_MANURE_KG",
    "SWINE_MANURE_KG",
    "CHICKEN_MANURE_KG",
    "BROILER_MANURE_KG",
]



def add_incremental_digest_machinery_units(
    crop_df: pd.DataFrame,
    *,
    unit_capacity_t_per_year: float = 2500.0,
    grouping_cols: tuple[str, ...] = ("CITY", "CROP"),
) -> pd.DataFrame:
    """Add annual required and newly purchased digestate-application machinery units.

    DIGEST_MACHINERY_UNITS is the annual DAV2500-equivalent stock needed to
    handle applied liquid digestate. DIGEST_MACHINERY_NEW_UNITS is only the
    positive increment in stock relative to the previous maximum stock for the
    same CITY x CROP group, so machinery burden/cost is counted only when
    additional machinery is purchased.
    """
    out = crop_df.copy()
    if out.empty:
        out["DIGEST_MACHINERY_UNITS"] = pd.Series(dtype=float)
        out["DIGEST_MACHINERY_NEW_UNITS"] = pd.Series(dtype=float)
        return out

    for col in ["CITY", "YEAR", "CROP", "DIGEST_KG"]:
        if col not in out.columns:
            out[col] = 0.0 if col != "CROP" else ""
    out["YEAR"] = pd.to_numeric(out["YEAR"], errors="coerce").fillna(0).astype(int)
    out["CITY"] = pd.to_numeric(out["CITY"], errors="coerce").fillna(-1).astype(int)
    out["CROP"] = out["CROP"].astype(str).str.upper().str.strip()
    out["DIGEST_KG"] = pd.to_numeric(out["DIGEST_KG"], errors="coerce").fillna(0.0)

    annual_units = pd.to_numeric(out.get("DIGEST_MACHINERY_UNITS", 0.0), errors="coerce").fillna(0.0)
    missing_or_zero = ("DIGEST_MACHINERY_UNITS" not in crop_df.columns) or np.isclose(float(annual_units.sum()), 0.0)
    if missing_or_zero:
        annual_units = pd.Series(0.0, index=out.index)
        rice_mask = out["CROP"].eq("RICE")
        annual_units.loc[rice_mask] = out.loc[rice_mask, "DIGEST_KG"] * 0.001 / float(unit_capacity_t_per_year)
    out["DIGEST_MACHINERY_UNITS"] = annual_units.clip(lower=0.0)

    existing_new = pd.to_numeric(out.get("DIGEST_MACHINERY_NEW_UNITS", 0.0), errors="coerce").fillna(0.0)
    if "DIGEST_MACHINERY_NEW_UNITS" in crop_df.columns and not np.isclose(float(existing_new.sum()), 0.0):
        out["DIGEST_MACHINERY_NEW_UNITS"] = existing_new.clip(lower=0.0)
        return out

    out["DIGEST_MACHINERY_NEW_UNITS"] = 0.0
    valid_group_cols = [c for c in grouping_cols if c in out.columns] or ["CROP"]
    totals = (
        out.groupby(valid_group_cols + ["YEAR"], dropna=False, as_index=False)["DIGEST_MACHINERY_UNITS"]
        .sum()
        .sort_values(valid_group_cols + ["YEAR"])
    )
    totals["DIGEST_MACHINERY_NEW_TOTAL"] = 0.0
    for _, g in totals.groupby(valid_group_cols, dropna=False, sort=False):
        max_stock = 0.0
        for idx, row in g.sort_values("YEAR").iterrows():
            required = max(float(row["DIGEST_MACHINERY_UNITS"]), 0.0)
            new_units = max(required - max_stock, 0.0)
            totals.at[idx, "DIGEST_MACHINERY_NEW_TOTAL"] = new_units
            max_stock = max(max_stock, required)

    out = out.merge(
        totals[valid_group_cols + ["YEAR", "DIGEST_MACHINERY_UNITS", "DIGEST_MACHINERY_NEW_TOTAL"]].rename(
            columns={"DIGEST_MACHINERY_UNITS": "_DIGEST_MACHINERY_GROUP_UNITS"}
        ),
        on=valid_group_cols + ["YEAR"],
        how="left",
    )
    denom = pd.to_numeric(out["_DIGEST_MACHINERY_GROUP_UNITS"], errors="coerce").fillna(0.0)
    new_total = pd.to_numeric(out["DIGEST_MACHINERY_NEW_TOTAL"], errors="coerce").fillna(0.0)
    share = np.where(denom > 0, out["DIGEST_MACHINERY_UNITS"] / denom, 0.0)
    out["DIGEST_MACHINERY_NEW_UNITS"] = new_total * share
    out = out.drop(columns=["_DIGEST_MACHINERY_GROUP_UNITS", "DIGEST_MACHINERY_NEW_TOTAL"], errors="ignore")
    return out


def enforce_nonnegative(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = np.maximum(pd.to_numeric(out[col], errors="coerce").fillna(0.0), 0.0)
    return out


def enforce_component_sum_limit(
    df: pd.DataFrame,
    component_cols: list[str],
    limit_col: str,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    out = df.copy()
    if limit_col not in out.columns:
        return out
    existing = [c for c in component_cols if c in out.columns]
    if not existing:
        return out
    limit = pd.to_numeric(out[limit_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    comps = out[existing].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sums = comps.sum(axis=1)
    mask = sums > (limit + tolerance)
    if np.any(mask):
        scale = np.ones(len(out), dtype=float)
        scale[mask] = np.where(sums[mask] > 0, limit[mask] / sums[mask], 0.0)
        comps = comps * scale[:, None]
        for j, col in enumerate(existing):
            out[col] = comps[:, j]
    return out


def enforce_species_manure_mass_balance(df: pd.DataFrame, total_col: str = "MANURE_KG") -> pd.DataFrame:
    out = df.copy()
    existing = [c for c in MANURE_SPECIES_KG_COLUMNS if c in out.columns]
    if total_col not in out.columns or not existing:
        return out
    out = enforce_component_sum_limit(out, existing, total_col)
    total = pd.to_numeric(out[total_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    comps = out[existing].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sums = comps.sum(axis=1)
    short = (total - sums) > 1e-6
    if np.any(short):
        gap = total - sums
        first = existing[0]
        out.loc[short, first] = pd.to_numeric(out.loc[short, first], errors="coerce").fillna(0.0) + gap[short]
    return out


def compute_species_manure_kg(species_kg: dict[str, float]) -> dict[str, float]:
    return {
        "DAIRY_MANURE_KG": float(species_kg.get("DAIRY", 0.0)),
        "CATTLE_MANURE_KG": float(species_kg.get("CATTLE", 0.0)),
        "SWINE_MANURE_KG": float(species_kg.get("SWINE", 0.0)),
        "CHICKEN_MANURE_KG": float(species_kg.get("CHICKEN", 0.0)),
        "BROILER_MANURE_KG": float(species_kg.get("BROILER", 0.0)),
    }


def cap_n_loss_pathways(nh3: pd.Series, no: pd.Series, nleach: pd.Series, available_n: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    nh3_s = pd.to_numeric(nh3, errors="coerce").fillna(0.0)
    no_s = pd.to_numeric(no, errors="coerce").fillna(0.0)
    nleach_s = pd.to_numeric(nleach, errors="coerce").fillna(0.0)
    avail_s = pd.to_numeric(available_n, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = nh3_s + no_s + nleach_s
    scale = pd.Series(np.where(total > avail_s, np.where(total > 0, avail_s / total, 0.0), 1.0), index=nh3_s.index)
    return nh3_s * scale, no_s * scale, nleach_s * scale


def is_alt_scenario(name: Any) -> bool:
    return str(name).strip().upper() == "ALT"


def event_from_years(rebuild_year: Any, renov_year: Any = np.nan) -> str:
    """Return facility event label from rebuild/renovation years.

    Used when exporting 05_param_facility.csv for the optimizer.
    Priority follows the optimizer-side helper: REBUILD first, then RENEW.
    """
    ry = pd.to_numeric(rebuild_year, errors="coerce")
    ny = pd.to_numeric(renov_year, errors="coerce")
    if pd.notna(ry) and np.isfinite(float(ry)) and 2030 <= int(float(ry)) <= 2050:
        return "REBUILD"
    if pd.notna(ny) and np.isfinite(float(ny)) and 2030 <= int(float(ny)) <= 2050:
        return "RENEW"
    return ""


def load_csv(path: Path, dtype: dict[str, Any] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=dtype)


def load_shapefile(path: Path, default_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")
    gdf = gpd.read_file(path)
    if gdf.crs is None and default_crs:
        gdf = gdf.set_crs(default_crs)
    return gdf


def load_rcom_metadata(paths: InventoryPaths) -> pd.DataFrame:
    df = load_csv(paths.rcom_csv, dtype={"KEY": str})
    df["KEY"] = df["KEY"].map(normalize_code)
    for col in ["PREF", "CITY", "KCITY", "RCOM"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_key_centroids(paths: InventoryPaths) -> pd.DataFrame:
    rcom_csv = load_rcom_metadata(paths)
    rcom_csv_rcom = rcom_csv[rcom_csv["RCOM"].fillna(0).astype(int) != 0].reset_index(drop=True)
    rcom_gdf = load_shapefile(paths.rcom_shp).reset_index(drop=True)

    if len(rcom_csv_rcom) != len(rcom_gdf):
        raise ValueError(
            "00_rcom.csv rows with RCOM != 0 and 00_rcom.shp row counts do not match. "
            "Current implementation assumes aligned row order for agricultural settlement polygons."
        )

    centroid_geom = rcom_gdf.to_crs("EPSG:3857").geometry.centroid.to_crs("EPSG:4326")
    out = rcom_csv_rcom[["KEY", "PREF", "CITY", "KCITY", "RCOM", "CITY_NAME", "KCITY_NAME"]].copy()
    out["CENT_LON"] = centroid_geom.x.values
    out["CENT_LAT"] = centroid_geom.y.values
    return out


def build_rcom_polygons_with_city(paths: InventoryPaths) -> gpd.GeoDataFrame:
    rcom_csv = load_rcom_metadata(paths)
    rcom_csv_rcom = rcom_csv[rcom_csv["RCOM"].fillna(0).astype(int) != 0].reset_index(drop=True)
    rcom_gdf = load_shapefile(paths.rcom_shp).to_crs("EPSG:4326").reset_index(drop=True)
    if len(rcom_csv_rcom) != len(rcom_gdf):
        raise ValueError(
            "00_rcom.csv rows with RCOM != 0 and 00_rcom.shp row counts do not match. "
            "Current implementation assumes aligned row order for agricultural settlement polygons."
        )
    keep_cols = [col for col in ["KEY", "CITY", "KCITY", "RCOM", "CITY_NAME", "KCITY_NAME"] if col in rcom_csv_rcom.columns]
    out = rcom_gdf.copy()
    for col in keep_cols:
        out[col] = rcom_csv_rcom[col].values
    return out


def load_offices(paths: InventoryPaths) -> gpd.GeoDataFrame:
    gdf = load_shapefile(paths.offices_shp).to_crs("EPSG:4326")
    gdf = gdf.copy()
    gdf["OFFICE_ID"] = gdf.index.astype(str)
    centroid_geom = gdf.to_crs("EPSG:3857").geometry.centroid.to_crs("EPSG:4326")
    gdf["LON"] = centroid_geom.x
    gdf["LAT"] = centroid_geom.y
    return gdf


def build_office_table(paths: InventoryPaths) -> pd.DataFrame:
    offices = load_offices(paths)
    rcom_poly = build_rcom_polygons_with_city(paths)
    poly_cols = [col for col in ["CITY", "KCITY", "RCOM", "CITY_NAME", "KCITY_NAME"] if col in rcom_poly.columns]
    joined = gpd.sjoin(
        offices,
        rcom_poly[poly_cols + ["geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    if "CITY" in joined.columns:
        miss = joined["CITY"].isna()
        if miss.any():
            office_m = joined.loc[miss, ["geometry"]].to_crs("EPSG:3857")
            rcom_m = rcom_poly[poly_cols + ["geometry"]].to_crs("EPSG:3857")
            nearest_idx = office_m.geometry.apply(lambda g: rcom_m.distance(g).idxmin())
            nearest_attrs = rcom_m.loc[nearest_idx, poly_cols].reset_index(drop=True)
            for col in poly_cols:
                joined.loc[miss, col] = nearest_attrs[col].values
    keep_cols = [col for col in ["OFFICE_ID", "CITY", "CITY_NAME", "LON", "LAT"] if col in joined.columns]
    out = pd.DataFrame(joined[keep_cols]).copy()
    if "CITY" in out.columns:
        out["CITY"] = pd.to_numeric(out["CITY"], errors="coerce")
    return out


def build_city_office_lookup(offices: pd.DataFrame) -> dict[int, tuple[float, float]]:
    if offices.empty or "CITY" not in offices.columns:
        return {}
    off = offices.copy()
    off["CITY"] = pd.to_numeric(off["CITY"], errors="coerce")
    off["LON"] = pd.to_numeric(off["LON"], errors="coerce")
    off["LAT"] = pd.to_numeric(off["LAT"], errors="coerce")
    off = off.dropna(subset=["CITY", "LON", "LAT"]).copy()
    if off.empty:
        return {}
    off["CITY"] = off["CITY"].astype(int)
    city_off = off.groupby("CITY", as_index=False)[["LON", "LAT"]].mean()
    return {int(r["CITY"]): (float(r["LON"]), float(r["LAT"])) for _, r in city_off.iterrows()}


def nearest_office_distance_plus_300(
    ctx: InventoryContext,
    lon: float,
    lat: float,
    offices: pd.DataFrame,
    origin_id: str,
) -> float:
    if offices.empty or (not np.isfinite(lon)) or (not np.isfinite(lat)):
        return 300.0
    best = np.inf
    for _, off in offices.iterrows():
        try:
            km = get_distance_km(
                ctx,
                origin_id,
                float(lon),
                float(lat),
                f"OFFICE:{off.get('OFFICE_ID', off.name)}",
                float(off["LON"]),
                float(off["LAT"]),
            )
        except Exception:
            continue
        if np.isfinite(km) and km < best:
            best = km
    return float((best if np.isfinite(best) else 0.0) + 300.0)


def get_city_relay_distance_km(
    ctx: InventoryContext,
    origin_id: str,
    origin_lon: float,
    origin_lat: float,
    origin_city: Any,
    dest_id: str,
    dest_lon: float,
    dest_lat: float,
    dest_city: Any,
    city_office_map: dict[int, tuple[float, float]],
) -> float:
    try:
        oc = int(origin_city) if pd.notna(origin_city) else None
    except Exception:
        oc = None
    try:
        dc = int(dest_city) if pd.notna(dest_city) else None
    except Exception:
        dc = None

    if oc is None or dc is None or oc == dc:
        return get_distance_km(ctx, origin_id, origin_lon, origin_lat, dest_id, dest_lon, dest_lat)

    o_off = city_office_map.get(oc)
    d_off = city_office_map.get(dc)
    if o_off is None or d_off is None:
        return get_distance_km(ctx, origin_id, origin_lon, origin_lat, dest_id, dest_lon, dest_lat)

    d1 = get_distance_km(ctx, origin_id, origin_lon, origin_lat, f"OFFICE_CITY:{oc}", o_off[0], o_off[1])
    d2 = get_distance_km(ctx, f"OFFICE_CITY:{oc}", o_off[0], o_off[1], f"OFFICE_CITY:{dc}", d_off[0], d_off[1])
    d3 = get_distance_km(ctx, f"OFFICE_CITY:{dc}", d_off[0], d_off[1], dest_id, dest_lon, dest_lat)
    return float(d1 + d2 + d3)


def validate_required_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def linear_regression_projection(x: np.ndarray, y: np.ndarray, years: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    years = np.asarray(years, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return np.full_like(years, np.nan, dtype=float)
    if len(x) == 1:
        return np.full_like(years, float(y[0]), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return intercept + slope * years


def estimate_exp_r(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr) & (y_arr > 0)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if len(np.unique(x_arr)) < 2 or len(y_arr) < 2:
        return 0.0
    r, _ = np.polyfit(x_arr, np.log(y_arr), 1)
    return float(np.clip(r, R_MIN, R_MAX))


def project_exponential(base_value: float, r: float, years: Iterable[int], base_year: int = 2023) -> np.ndarray:
    years_arr = np.asarray(list(years), dtype=float)
    if (not np.isfinite(base_value)) or (base_value <= 0):
        return np.zeros(len(years_arr), dtype=int)
    vals = base_value * np.exp(r * (years_arr - float(base_year)))
    vals = np.where(np.isfinite(vals), vals, 0.0)
    vals = np.where(vals < 0, 0.0, vals)
    return np.round(vals).astype(int)


def cache_key(*parts: Any) -> str:
    return "|".join(map(str, parts))


def load_distance_cache(cache_path: Path) -> dict[str, float]:
    if not cache_path.exists():
        return {}
    with cache_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): float(v) for k, v in raw.items()}


def save_distance_cache(cache: dict[str, float], cache_path: Path) -> None:
    ensure_dir(cache_path.parent)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_km = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def osrm_road_km(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
    session: requests.Session,
    osrm_url: str = DEFAULT_OSRM_URL,
    timeout_sec: int = 30,
) -> float:
    url = f"{osrm_url}/{lon1},{lat1};{lon2},{lat2}"
    params = {"overview": "false", "steps": "false", "annotations": "false"}
    resp = session.get(url, params=params, timeout=timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    routes = data.get("routes", [])
    if not routes:
        raise ValueError("OSRM returned no routes")
    return float(routes[0]["distance"]) / 1000.0


def get_distance_km(
    ctx: InventoryContext,
    origin_id: str,
    origin_lon: float,
    origin_lat: float,
    dest_id: str,
    dest_lon: float,
    dest_lat: float,
) -> float:
    key = cache_key(origin_id, dest_id)
    if key in ctx.distance_cache and np.isfinite(ctx.distance_cache[key]):
        return float(ctx.distance_cache[key])
    if ctx.scenario.use_osrm:
        try:
            dist = osrm_road_km(
                origin_lon,
                origin_lat,
                dest_lon,
                dest_lat,
                session=ctx.get_session(),
                osrm_url=ctx.osrm_url,
                timeout_sec=ctx.timeout_sec,
            )
        except Exception:
            dist = haversine_km(origin_lon, origin_lat, dest_lon, dest_lat)
    else:
        dist = haversine_km(origin_lon, origin_lat, dest_lon, dest_lat)
    ctx.distance_cache[key] = float(dist)
    return float(dist)


def safe_div(num: Any, den: Any, default: float = 0.0):
    num_arr = np.asarray(num, dtype=float)
    den_arr = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num_arr, den_arr).shape, default, dtype=float)
    mask = np.isfinite(num_arr) & np.isfinite(den_arr) & (den_arr != 0)
    out[mask] = num_arr[mask] / den_arr[mask]
    return out


def empty_crop_output() -> pd.DataFrame:
    return pd.DataFrame(columns=CROP_OUTPUT_COLUMNS)


def build_shared_inputs(ctx: InventoryContext) -> dict[str, pd.DataFrame]:
    if ctx.shared_inputs is not None:
        return ctx.shared_inputs
    rcom = load_rcom_metadata(ctx.paths)
    validate_required_columns(
        rcom,
        [
            "KEY", "PREF", "CITY", "KCITY", "RCOM", "RICE_RATIO", "TEA_RATIO", "VEG_RATIO",
            "DAIRY_RATIO", "CATTLE_RATIO", "SWINE_RATIO", "CHICKEN_RATIO", "BROILER_RATIO",
        ],
        "00_rcom.csv",
    )
    manurefac = standardize_inventory_input(load_csv(ctx.paths.manurefac_csv), "00_manurefac.csv")
    validate_required_columns(
        manurefac,
        ["FACILITY", "LAT", "LON", "TYPE", "SPECIES", "CITY", "CAPACITY", "COMPOST_N", "COMPOST_P", "COMPOST_K"],
        "00_manurefac.csv",
    )
    livestock = standardize_inventory_input(load_csv(ctx.paths.livestock_csv, dtype={"KEY": str}), "00_livestock.csv")
    validate_required_columns(livestock, ["KEY", "CITY", "RCOM", "SPECIES", "YEAR", "NUMBER"], "00_livestock.csv")
    cropland = standardize_inventory_input(load_csv(ctx.paths.cropland_csv, dtype={"KEY": str}), "00_cropland.csv")
    validate_required_columns(cropland, ["KEY", "CITY", "RCOM", "CROP", "CULTIVAR", "YEAR", "EXTENT"], "00_cropland.csv")
    agroinput = standardize_inventory_input(load_csv(ctx.paths.agroinput_csv), "00_agroinput.csv")
    validate_required_columns(
        agroinput,
        ["CITY", "CROP", "CULTIVAR", "YEAR", "NIT", "PHO", "KAL", "N_REQ", "AGROCHEM", "DIESEL", "GASOL", "KEROS", "ELEC"],
        "00_agroinput.csv",
    )
    biowaste = standardize_inventory_input(load_csv(ctx.paths.biowaste_csv), "00_biowaste.csv")
    validate_required_columns(
        biowaste,
        ["BIOWASTE", "WATER", "VS", "VS_REMOVE", "CAR", "HYD", "OXY", "NIT", "PHO", "KAL"],
        "00_biowaste.csv",
    )
    env = standardize_inventory_input(load_csv(ctx.paths.env_csv, dtype={"KEY": str}), "00_env.csv")
    validate_required_columns(
        env,
        ["KEY", "CITY", "RCOM", "AIRTEMP", "WIND", *NH3_ENV_MONTH_COLS, "PRECIP", "TN", "SOILTEMP", "pH", "CEC", "SAND"],
        "00_env.csv",
    )
    wastegen = standardize_inventory_input(load_csv(ctx.paths.wastegen_csv), "00_wastegen.csv")
    validate_required_columns(
        wastegen,
        ["CITY_NAME", "YEAR", "UNIT_WASTE", "FW_RATIO", "PLASTIC_RATIO", "SYNTEX_RATIO", "PAPER_RATIO", "NATTEX_RATIO", "WOOD_RATIO"],
        "00_wastegen.csv",
    )
    population = standardize_inventory_input(load_csv(ctx.paths.population_csv, dtype={"KEY": str}), "00_population.csv")
    validate_required_columns(
        population,
        ["KEY", "PREF", "CITY", "KCITY", "RCOM", "CITY_NAME", "YEAR", "POPULATION"],
        "00_population.csv",
    )
    mswfac = standardize_inventory_input(load_csv(ctx.paths.mswfac_csv), "00_mswfac.csv")
    validate_required_columns(
        mswfac,
        ["FACILITY", "TYPE", "CAPACITY", "LAT", "LON"],
        "00_mswfac.csv",
    )
    key_centroids = build_key_centroids(ctx.paths)
    offices = build_office_table(ctx.paths)
    city_office_map = build_city_office_lookup(offices)
    ctx.shared_inputs = {
        "rcom": rcom,
        "manurefac": manurefac,
        "livestock": livestock,
        "cropland": cropland,
        "agroinput": agroinput,
        "biowaste": biowaste,
        "env": env,
        "wastegen": wastegen,
        "population": population,
        "mswfac": mswfac,
        "key_centroids": key_centroids,
        "offices": offices,
        "city_office_map": city_office_map,
    }
    ctx.shared_inputs = apply_facility_plan_to_inputs(ctx.shared_inputs, ctx.scenario.facility_plan_path)
    return ctx.shared_inputs


def _mwad_base_name_for_year_lookup(name: Any) -> str:
    text = str(name).strip()
    parts = text.split("_")
    while parts and (parts[-1].isdigit() or parts[-1] in {"R", "ALT", "OPT", "INC", "RDF", "TDAD", "MWAD", "COMPOST", "AD"}):
        parts = parts[:-1]
    return "_".join(parts)


def build_mwad_year_lookup_from_mswfac(mswfac: pd.DataFrame) -> tuple[dict[str, int], dict[int, int], dict[int, int]]:
    """Return MWAD_YEAR lookup by base, exact city, and boundary city.

    00_mswfac.csv::MWAD_YEAR is the source of truth for installed MWAD timing.
    Direct MSW-base matches have highest priority; manure-side/other candidates
    fall back to exact CITY and then BOUNDARY membership.
    """
    if mswfac.empty or "MWAD_YEAR" not in mswfac.columns:
        return {}, {}, {}
    df = mswfac.copy()
    df["BASE"] = df.get("FACILITY", "").astype(str).map(_mwad_base_name_for_year_lookup)
    df["MWAD_YEAR"] = pd.to_numeric(df["MWAD_YEAR"], errors="coerce")
    df["CITY"] = pd.to_numeric(df.get("CITY", np.nan), errors="coerce")
    df = df.dropna(subset=["MWAD_YEAR"]).copy()
    df = df[(df["MWAD_YEAR"] >= 2030) & (df["MWAD_YEAR"] <= 2050)].copy()
    if df.empty:
        return {}, {}, {}
    base_map = df.groupby("BASE")["MWAD_YEAR"].min().astype(int).to_dict()
    city_df = df.dropna(subset=["CITY"]).copy()
    city_df["CITY_INT"] = city_df["CITY"].astype(int)
    city_map = city_df.groupby("CITY_INT")["MWAD_YEAR"].min().astype(int).to_dict()
    boundary_map: dict[int, int] = {}
    for base, cities in BOUNDARY.items():
        year = base_map.get(str(base))
        if year is None:
            continue
        for city in cities:
            city_i = int(city)
            boundary_map[city_i] = min(boundary_map.get(city_i, int(year)), int(year))
    return {str(k): int(v) for k, v in base_map.items()}, {int(k): int(v) for k, v in city_map.items()}, boundary_map


def mwad_year_for_plan_row(row: pd.Series | dict[str, Any], base_map: dict[str, int], city_map: dict[int, int], boundary_map: dict[int, int]) -> int | None:
    base = _mwad_base_name_for_year_lookup(row.get("FACILITY", ""))
    if base in base_map:
        return int(base_map[base])
    try:
        city = int(float(row.get("CITY", np.nan)))
    except Exception:
        city = None
    if city is not None:
        if city in city_map:
            return int(city_map[city])
        if city in boundary_map:
            return int(boundary_map[city])
    return None

# 最適化された施設の入力
def apply_facility_plan_to_inputs(shared: dict[str, pd.DataFrame], facility_plan_path: Path | None) -> dict[str, pd.DataFrame]:
    if facility_plan_path is None:
        return shared
    plan_path = Path(facility_plan_path)
    if not plan_path.exists():
        log(f"Facility plan not found, skipping overlay: {plan_path}")
        return shared

    plan = pd.read_csv(plan_path)
    if plan.empty or "FACILITY" not in plan.columns or "TYPE" not in plan.columns:
        log(f"Facility plan is empty or missing FACILITY/TYPE, skipping overlay: {plan_path}")
        return shared

    plan = plan.copy()
    plan["FACILITY"] = plan["FACILITY"].astype(str)
    plan["TYPE"] = plan["TYPE"].astype(str).str.upper().str.strip()
    plan["SCALE_T_DAY"] = pd.to_numeric(plan.get("SCALE_T_DAY", 0.0), errors="coerce").fillna(0.0)
    plan["CAPACITY_OPT_KG"] = np.where(
        plan["TYPE"].isin(["MWAD", "TDAD"]),
        plan["SCALE_T_DAY"] * 1000.0 * 365.0,
        np.nan,
    )
    plan["BUILT_YEAR"] = pd.to_numeric(plan.get("BUILT_YEAR", 2030), errors="coerce").fillna(2030).astype(int)
    if "ENERGY_USE" not in plan.columns:
        plan["ENERGY_USE"] = ""
    plan["ENERGY_USE"] = plan["ENERGY_USE"].fillna("").astype(str).str.upper().str.strip()

    manurefac = shared.get("manurefac", pd.DataFrame()).copy()
    mswfac = shared.get("mswfac", pd.DataFrame()).copy()
    mwad_base_year_map, mwad_city_year_map, mwad_boundary_year_map = build_mwad_year_lookup_from_mswfac(mswfac)

    # Normalize/override MWAD plan timing from 00_mswfac.csv::MWAD_YEAR before overlay.
    if not plan.empty and "TYPE" in plan.columns:
        mwad_plan_mask = plan["TYPE"].astype(str).str.upper().str.strip().eq("MWAD")
        for idx in plan.index[mwad_plan_mask]:
            mwad_year = mwad_year_for_plan_row(plan.loc[idx], mwad_base_year_map, mwad_city_year_map, mwad_boundary_year_map)
            if mwad_year is not None:
                plan.at[idx, "BUILT_YEAR"] = int(mwad_year)

    if not manurefac.empty:
        manurefac["FACILITY"] = manurefac["FACILITY"].astype(str)
        manurefac["TYPE"] = manurefac["TYPE"].astype(str).str.upper().str.strip()
        for _, r in plan.iterrows():
            fac = str(r["FACILITY"])
            typ = str(r["TYPE"])
            source = str(r.get("SOURCE", "")).upper()
            if "MANUREFAC" not in source or typ not in {"COMPOST", "MWAD", "TDAD"}:
                continue
            mask_fac = manurefac["FACILITY"].eq(fac)
            if not mask_fac.any():
                continue

            if typ in {"MWAD", "TDAD"}:
                keep_mask = ~(mask_fac & manurefac["TYPE"].eq("COMPOST"))
                ref_rows = manurefac.loc[mask_fac].copy()
                ref = ref_rows.iloc[0].copy()
                manurefac = manurefac.loc[keep_mask].copy()

                mask_type = manurefac["FACILITY"].eq(fac) & manurefac["TYPE"].eq(typ)
                if not mask_type.any():
                    ref["TYPE"] = typ
                    manurefac = pd.concat([manurefac, pd.DataFrame([ref])], ignore_index=True)
                    mask_type = manurefac["FACILITY"].eq(fac) & manurefac["TYPE"].eq(typ)
                if np.isfinite(float(r.get("CAPACITY_OPT_KG", np.nan))):
                    manurefac.loc[mask_type, "CAPACITY"] = float(r["CAPACITY_OPT_KG"])
                manurefac.loc[mask_type, "CONST_YEAR"] = int(r["BUILT_YEAR"])
                if r.get("ENERGY_USE", ""):
                    manurefac.loc[mask_type, "ENERGY_USE"] = r.get("ENERGY_USE", "")
            else:
                mask_type = mask_fac & manurefac["TYPE"].eq("COMPOST")
                if mask_type.any() and np.isfinite(float(r.get("CAPACITY_OPT_KG", np.nan))):
                    manurefac.loc[mask_type, "CAPACITY"] = float(r["CAPACITY_OPT_KG"])

    if mswfac.empty:
        mswfac = pd.DataFrame(columns=["FACILITY", "TYPE", "CAPACITY", "LAT", "LON", "CITY", "CONST_YEAR", "ENERGY_USE"])
    mswfac["FACILITY"] = mswfac["FACILITY"].astype(str)
    for col in ["TYPE", "CAPACITY", "LAT", "LON", "CITY", "CONST_YEAR", "ENERGY_USE"]:
        if col not in mswfac.columns:
            mswfac[col] = np.nan if col not in {"TYPE", "ENERGY_USE"} else ""
    mswfac["TYPE"] = mswfac["TYPE"].astype(str).str.upper().str.strip()

    def unique_msw_unit_name(base_fac: str, typ: str) -> str:
        existing = set(mswfac["FACILITY"].astype(str).tolist())
        cand = f"{base_fac}_{typ}"
        if cand not in existing:
            return cand
        i = 2
        while f"{cand}_{i}" in existing:
            i += 1
        return f"{cand}_{i}"

    append_rows: list[dict[str, Any]] = []
    for _, r in plan.iterrows():
        fac = str(r["FACILITY"])
        typ = str(r["TYPE"])
        source = str(r.get("SOURCE", "")).upper()
        if typ not in {"INC", "RDF", "MWAD", "TDAD"}:
            continue
        cap = float(r["CAPACITY_OPT_KG"]) if np.isfinite(float(r.get("CAPACITY_OPT_KG", np.nan))) else np.nan

        if typ in {"MWAD", "TDAD"}:
            unit_fac = unique_msw_unit_name(fac, typ) if (mswfac["FACILITY"].eq(fac).any()) else fac
            append_rows.append({
                "FACILITY": unit_fac,
                "TYPE": typ,
                "CAPACITY": cap if np.isfinite(cap) else 0.0,
                "LAT": pd.to_numeric(r.get("LAT", np.nan), errors="coerce"),
                "LON": pd.to_numeric(r.get("LON", np.nan), errors="coerce"),
                "CITY": pd.to_numeric(r.get("CITY", np.nan), errors="coerce"),
                "CONST_YEAR": int(r["BUILT_YEAR"]),
                "REBUILD_YEAR": np.nan,
                "RENOV_YEAR": np.nan,
                "CAP_YEAR": np.nan,
                "GEN_EFF": np.nan,
                "ASH_RATIO": DEFAULT_ASH_RATIO,
                "ENERGY_USE": r.get("ENERGY_USE", ""),
            })
            continue

        mask = mswfac["FACILITY"].eq(fac)
        mask_same_type = mask & mswfac["TYPE"].eq(typ)
        if mask_same_type.any():
            if np.isfinite(cap):
                mswfac.loc[mask_same_type, "CAPACITY"] = cap
            mswfac.loc[mask_same_type, "CONST_YEAR"] = int(r["BUILT_YEAR"])
            if r.get("ENERGY_USE", ""):
                mswfac.loc[mask_same_type, "ENERGY_USE"] = r.get("ENERGY_USE", "")
        else:
            append_rows.append({
                "FACILITY": fac,
                "TYPE": typ,
                "CAPACITY": cap if np.isfinite(cap) else 0.0,
                "LAT": pd.to_numeric(r.get("LAT", np.nan), errors="coerce"),
                "LON": pd.to_numeric(r.get("LON", np.nan), errors="coerce"),
                "CITY": pd.to_numeric(r.get("CITY", np.nan), errors="coerce"),
                "CONST_YEAR": int(r["BUILT_YEAR"]),
                "REBUILD_YEAR": np.nan,
                "RENOV_YEAR": np.nan,
                "CAP_YEAR": np.nan,
                "GEN_EFF": np.nan,
                "ASH_RATIO": DEFAULT_ASH_RATIO,
                "ENERGY_USE": r.get("ENERGY_USE", ""),
            })
    if append_rows:
        mswfac = pd.concat([mswfac, pd.DataFrame(append_rows)], ignore_index=True)

    shared["manurefac"] = manurefac
    shared["mswfac"] = mswfac
    log(f"Applied facility plan overlay from {plan_path}")
    return shared


def build_biowaste_lookup(biowaste: pd.DataFrame) -> pd.DataFrame:
    out = biowaste.copy()
    out["BIOWASTE"] = out["BIOWASTE"].astype(str).str.upper().str.strip()
    for col in ["WATER", "VS", "VS_REMOVE", "CAR", "HYD", "OXY", "NIT", "PHO", "KAL"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.set_index("BIOWASTE")


def get_species_n_per_kg(bw: pd.DataFrame, species: str) -> float:
    species = normalize_species(species)
    if species in {"DAIRY", "CATTLE"}:
        f_row = bw.loc["CATTLE_F"]
        u_row = bw.loc["CATTLE_U"]
        rates = MANURE_UNIT_RATES[species]
        return float(rates["F"] * f_row["VS"] * f_row["NIT"] + rates["U"] * u_row["VS"] * u_row["NIT"])
    if species == "SWINE":
        f_row = bw.loc["SWINE_F"]
        u_row = bw.loc["SWINE_U"]
        rates = MANURE_UNIT_RATES[species]
        return float(rates["F"] * f_row["VS"] * f_row["NIT"] + rates["U"] * u_row["VS"] * u_row["NIT"])
    if species in {"CHICKEN", "BROILER"}:
        row = bw.loc["POULTRY_E"]
        rates = MANURE_UNIT_RATES[species]
        return float(rates["E"] * row["VS"] * row["NIT"])
    return 0.0


def project_city_species_livestock(livestock: pd.DataFrame, years: Iterable[int]) -> pd.DataFrame:
    lv = livestock.copy()
    lv["SPECIES"] = lv["SPECIES"].map(normalize_species)
    lv["YEAR"] = pd.to_numeric(lv["YEAR"], errors="coerce")
    lv["NUMBER"] = pd.to_numeric(lv["NUMBER"], errors="coerce")
    lv["CITY"] = pd.to_numeric(lv["CITY"], errors="coerce")
    lv = lv.dropna(subset=["YEAR", "NUMBER", "CITY", "SPECIES"])
    lv = lv[lv["CITY"] != 0].copy()
    rows: list[dict[str, Any]] = []
    for keys, grp in lv.groupby(["PREF", "CITY", "CITY_NAME", "SPECIES"], dropna=False):
        pref, city, city_name, species = keys
        grp = grp.sort_values("YEAR").copy()
        r = estimate_exp_r(grp["YEAR"], grp["NUMBER"])
        base = grp[grp["YEAR"] == 2023]
        if not base.empty:
            base_value = float(base["NUMBER"].iloc[-1])
        else:
            last_row = grp.iloc[-1]
            last_year = float(last_row["YEAR"])
            last_value = float(last_row["NUMBER"])
            base_value = float(last_value * np.exp(r * (2023 - last_year))) if last_value > 0 else 0.0
        projected = project_exponential(base_value, r, years, base_year=2023)
        for year, number in zip(years, projected, strict=False):
            rows.append({
                "PREF": pref,
                "CITY": int(city),
                "CITY_NAME": city_name,
                "SPECIES": species,
                "YEAR": int(year),
                "UNIT_PROJECTED": int(number),
            })
    return pd.DataFrame(rows)


def distribute_city_species_to_keys(city_proj: pd.DataFrame, rcom: pd.DataFrame) -> pd.DataFrame:
    rcom_rc = rcom.copy()
    rcom_rc["KEY"] = rcom_rc["KEY"].map(normalize_code)
    rcom_rc = rcom_rc[rcom_rc["RCOM"].fillna(0).astype(int) != 0].copy()
    rows: list[dict[str, Any]] = []
    base_cols = ["KEY", "PREF", "CITY", "KCITY", "RCOM", "CITY_NAME"]
    for _, row in city_proj.iterrows():
        species = normalize_species(row["SPECIES"])
        ratio_col = SPECIES_RATIO_MAP.get(species)
        sub = rcom_rc[rcom_rc["CITY"] == row["CITY"]].copy()
        if sub.empty:
            continue
        if ratio_col and ratio_col in sub.columns:
            sub_valid = sub[pd.to_numeric(sub[ratio_col], errors="coerce").fillna(0.0) > 0].copy()
            if not sub_valid.empty:
                sub = sub_valid
                ratios = pd.to_numeric(sub[ratio_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                ratios = ratios / ratios.sum() if ratios.sum() > 0 else np.ones(len(sub)) / len(sub)
            else:
                ratios = np.ones(len(sub), dtype=float) / float(len(sub))
        else:
            ratios = np.ones(len(sub), dtype=float) / float(len(sub))
        raw_alloc = float(row["UNIT_PROJECTED"]) * ratios
        ints = np.floor(raw_alloc).astype(int)
        remainder = int(round(float(row["UNIT_PROJECTED"]) - ints.sum()))
        if remainder > 0:
            frac_order = np.argsort(-(raw_alloc - ints))
            ints[frac_order[:remainder]] += 1
        alloc = sub[base_cols].copy().reset_index(drop=True)
        alloc["SPECIES"] = species
        alloc["YEAR"] = int(row["YEAR"])
        alloc["NUMBER"] = ints
        rows.extend(alloc.to_dict(orient="records"))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["NUMBER"] = pd.to_numeric(out["NUMBER"], errors="coerce").fillna(0).astype(int)
    return out


def build_manure_sources(livestock_key: pd.DataFrame, biowaste_lookup: pd.DataFrame) -> pd.DataFrame:
    if livestock_key.empty:
        return pd.DataFrame(columns=["KEY", "CITY", "YEAR", "SPECIES", "NUMBER", "MANURE_KG", "MANURE_N"])
    df = livestock_key.copy()
    df["SPECIES"] = df["SPECIES"].map(normalize_species)
    daily_rate = df["SPECIES"].map({k: sum(v.values()) for k, v in MANURE_UNIT_RATES.items()}).fillna(0.0)
    df["MANURE_KG"] = pd.to_numeric(df["NUMBER"], errors="coerce").fillna(0.0) * daily_rate * 365.0
    species_n_lookup = {species: get_species_n_per_kg(biowaste_lookup, species) for species in MANURE_UNIT_RATES}
    df["MANURE_N"] = pd.to_numeric(df["NUMBER"], errors="coerce").fillna(0.0) * df["SPECIES"].map(species_n_lookup).fillna(0.0) * 365.0
    return df[["KEY", "CITY", "YEAR", "SPECIES", "NUMBER", "MANURE_KG", "MANURE_N"]].copy()



def build_key_year_manure_species(manure_sources: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "KEY",
        "YEAR",
        "MANURE_DAIRY",
        "MANURE_CATTLE",
        "MANURE_SWINE",
        "MANURE_CHICKEN",
        "MANURE_BROILER",
    ]
    if manure_sources.empty:
        return pd.DataFrame(columns=cols)

    df = manure_sources.copy()
    df["KEY"] = df["KEY"].map(normalize_code)
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df["SPECIES"] = df["SPECIES"].map(normalize_species)
    df["MANURE_KG"] = pd.to_numeric(df["MANURE_KG"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["YEAR"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    pivot = (
        df.groupby(["KEY", "YEAR", "SPECIES"], as_index=False)["MANURE_KG"]
        .sum()
        .pivot_table(
            index=["KEY", "YEAR"],
            columns="SPECIES",
            values="MANURE_KG",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    pivot.columns.name = None
    rename_map = {
        "DAIRY": "MANURE_DAIRY",
        "CATTLE": "MANURE_CATTLE",
        "SWINE": "MANURE_SWINE",
        "CHICKEN": "MANURE_CHICKEN",
        "BROILER": "MANURE_BROILER",
    }
    pivot = pivot.rename(columns=rename_map)
    for col in cols:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot["YEAR"] = pivot["YEAR"].astype(int)
    return pivot[cols]


def manure_species_match(source_species: str, facility_species: str) -> bool:
    ss = normalize_species(source_species)
    fs = normalize_species(facility_species)
    if ss in POULTRY:
        return fs in POULTRY
    return ss == fs


def _collapse_duplicate_facility_slots(facilities: pd.DataFrame) -> pd.DataFrame:
    if facilities.empty or not {"SLOT_ID", "YEAR"}.issubset(facilities.columns):
        return facilities
    df = facilities.copy()
    if not df.duplicated(["SLOT_ID", "YEAR"]).any():
        return df

    rows: list[dict[str, Any]] = []
    for (_, _), g in df.groupby(["SLOT_ID", "YEAR"], dropna=False, sort=False):
        rec = g.iloc[0].to_dict()
        for col in ["CAPACITY", "REM_CAP"]:
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors="coerce")
                rec[col] = np.inf if np.isinf(vals).any() else float(vals.fillna(0.0).sum())
        for col in ["COMPOST_N", "COMPOST_P", "COMPOST_K", "LON", "LAT", "CITY"]:
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors="coerce").dropna()
                if not vals.empty:
                    rec[col] = float(vals.mean())
        rows.append(rec)
    log(f"Collapsed duplicate manure facility slots: {len(df)} rows -> {len(rows)} rows")
    return pd.DataFrame(rows)


def expand_facilities(manurefac: pd.DataFrame, years: Iterable[int]) -> pd.DataFrame:
    fac = manurefac.copy()
    fac["TYPE"] = fac["TYPE"].astype(str).str.upper().str.strip()
    fac["SPECIES"] = fac["SPECIES"].map(normalize_species)
    fac["LON"] = pd.to_numeric(fac["LON"], errors="coerce")
    fac["LAT"] = pd.to_numeric(fac["LAT"], errors="coerce")
    fac["CAPACITY"] = pd.to_numeric(fac["CAPACITY"], errors="coerce")
    fac["COMPOST_N"] = pd.to_numeric(fac["COMPOST_N"], errors="coerce")
    fac["COMPOST_P"] = pd.to_numeric(fac.get("COMPOST_P", np.nan), errors="coerce")
    fac["COMPOST_K"] = pd.to_numeric(fac.get("COMPOST_K", np.nan), errors="coerce")
    rows = []
    for _, row in fac.iterrows():
        slot_id = f"{row['FACILITY']}__{row['TYPE']}__{row['SPECIES']}"
        for year in years:
            rec = row.to_dict()
            rec["YEAR"] = int(year)
            rec["SLOT_ID"] = slot_id
            rec["REM_CAP"] = np.inf if str(row["TYPE"]).upper() == "COMPOST" else float(row["CAPACITY"])
            rows.append(rec)
    return _collapse_duplicate_facility_slots(pd.DataFrame(rows))


def assign_manure_to_facilities(
    ctx: InventoryContext,
    manure_sources: pd.DataFrame,
    facilities: pd.DataFrame,
    key_centroids: pd.DataFrame,
) -> pd.DataFrame:
    if manure_sources.empty or facilities.empty:
        return pd.DataFrame(columns=["FACILITY", "TYPE", "SPECIES", "YEAR", "KEY", "ASSIGNED_KG", "ASSIGNED_N", "DIST_KM", "COMPOST_N_CONC", "COMPOST_P_CONC", "COMPOST_K_CONC"])
    shared = build_shared_inputs(ctx)
    city_office_map = shared.get("city_office_map", {})
    cent = key_centroids.copy()
    cent["KEY"] = cent["KEY"].map(normalize_code)
    cent["CITY"] = pd.to_numeric(cent["CITY"], errors="coerce")
    cent_map = {str(r["KEY"]): (float(r["CENT_LON"]), float(r["CENT_LAT"]), r.get("CITY", np.nan)) for _, r in cent.iterrows()}
    fac_state = _collapse_duplicate_facility_slots(facilities.copy())
    fac_state["TYPE_PRIORITY"] = np.where(fac_state["TYPE"].eq("MWAD"), 0, 1)
    facilities = fac_state.copy()
    fac_state = fac_state.set_index(["SLOT_ID", "YEAR"], drop=False)
    if not fac_state.index.is_unique:
        dup = fac_state.index[fac_state.index.duplicated()].unique().tolist()[:10]
        raise ValueError(f"Duplicate manure facility SLOT_ID×YEAR rows remain after collapse: {dup}")
    rows: list[dict[str, Any]] = []

    for _, src in manure_sources.sort_values(["YEAR", "MANURE_KG"], ascending=[True, False]).iterrows():
        key = str(src["KEY"])
        year = int(src["YEAR"])
        species = normalize_species(src["SPECIES"])
        amount_left = float(src["MANURE_KG"])
        n_left = float(src["MANURE_N"])
        if amount_left <= 0:
            continue
        origin = cent_map.get(key)
        if origin is None:
            continue
        cands = facilities[(facilities["YEAR"] == year) & (facilities["SPECIES"].map(lambda x: manure_species_match(species, x)))].copy()
        if cands.empty:
            continue
        dists = []
        for _, fac in cands.iterrows():
            dist = get_city_relay_distance_km(ctx, f"KEY:{key}", origin[0], origin[1], origin[2], f"FAC:{fac['SLOT_ID']}", float(fac["LON"]), float(fac["LAT"]), fac.get("CITY", np.nan), city_office_map)
            dists.append(dist)
        cands["DIST_KM"] = dists
        cands = cands.sort_values(["TYPE_PRIORITY", "DIST_KM"], ascending=[True, True], na_position="last")
        n_per_kg = n_left / amount_left if amount_left > 0 else 0.0
        for _, fac in cands.iterrows():
            idx = (fac["SLOT_ID"], year)
            rem_cap = float(fac_state.at[idx, "REM_CAP"])
            if rem_cap <= 0:
                continue
            take = amount_left if np.isinf(rem_cap) else min(amount_left, rem_cap)
            if take <= 0:
                continue
            assigned_n = take * n_per_kg
            if np.isfinite(rem_cap):
                fac_state.at[idx, "REM_CAP"] = rem_cap - take
            rows.append({
                "FACILITY": fac["FACILITY"],
                "TYPE": fac["TYPE"],
                "SPECIES": species,
                "YEAR": year,
                "KEY": key,
                "ASSIGNED_KG": take,
                "ASSIGNED_N": assigned_n,
                "DIST_KM": float(fac["DIST_KM"]),
                "COMPOST_N_CONC": float(fac.get("COMPOST_N", np.nan)),
                "COMPOST_P_CONC": float(fac.get("COMPOST_P", np.nan)),
                "COMPOST_K_CONC": float(fac.get("COMPOST_K", np.nan)),
                "CAPACITY_KG": float(fac.get("CAPACITY", np.nan)) if pd.notna(fac.get("CAPACITY", np.nan)) else np.nan,
            })
            amount_left -= take
            n_left -= assigned_n
            if amount_left <= 1e-9:
                break
    return pd.DataFrame(rows)


def _resolve_module_path(script_dir: Path | None, filename: str) -> Path | None:
    candidates: list[Path] = []
    if script_dir is not None:
        candidates.append(Path(script_dir) / filename)
    try:
        candidates.append(Path(__file__).resolve().parent / filename)
    except Exception:
        pass
    candidates.append(Path.cwd() / filename)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def load_mwad_module(script_dir: Path | None):
    module_path = _resolve_module_path(script_dir, "MWAD_module.py")
    if module_path is None:
        return None
    spec = importlib.util.spec_from_file_location("MWAD_module_dynamic", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_manure_inventory(assignments: pd.DataFrame, mwad_module, manurefac: pd.DataFrame, ctx: InventoryContext | None = None) -> pd.DataFrame:
    fac_master = manurefac[["FACILITY", "TYPE"]].drop_duplicates().copy()
    grid = pd.MultiIndex.from_product([sorted(fac_master["FACILITY"].unique()), YEARS], names=["FACILITY", "YEAR"]).to_frame(index=False)
    grid = grid.merge(fac_master, on="FACILITY", how="left").drop_duplicates(["FACILITY", "YEAR", "TYPE"]) 
    if assignments.empty:
        out = grid.copy()
        for col in MANURE_OUTPUT_COLUMNS:
            if col not in out.columns:
                out[col] = 0.0 if col not in {"FACILITY", "TYPE", "YEAR"} else out.get(col)
        out["YEAR"] = out["YEAR"].astype(int)
        return finalize_output_schema(out, MANURE_OUTPUT_COLUMNS)

    rows = []
    for (facility, year, type_), g in assignments.groupby(["FACILITY", "YEAR", "TYPE"], dropna=False):
        type_upper = str(type_).upper().strip()
        is_mwad = type_upper == "MWAD"
        residue_req_raw = np.sum(g["ASSIGNED_KG"] * g["SPECIES"].map(RESIDUE_REQ_COEF).fillna(0.0))
        compost_kg_raw = np.sum(g["ASSIGNED_KG"] * g["SPECIES"].map(COMPOST_COEF).fillna(0.0))
        residue_req = 0.0 if is_mwad else residue_req_raw
        compost_kg = 0.0 if is_mwad else compost_kg_raw
        compost_n = 0.0 if is_mwad else np.sum(g["ASSIGNED_KG"] * g["SPECIES"].map(COMPOST_COEF).fillna(0.0) * g["COMPOST_N_CONC"].fillna(0.0))
        compost_p2o5 = 0.0 if is_mwad else np.sum(g["ASSIGNED_KG"] * g["SPECIES"].map(COMPOST_COEF).fillna(0.0) * g["COMPOST_P_CONC"].fillna(0.0))
        manure_total = float(g["ASSIGNED_KG"].sum())
        species_kg = g.groupby("SPECIES")["ASSIGNED_KG"].sum().to_dict()
        rows.append({
            "FACILITY": facility,
            "YEAR": int(year),
            "TYPE": type_,
            "MANURE_KG": manure_total,
            "MANURE_N": g["ASSIGNED_N"].sum(),
            **compute_species_manure_kg(species_kg),
            "MANURE_KM": np.average(g["DIST_KM"], weights=np.maximum(g["ASSIGNED_KG"], 1e-9)),
            "RESIDUE_REQ_KG": residue_req,
            "RESIDUE_SUPPLY_KG": 0.0,
            "RESIDUE_IMPORT_KG": residue_req,
            "RESIDUE_SUPPLIED_KM": 0.0,
            "RESIDUE_IMPORT_KM": 0.0,
            "COMPOST_KG": compost_kg,
            "COMPOST_N": compost_n,
            "COMPOST_P2O5": compost_p2o5,
        })
    out = pd.DataFrame(rows)
    out = grid.merge(out, on=["FACILITY", "YEAR", "TYPE"], how="left")
    numeric_cols = [c for c in out.columns if c not in {"FACILITY", "TYPE", "YEAR"}]
    out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    for col in ["DIGEST_KG", "DIGEST_C", "DIGEST_N", "DIGEST_P2O5", "BIOGAS_CH4_NM3", "MWAD_ELEC_USE_KWH", "MWAD_HEAT_USE_MJ", "MWAD_WATER_USE_M3", "MWAD_REFUSED_KG"]:
        out[col] = 0.0

    if mwad_module is not None:
        cap_map = manurefac[manurefac["TYPE"].astype(str).str.upper() == "MWAD"].copy()
        cap_map["CAPACITY"] = pd.to_numeric(cap_map["CAPACITY"], errors="coerce")
        cap_map = cap_map.groupby("FACILITY", dropna=False)["CAPACITY"].sum().to_dict()
        for idx in out.index[out["TYPE"].eq("MWAD")]:
            manure_kg = float(out.at[idx, "MANURE_KG"])
            capacity = float(cap_map.get(out.at[idx, "FACILITY"], manure_kg))
            try:
                res = mwad_module.mwad_run(food_waste_amount=manure_kg, capacity_amount=capacity)
            except Exception:
                res = {}
            out.at[idx, "DIGEST_KG"] = float(res.get("MWAD_DIGESTATE", 0.0))
            out.at[idx, "DIGEST_C"] = float(res.get("MWAD_DIG_ORGC", 0.0))
            out.at[idx, "DIGEST_N"] = float(res.get("MWAD_DIG_N", 0.0))
            out.at[idx, "DIGEST_P2O5"] = float(res.get("MWAD_DIG_P", 0.0))
            out.at[idx, "BIOGAS_CH4_NM3"] = float(res.get("MWAD_BIOGAS", 0.0))
            out.at[idx, "MWAD_ELEC_USE_KWH"] = float(res.get("MWAD_ELEC_REQ", 0.0))
            out.at[idx, "MWAD_HEAT_USE_MJ"] = float(res.get("MWAD_HEAT_REQ", 0.0))
            out.at[idx, "MWAD_WATER_USE_M3"] = float(res.get("MWAD_WATER_REQ", 0.0))
            out.at[idx, "MWAD_REFUSED_KG"] = float(res.get("MWAD_REFUSED", 0.0))
    if ctx is not None and not out.empty:
        try:
            shared = build_shared_inputs(ctx)
            offices = shared.get("offices", pd.DataFrame()).copy()
            fac_loc = manurefac[["FACILITY", "LAT", "LON"]].drop_duplicates().copy()
            fac_loc["LAT"] = pd.to_numeric(fac_loc["LAT"], errors="coerce")
            fac_loc["LON"] = pd.to_numeric(fac_loc["LON"], errors="coerce")
            loc_map = {
                str(r["FACILITY"]): (float(r["LON"]), float(r["LAT"]))
                for _, r in fac_loc.iterrows()
                if pd.notna(r["LON"]) and pd.notna(r["LAT"])
            }
            for idx in out.index:
                if float(out.at[idx, "RESIDUE_IMPORT_KG"]) > 0:
                    fac = str(out.at[idx, "FACILITY"])
                    lonlat = loc_map.get(fac)
                    if lonlat is not None:
                        out.at[idx, "RESIDUE_IMPORT_KM"] = nearest_office_distance_plus_300(
                            ctx, lonlat[0], lonlat[1], offices, f"FAC:{fac}"
                        )
                if str(out.at[idx, "TYPE"]).upper() == "MWAD":
                    out.at[idx, "RESIDUE_REQ_KG"] = 0.0
                    out.at[idx, "RESIDUE_SUPPLY_KG"] = 0.0
                    out.at[idx, "RESIDUE_IMPORT_KG"] = 0.0
                    out.at[idx, "RESIDUE_SUPPLIED_KM"] = 0.0
                    out.at[idx, "RESIDUE_IMPORT_KM"] = 0.0
        except Exception:
            pass
    out = enforce_species_manure_mass_balance(out)
    out = zero_mwad_compost_residue_fields(out)
    out["YEAR"] = out["YEAR"].astype(int)
    return finalize_output_schema(out, MANURE_OUTPUT_COLUMNS)


def allocate_residue_supply(
    ctx: InventoryContext,
    crop_prelim: pd.DataFrame,
    manure_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = build_shared_inputs(ctx)
    key_cent = shared["key_centroids"].copy()
    key_cent["KEY"] = key_cent["KEY"].map(normalize_code)
    key_cent["CITY"] = pd.to_numeric(key_cent["CITY"], errors="coerce")
    key_map = {str(r["KEY"]): (float(r["CENT_LON"]), float(r["CENT_LAT"]), r.get("CITY", np.nan)) for _, r in key_cent.iterrows()}
    city_office_map = shared.get("city_office_map", {})

    crop_adj = crop_prelim.copy()
    crop_adj["KEY"] = crop_adj["KEY"].map(normalize_code)
    crop_adj["YEAR"] = pd.to_numeric(crop_adj["YEAR"], errors="coerce").astype(int)
    crop_adj["CROP"] = crop_adj["CROP"].astype(str).str.upper().str.strip()
    for c in ["RESIDUE_KG", "RESIDUE_C_KG", "RESIDUE_N_KG"]:
        crop_adj[c] = pd.to_numeric(crop_adj[c], errors="coerce").fillna(0.0)

    supply_df = crop_adj[crop_adj["CROP"].eq("RICE")].groupby(["KEY", "YEAR"], as_index=False)[["RESIDUE_KG", "RESIDUE_C_KG", "RESIDUE_N_KG"]].sum()
    supply_df["SOURCE_YEAR"] = supply_df["YEAR"].astype(int)
    if supply_df.empty or manure_summary.empty:
        manure_out = manure_summary.copy()
        if not manure_out.empty:
            manure_out["RESIDUE_SUPPLY_KG"] = pd.to_numeric(manure_out["RESIDUE_SUPPLY_KG"], errors="coerce").fillna(0.0)
            manure_out["RESIDUE_IMPORT_KG"] = pd.to_numeric(manure_out["RESIDUE_REQ_KG"], errors="coerce").fillna(0.0)
            manure_out["RESIDUE_SUPPLIED_KM"] = pd.to_numeric(manure_out.get("RESIDUE_SUPPLIED_KM", 0.0), errors="coerce").fillna(0.0)
            manure_out["RESIDUE_IMPORT_KM"] = pd.to_numeric(manure_out.get("RESIDUE_IMPORT_KM", 0.0), errors="coerce").fillna(0.0)
        manure_out = zero_mwad_compost_residue_fields(manure_out)
        return crop_adj, manure_out

    supply_state = supply_df.copy()
    fac_master = shared["manurefac"].copy()
    fac_master["FACILITY"] = fac_master["FACILITY"].astype(str)
    fac_master["CITY"] = pd.to_numeric(fac_master["CITY"], errors="coerce")
    fac_master["LON"] = pd.to_numeric(fac_master["LON"], errors="coerce")
    fac_master["LAT"] = pd.to_numeric(fac_master["LAT"], errors="coerce")
    fac_master = fac_master.drop_duplicates(["FACILITY"])
    fac_loc = {str(r["FACILITY"]): (float(r["LON"]), float(r["LAT"]), r.get("CITY", np.nan)) for _, r in fac_master.iterrows() if pd.notna(r["LON"]) and pd.notna(r["LAT"])}

    manure_out = manure_summary.copy()
    manure_out["RESIDUE_SUPPLY_KG"] = 0.0
    manure_out["RESIDUE_IMPORT_KG"] = pd.to_numeric(manure_out["RESIDUE_REQ_KG"], errors="coerce").fillna(0.0)
    manure_out["RESIDUE_SUPPLIED_KM"] = 0.0
    manure_out["RESIDUE_IMPORT_KM"] = 0.0

    allocations = []
    req_rows = manure_out[manure_out["RESIDUE_REQ_KG"] > 0].sort_values(["YEAR", "RESIDUE_REQ_KG"], ascending=[True, False])
    for idx, req in req_rows.iterrows():
        facility = str(req["FACILITY"])
        year = int(req["YEAR"])
        need = float(req["RESIDUE_REQ_KG"])
        origin = fac_loc.get(facility)
        if need <= 0 or origin is None:
            continue
        avail = supply_state[(supply_state["YEAR"] == year) & (supply_state["RESIDUE_KG"] > 0)].copy()
        if avail.empty:
            continue
        dists = []
        for _, src in avail.iterrows():
            key = str(src["KEY"])
            loc = key_map.get(key)
            if loc is None:
                dists.append(np.inf)
            else:
                dists.append(get_city_relay_distance_km(ctx, f"FAC:{facility}", origin[0], origin[1], origin[2], f"KEY:{key}", loc[0], loc[1], loc[2], city_office_map))
        avail["DIST_KM"] = dists
        avail = avail.replace([np.inf, -np.inf], np.nan).dropna(subset=["DIST_KM"]).sort_values(["DIST_KM"], ascending=[True])
        if avail.empty:
            continue
        supplied = 0.0
        wdist_num = 0.0
        for sidx, src in avail.iterrows():
            take = min(need, float(src["RESIDUE_KG"]))
            if take <= 0:
                continue
            frac = take / float(src["RESIDUE_KG"]) if float(src["RESIDUE_KG"]) > 0 else 0.0
            supply_state.at[sidx, "RESIDUE_KG"] = float(src["RESIDUE_KG"]) - take
            supply_state.at[sidx, "RESIDUE_C_KG"] = float(src["RESIDUE_C_KG"]) * (1.0 - frac)
            supply_state.at[sidx, "RESIDUE_N_KG"] = float(src["RESIDUE_N_KG"]) * (1.0 - frac)
            supplied += take
            wdist_num += take * float(src["DIST_KM"])
            residue_n_per_kg = float(src["RESIDUE_N_KG"]) / float(src["RESIDUE_KG"]) if float(src["RESIDUE_KG"]) > 0 else RESIDUE_N_PER_KG
            allocations.append({
                "FACILITY": facility,
                "YEAR": year,
                "SOURCE_YEAR": int(src.get("SOURCE_YEAR", year)),
                "KEY": str(src["KEY"]),
                "TAKE_KG": take,
                "TAKE_N_KG": take * residue_n_per_kg,
            })
            need -= take
            if need <= 1e-9:
                break
        manure_out.at[idx, "RESIDUE_SUPPLY_KG"] = supplied
        manure_out.at[idx, "RESIDUE_IMPORT_KG"] = max(float(req["RESIDUE_REQ_KG"]) - supplied, 0.0)
        manure_out.at[idx, "RESIDUE_SUPPLIED_KM"] = (wdist_num / supplied) if supplied > 0 else 0.0

    if allocations:
        alloc_df = pd.DataFrame(allocations)
        try:
            shared["_residue_allocations"] = alloc_df.copy()
        except Exception:
            pass
        used = (
            alloc_df.groupby(["KEY", "SOURCE_YEAR"], as_index=False)["TAKE_KG"]
            .sum()
            .rename(columns={"SOURCE_YEAR": "YEAR", "TAKE_KG": "USED_RESIDUE_KG"})
        )
        crop_adj = crop_adj.merge(used, on=["KEY", "YEAR"], how="left")
        crop_adj["USED_RESIDUE_KG"] = pd.to_numeric(crop_adj["USED_RESIDUE_KG"], errors="coerce").fillna(0.0)
        rice_mask = crop_adj["CROP"].eq("RICE")
        available = crop_adj["RESIDUE_KG"].where(rice_mask, 0.0)
        frac = np.where((rice_mask) & (available > 0), np.minimum(crop_adj["USED_RESIDUE_KG"], available) / available, 0.0)
        crop_adj.loc[rice_mask, "RESIDUE_C_KG"] = crop_adj.loc[rice_mask, "RESIDUE_C_KG"] * (1.0 - frac[rice_mask])
        crop_adj.loc[rice_mask, "RESIDUE_N_KG"] = crop_adj.loc[rice_mask, "RESIDUE_N_KG"] * (1.0 - frac[rice_mask])
        crop_adj.loc[rice_mask, "RESIDUE_KG"] = np.maximum(crop_adj.loc[rice_mask, "RESIDUE_KG"] - np.minimum(crop_adj.loc[rice_mask, "USED_RESIDUE_KG"], crop_adj.loc[rice_mask, "RESIDUE_KG"]), 0.0)
        crop_adj = crop_adj.drop(columns=["USED_RESIDUE_KG"])

    if not manure_out.empty:
        import_tol = 1e-6
        offices = shared.get("offices", pd.DataFrame()).copy()
        fac_master2 = shared["manurefac"].copy()
        fac_master2["FACILITY"] = fac_master2["FACILITY"].astype(str)
        fac_master2["LON"] = pd.to_numeric(fac_master2["LON"], errors="coerce")
        fac_master2["LAT"] = pd.to_numeric(fac_master2["LAT"], errors="coerce")
        fac_loc2 = {
            str(r["FACILITY"]): (float(r["LON"]), float(r["LAT"]))
            for _, r in fac_master2.drop_duplicates(["FACILITY"]).iterrows()
            if pd.notna(r["LON"]) and pd.notna(r["LAT"])
        }
        for idx in manure_out.index:
            imp = float(pd.to_numeric(manure_out.at[idx, "RESIDUE_IMPORT_KG"], errors="coerce") or 0.0)
            if abs(imp) < import_tol:
                manure_out.at[idx, "RESIDUE_IMPORT_KG"] = 0.0
                if abs(float(pd.to_numeric(manure_out.at[idx, "RESIDUE_SUPPLY_KG"], errors="coerce") or 0.0)) < import_tol:
                    manure_out.at[idx, "RESIDUE_SUPPLIED_KM"] = 0.0
                    manure_out.at[idx, "RESIDUE_IMPORT_KM"] = 0.0
                continue
            fac = str(manure_out.at[idx, "FACILITY"])
            lonlat = fac_loc2.get(fac)
            if lonlat is not None:
                manure_out.at[idx, "RESIDUE_IMPORT_KM"] = nearest_office_distance_plus_300(ctx, lonlat[0], lonlat[1], offices, f"FAC:{fac}")

    if "IS_CARRYOVER_YEAR" in crop_adj.columns and str(ctx.scenario.scenario_name).upper() != "OPT":
        crop_adj = crop_adj.loc[~crop_adj["IS_CARRYOVER_YEAR"].astype(bool)].copy()
        crop_adj = crop_adj.drop(columns=["IS_CARRYOVER_YEAR"], errors="ignore")

    manure_out = clean_small_numeric(manure_out)
    return crop_adj, finalize_output_schema(manure_out, MANURE_OUTPUT_COLUMNS)


def build_manure_inventory(
    ctx: InventoryContext,
    crop_prelim: pd.DataFrame | None = None,
) -> pd.DataFrame:
    shared = build_shared_inputs(ctx)
    biowaste_lookup = build_biowaste_lookup(shared["biowaste"])
    city_proj = project_city_species_livestock(shared["livestock"], ctx.years)
    livestock_key = distribute_city_species_to_keys(city_proj, shared["rcom"])
    manure_sources = build_manure_sources(livestock_key, biowaste_lookup)
    facilities = expand_facilities(shared["manurefac"], ctx.years)
    assignments = assign_manure_to_facilities(ctx, manure_sources, facilities, shared["key_centroids"])
    # Preserve source-level manure routing for 02_waste_link.csv export.
    try:
        shared["_manure_assignments"] = assignments.copy()
    except Exception:
        pass
    mwad_module = load_mwad_module(ctx.paths.script_dir)
    manure = summarize_manure_inventory(assignments, mwad_module, shared["manurefac"], ctx=ctx)
    manure = apply_alt_manure_rerouting(ctx, manure, mwad_module)
    if crop_prelim is not None and not crop_prelim.empty:
        _, manure = allocate_residue_supply(ctx, crop_prelim, manure)
    return manure


def build_crop_inventory(
    ctx: InventoryContext,
    manure_inventory: pd.DataFrame | None = None,
) -> pd.DataFrame:
    shared = build_shared_inputs(ctx)
    cropland = shared["cropland"].copy()
    rcom = shared["rcom"].copy()
    agro = shared["agroinput"].copy()
    env = shared["env"].copy()
    key_cent = shared["key_centroids"].copy()
    offices = shared["offices"].copy()
    manurefac = shared["manurefac"].copy()
    biowaste_lookup = build_biowaste_lookup(shared["biowaste"])
    city_proj_livestock = project_city_species_livestock(shared["livestock"], ctx.years)
    livestock_key = distribute_city_species_to_keys(city_proj_livestock, shared["rcom"])
    manure_sources_key = build_manure_sources(livestock_key, biowaste_lookup)
    manure_key_species = build_key_year_manure_species(manure_sources_key)

    for df in [cropland, rcom, env, key_cent]:
        if "KEY" in df.columns:
            df["KEY"] = df["KEY"].map(normalize_code)
    cropland["CROP"] = cropland["CROP"].astype(str).str.upper().str.strip()
    cropland["CULTIVAR"] = cropland["CULTIVAR"].astype(str).str.lower().str.strip()
    cropland["YEAR"] = pd.to_numeric(cropland["YEAR"], errors="coerce")
    cropland["EXTENT"] = pd.to_numeric(cropland["EXTENT"], errors="coerce").fillna(0.0)
    cropland["CITY"] = pd.to_numeric(cropland["CITY"], errors="coerce")
    cropland["KCITY"] = pd.to_numeric(cropland["KCITY"], errors="coerce")
    cropland["RCOM"] = pd.to_numeric(cropland["RCOM"], errors="coerce").fillna(0).astype(int)

    rcom["CITY"] = pd.to_numeric(rcom["CITY"], errors="coerce")
    rcom["KCITY"] = pd.to_numeric(rcom["KCITY"], errors="coerce")
    rcom["RCOM"] = pd.to_numeric(rcom["RCOM"], errors="coerce").fillna(0).astype(int)

    agro["CROP"] = agro["CROP"].astype(str).str.upper().str.strip()
    agro["CULTIVAR"] = agro["CULTIVAR"].astype(str).str.lower().str.strip()
    agro["YEAR"] = pd.to_numeric(agro["YEAR"], errors="coerce")
    agro["CITY"] = pd.to_numeric(agro["CITY"], errors="coerce")
    for c in ["NIT", "PHO", "KAL", "N_REQ", "AGROCHEM", "DIESEL", "GASOL", "KEROS", "ELEC"]:
        agro[c] = pd.to_numeric(agro[c], errors="coerce")

    env["CITY"] = pd.to_numeric(env["CITY"], errors="coerce")
    env["SOILTEMP"] = pd.to_numeric(env["SOILTEMP"], errors="coerce")
    env["PRECIP"] = pd.to_numeric(env["PRECIP"], errors="coerce")
    env["TN"] = pd.to_numeric(env["TN"], errors="coerce")
    for c in [col for col in env.columns if col.startswith("AIRTEMP") or col.startswith("WIND") or col in ["pH", "CEC", "SAND"]]:
        env[c] = pd.to_numeric(env[c], errors="coerce")

    offices["CITY"] = pd.to_numeric(offices.get("CITY", np.nan), errors="coerce")
    manurefac["CITY"] = pd.to_numeric(manurefac["CITY"], errors="coerce")
    manurefac["TYPE"] = manurefac["TYPE"].astype(str).str.upper().str.strip()
    manurefac["LON"] = pd.to_numeric(manurefac["LON"], errors="coerce")
    manurefac["LAT"] = pd.to_numeric(manurefac["LAT"], errors="coerce")
    manurefac["COMPOST_N"] = pd.to_numeric(manurefac["COMPOST_N"], errors="coerce")
    manurefac["COMPOST_P"] = pd.to_numeric(manurefac.get("COMPOST_P", np.nan), errors="coerce")
    manurefac["COMPOST_K"] = pd.to_numeric(manurefac.get("COMPOST_K", np.nan), errors="coerce")

    # project city/KCITY crop extents
    crop_years = sorted({int(y) for y in ctx.years} | {CROP_CARRYOVER_START_YEAR})
    city_src = cropland[cropland["RCOM"] == 0].copy()
    rows = []

    city_src_nonveg = city_src[city_src["CROP"] != "VEG"].copy()
    for keys, grp in city_src_nonveg.groupby(["CITY", "KCITY", "CROP", "CULTIVAR"], dropna=False):
        city, kcity, crop, cultivar = keys
        if pd.isna(city) or grp["EXTENT"].sum() <= 0:
            continue
        yhat = linear_regression_projection(grp["YEAR"].to_numpy(), grp["EXTENT"].to_numpy(), np.asarray(crop_years))
        yhat = np.where(np.isfinite(yhat), np.maximum(yhat, 0.0), 0.0)
        for year, extent in zip(crop_years, yhat, strict=False):
            rows.append({
                "CITY": int(city),
                "KCITY": int(kcity) if pd.notna(kcity) else 0,
                "CROP": crop,
                "CULTIVAR": cultivar,
                "YEAR": int(year),
                "CITY_EXTENT_A": float(extent),
            })

    city_src_veg = city_src[city_src["CROP"] == "VEG"].copy()
    for keys, grp in city_src_veg.groupby(["CITY", "KCITY", "CROP"], dropna=False):
        city, kcity, crop = keys
        if pd.isna(city) or grp["EXTENT"].sum() <= 0:
            continue
        yhat = linear_regression_projection(grp["YEAR"].to_numpy(), grp["EXTENT"].to_numpy(), np.asarray(crop_years))
        yhat = np.where(np.isfinite(yhat), np.maximum(yhat, 0.0), 0.0)
        ratio_map = VEG_CULTIVAR_RATIOS_KYOTO if int(city) in KYOTO_CITIES else VEG_CULTIVAR_RATIOS_OTHER
        for year, extent in zip(crop_years, yhat, strict=False):
            for cultivar, ratio in ratio_map.items():
                rows.append({
                    "CITY": int(city),
                    "KCITY": int(kcity) if pd.notna(kcity) else 0,
                    "CROP": crop,
                    "CULTIVAR": cultivar,
                    "YEAR": int(year),
                    "CITY_EXTENT_A": float(extent) * float(ratio),
                })

    city_proj = pd.DataFrame(rows)
    if city_proj.empty:
        return finalize_output_schema(empty_crop_output(), CROP_OUTPUT_COLUMNS)


    crop_ratio_col = {"RICE": "RICE_RATIO", "TEA": "TEA_RATIO", "VEG": "VEG_RATIO"}
    rcom_keys = rcom[rcom["RCOM"] != 0].copy()
    alloc_rows = []
    for _, row in city_proj.iterrows():
        crop = row["CROP"]
        ratio_col = crop_ratio_col.get(crop)
        if int(row["CITY"]) == 108 and int(row["KCITY"]) != 0:
            sub = rcom_keys[(rcom_keys["CITY"] == row["CITY"]) & (rcom_keys["KCITY"] == row["KCITY"])]
        else:
            sub = rcom_keys[rcom_keys["CITY"] == row["CITY"]]
        if sub.empty:
            continue
        ratios = pd.to_numeric(sub[ratio_col], errors="coerce").fillna(0.0) if ratio_col in sub.columns else pd.Series(0.0, index=sub.index)
        if ratios.sum() <= 0:
            ratios = pd.Series(np.ones(len(sub)) / len(sub), index=sub.index)
        else:
            ratios = ratios / ratios.sum()
        for idx, skey in enumerate(sub["KEY"]):
            extent_ha = float(row["CITY_EXTENT_A"]) * float(ratios.iloc[idx]) * 0.01
            if extent_ha < 0.1:
                extent_ha = 0.0
            alloc_rows.append({
                "KEY": skey, "CITY": int(row["CITY"]), "KCITY": int(row["KCITY"]), "YEAR": int(row["YEAR"]),
                "CROP": crop, "CULTIVAR": row["CULTIVAR"], "EXTENT_HA": extent_ha
            })
    crop_df = pd.DataFrame(alloc_rows)
    crop_df = crop_df[crop_df["EXTENT_HA"] > 0].copy()
    if crop_df.empty:
        return finalize_output_schema(empty_crop_output(), CROP_OUTPUT_COLUMNS)

    # Keep one internal carryover year so rice residue from 2029 can supply 2030.
    crop_df["IS_CARRYOVER_YEAR"] = crop_df["YEAR"].astype(int) < min(ctx.years)

    # organic flag assignment (persistent by KEY x CROP)
    org_status: dict[tuple[str, str], bool] = {}
    flags = []
    for year in sorted(crop_years):
        for crop in ["RICE", "TEA", "VEG"]:
            sub = crop_df[(crop_df["YEAR"] == year) & (crop_df["CROP"] == crop)].copy()
            if sub.empty:
                continue
            key_sub = sub.groupby(["KEY", "CITY"], as_index=False)["EXTENT_HA"].sum()
            eligible = key_sub[key_sub["EXTENT_HA"] > 0].copy()
            if eligible.empty:
                continue
            org_share = ORG_SHARE_2020[crop] + (ORG_SHARE_2050 - ORG_SHARE_2020[crop]) * ((year - 2020) / (2050 - 2020))
            target_count = int(math.ceil(len(eligible) * org_share))
            existing = [k for k in eligible["KEY"] if org_status.get((str(k), crop), False)]
            current = set(existing)
            need = max(0, target_count - len(current))
            if need > 0:
                cand = eligible[~eligible["KEY"].isin(current)].copy()
                cand["PRIORITY"] = cand["CITY"].astype(int).isin(ORG_PRIORITY_CITIES[crop]).astype(int)
                cand = cand.sort_values(["PRIORITY", "EXTENT_HA", "KEY"], ascending=[False, False, True])
                for key in cand["KEY"].head(need):
                    org_status[(str(key), crop)] = True
                    current.add(str(key))
            for key in eligible["KEY"]:
                flags.append({"KEY": str(key), "YEAR": year, "CROP": crop, "ORG_FLAG": "Y" if org_status.get((str(key), crop), False) else "X"})
    flag_df = pd.DataFrame(flags).drop_duplicates(["KEY", "YEAR", "CROP"], keep="last")
    crop_df = crop_df.merge(flag_df, on=["KEY", "YEAR", "CROP"], how="left")
    crop_df["ORG_FLAG"] = crop_df["ORG_FLAG"].fillna("X")

    def agro_group_key(row):
        crop = row["CROP"]
        if crop == "TEA":
            return (crop, int(row["CITY"]) if pd.notna(row["CITY"]) else -1, row["CULTIVAR"])
        return (crop, -1, row["CULTIVAR"])

    years_arr = np.asarray(crop_years)
    agchem_global = linear_regression_projection(
        agro.loc[agro["AGROCHEM"].notna(), "YEAR"].to_numpy(),
        agro.loc[agro["AGROCHEM"].notna(), "AGROCHEM"].to_numpy(),
        years_arr,
    )
    agchem_global = np.where(np.isfinite(agchem_global), np.maximum(agchem_global, 0.0), 0.0)

    proj_rows = []
    for keys, grp in agro.groupby([agro.apply(agro_group_key, axis=1)], dropna=False):
        crop, city_key, cultivar = keys[0] if isinstance(keys, tuple) and len(keys)==1 else keys
        grp = grp.sort_values("YEAR").copy()
        row_base = {"CROP": crop, "CITY_AGRO": city_key, "CULTIVAR": cultivar}
        vals = {}
        hist_years = grp["YEAR"].to_numpy(dtype=float)
        for col in ["NIT", "PHO", "KAL", "N_REQ", "DIESEL", "GASOL", "KEROS", "ELEC"]:
            vals[col] = linear_regression_projection(hist_years, grp[col].to_numpy(), years_arr)
            vals[col] = np.where(np.isfinite(vals[col]), np.maximum(vals[col], 0.0), 0.0)
            if col == "NIT":
                hist_vals = pd.to_numeric(grp[col], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(hist_years) & np.isfinite(hist_vals)
                hyears = hist_years[valid]
                hvals = hist_vals[valid]
                if len(hvals) > 0:
                    if np.any(hyears == 2019):
                        nit_2019 = float(hvals[np.where(hyears == 2019)[0][-1]])
                    else:
                        nit_2019 = float(hvals[np.argmin(np.abs(hyears - 2019))])
                    target_2050 = max(0.7 * nit_2019, 0.0)
                    if np.any(years_arr == 2050):
                        proj_2050 = float(vals[col][np.where(years_arr == 2050)[0][0]])
                        if np.isfinite(proj_2050) and proj_2050 > 0:
                            vals[col] = vals[col] * (target_2050 / proj_2050)
                        else:
                            vals[col] = np.interp(years_arr, [years_arr.min(), 2050], [max(nit_2019, 0.0), target_2050])
                    vals[col] = np.where(np.isfinite(vals[col]), np.maximum(vals[col], 0.0), 0.0)
        for i, year in enumerate(crop_years):
            rec = {**row_base, "YEAR": int(year), "AGROCHEM": float(agchem_global[i])}
            for col in vals:
                rec[col] = float(vals[col][i])
            proj_rows.append(rec)
    agro_proj = pd.DataFrame(proj_rows)

    crop_df["CITY_AGRO"] = np.where(crop_df["CROP"].eq("TEA"), crop_df["CITY"], -1)
    crop_df = crop_df.merge(agro_proj, left_on=["CROP", "CITY_AGRO", "CULTIVAR", "YEAR"], right_on=["CROP", "CITY_AGRO", "CULTIVAR", "YEAR"], how="left")
    for c in ["NIT", "PHO", "KAL", "N_REQ", "AGROCHEM", "DIESEL", "GASOL", "KEROS", "ELEC"]:
        crop_df[c] = pd.to_numeric(crop_df[c], errors="coerce").fillna(0.0)

    city_office_map = shared.get("city_office_map", {})
    cent_map = {str(r["KEY"]): (float(r["CENT_LON"]), float(r["CENT_LAT"]), r.get("CITY", np.nan)) for _, r in key_cent.iterrows()}
    offices = offices.copy()
    if not offices.empty:
        tmp_off = pd.to_numeric(offices["OFFICE_ID"], errors="coerce")
        offices["OFFICE_NUM"] = tmp_off.where(tmp_off.notna(), pd.Series(np.arange(len(offices)), index=offices.index)).astype(float) + 300
    compost_f = (
        manurefac[manurefac["TYPE"] == "COMPOST"]
        .groupby("FACILITY", as_index=False)
        .agg({
            "CITY": "first",
            "LON": "mean",
            "LAT": "mean",
            "COMPOST_N": "mean",
            "COMPOST_P": "mean",
            "COMPOST_K": "mean",
        })
        .copy()
    )
    mwad_f = manurefac[manurefac["TYPE"] == "MWAD"].drop_duplicates(["FACILITY", "CITY", "LON", "LAT"]).copy()

    def nearest_dest(key, dest_df, prefix, use_city_relay: bool = False):
        if dest_df.empty or key not in cent_map:
            return (None, np.nan, np.nan, np.nan, np.nan)
        lon, lat, city = cent_map[key]
        best = None
        best_km = np.inf
        best_n = np.nan
        best_p = np.nan
        best_k = np.nan
        for _, d in dest_df.iterrows():
            did = f"{prefix}:{d.get('FACILITY', d.get('OFFICE_NUM', d.get('OFFICE_ID')))}"
            if use_city_relay:
                km = get_city_relay_distance_km(
                    ctx,
                    f"KEY:{key}",
                    lon,
                    lat,
                    city,
                    did,
                    float(d["LON"]),
                    float(d["LAT"]),
                    d.get("CITY", np.nan),
                    city_office_map,
                )
            else:
                km = get_distance_km(ctx, f"KEY:{key}", lon, lat, did, float(d["LON"]), float(d["LAT"]))
            if km < best_km:
                best = d
                best_km = km
                best_n = float(d.get("COMPOST_N", np.nan))
                best_p = float(d.get("COMPOST_P", np.nan))
                best_k = float(d.get("COMPOST_K", np.nan))
        return (best, float(best_km), best_n, best_p, best_k)

    digest_n_conc_map = {}
    digest_p_conc_map = {}
    if manure_inventory is not None and not manure_inventory.empty:
        mi = manure_inventory.copy()
        mi = mi[mi["TYPE"].astype(str).str.upper() == "MWAD"].copy()
        if not mi.empty:
            digest_kg = pd.to_numeric(mi["DIGEST_KG"], errors="coerce")
            mi["DIGEST_N_CONC"] = np.where(digest_kg > 0, pd.to_numeric(mi["DIGEST_N"], errors="coerce") / digest_kg, np.nan)
            mi["DIGEST_P_CONC"] = np.where(digest_kg > 0, pd.to_numeric(mi.get("DIGEST_P2O5", 0.0), errors="coerce") / digest_kg, np.nan)
            digest_n_conc_map = {(str(r["FACILITY"]), int(r["YEAR"])): float(r["DIGEST_N_CONC"]) for _, r in mi.iterrows() if pd.notna(r["DIGEST_N_CONC"])}
            digest_p_conc_map = {(str(r["FACILITY"]), int(r["YEAR"])): float(r["DIGEST_P_CONC"]) for _, r in mi.iterrows() if pd.notna(r["DIGEST_P_CONC"])}
    if not digest_n_conc_map:
        digest_n_conc_default = 0.037
    else:
        digest_n_conc_default = float(np.nanmean(list(digest_n_conc_map.values())))
    if not digest_p_conc_map:
        digest_p_conc_default = 0.0
    else:
        digest_p_conc_default = float(np.nanmean(list(digest_p_conc_map.values())))

    syn_km = []
    comp_loc_km = []
    comp_imp_km = []
    dig_km = []
    agchem_km = []
    source_type = []
    comp_n_conc = []
    comp_p_conc = []
    comp_k_conc = []
    dig_n_conc = []
    dig_p_conc = []
    linked_digest_facility = []

    linked_compost_by_key: dict[str, tuple[Any, float, float, float, float]] = {}
    for key in crop_df["KEY"].astype(str).drop_duplicates():
        comp_fac, ck, cn, cp, ckc = nearest_dest(key, compost_f, "COMP", use_city_relay=True)
        linked_compost_by_key[key] = (comp_fac, ck, cn, cp, ckc)

    compost_facility = []

    for _, row in crop_df.iterrows():
        key = str(row["KEY"])
        _, office_km, _, _, _ = nearest_dest(key, offices.rename(columns={"OFFICE_NUM": "FACILITY"}), "OFF", use_city_relay=False)
        comp_fac, ck, cn, cp, ckc = linked_compost_by_key.get(key, (None, np.nan, np.nan, np.nan, np.nan))
        if row["CROP"] == "RICE" and int(row["CITY"]) == 213:
            pref_mwad = mwad_f[mwad_f["FACILITY"].astype(str).eq("NANTAN_YAGI")].copy()
            dig_fac, dk, _, _, _ = nearest_dest(key, pref_mwad if not pref_mwad.empty else mwad_f, "MWAD", use_city_relay=True)
        else:
            dig_fac, dk, _, _, _ = nearest_dest(key, mwad_f, "MWAD", use_city_relay=True)
        syn_km.append(office_km if np.isfinite(office_km) else 0.0)
        agchem_km.append(office_km if np.isfinite(office_km) else 0.0)
        comp_loc_km.append(ck if np.isfinite(ck) else 0.0)
        comp_imp_km.append((300.0 + office_km) if np.isfinite(office_km) else 300.0)
        dig_km.append(dk if np.isfinite(dk) else 0.0)
        comp_n_conc.append(cn if pd.notna(cn) and cn > 0 else 0.037)
        comp_p_conc.append(cp if pd.notna(cp) and cp > 0 else np.nan)
        comp_k_conc.append(ckc if pd.notna(ckc) and ckc > 0 else np.nan)
        if dig_fac is not None:
            dig_n_conc.append(digest_n_conc_map.get((str(dig_fac["FACILITY"]), int(row["YEAR"])), digest_n_conc_default))
            dig_p_conc.append(digest_p_conc_map.get((str(dig_fac["FACILITY"]), int(row["YEAR"])), digest_p_conc_default))
        else:
            dig_n_conc.append(digest_n_conc_default)
            dig_p_conc.append(digest_p_conc_default)
        compost_facility.append(str(comp_fac["FACILITY"]) if comp_fac is not None and "FACILITY" in comp_fac else "")
        linked_digest_facility.append(str(dig_fac["FACILITY"]) if dig_fac is not None and "FACILITY" in dig_fac else "")
        use_digest_source = str(ctx.scenario.active_organic_n_source).upper() == "DIGEST"

        if dig_fac is not None and (use_digest_source or (row["CROP"] == "RICE" and int(row["CITY"]) == 213)):
            source_type.append("DIGEST")
        else:
            source_type.append("COMPOST")
    crop_df["SYN_KM"] = np.array(syn_km, dtype=float)
    crop_df["COMPOST_LOC_KM"] = np.array(comp_loc_km, dtype=float)
    crop_df["COMPOST_IMP_KM"] = np.array(comp_imp_km, dtype=float)
    crop_df["DIGEST_KM"] = np.array(dig_km, dtype=float)
    crop_df["AGROCHEM_KM"] = np.array(agchem_km, dtype=float)
    crop_df["ACTIVE_ORG_SOURCE"] = source_type
    crop_df["LINKED_COMPOST_FACILITY"] = compost_facility
    crop_df["LINKED_DIGEST_FACILITY"] = linked_digest_facility
    crop_df["COMPOST_N_CONC"] = np.array(comp_n_conc, dtype=float)
    crop_df["COMPOST_P_CONC"] = np.array(comp_p_conc, dtype=float)
    crop_df["COMPOST_K_CONC"] = np.array(comp_k_conc, dtype=float)
    crop_df["DIGEST_N_CONC"] = np.array(dig_n_conc, dtype=float)
    crop_df["DIGEST_P_CONC"] = np.array(dig_p_conc, dtype=float)

    crop_df["SYN_KG"] = np.where(crop_df["ORG_FLAG"] == "X", (crop_df["NIT"] + crop_df["PHO"] + crop_df["KAL"]) * 100.0 * crop_df["EXTENT_HA"], 0.0)
    crop_df["COMPOST_KG"] = 0.0
    crop_df["COMPOST_LOC_KG"] = 0.0
    crop_df["COMPOST_IMP_KG"] = 0.0
    crop_df["DIGEST_KG"] = 0.0
    crop_df["COMPOST_N_KG"] = 0.0
    crop_df["COMPOST_P_KG"] = 0.0
    crop_df["DIGEST_N_KG"] = 0.0
    crop_df["DIGEST_P_KG"] = 0.0
    crop_df["DIGEST_C_KG"] = 0.0
    crop_df["SYN_P_KG"] = 0.0
    crop_df["SYN_K_KG"] = 0.0


    crop_df = crop_df.reset_index(drop=True)
    demand_n = np.maximum(crop_df["N_REQ"], 0.0) * 100.0 * crop_df["EXTENT_HA"]
    n_gap = np.maximum(crop_df["N_REQ"] - crop_df["NIT"], 0.0) * 100.0 * crop_df["EXTENT_HA"]
    is_digest = crop_df["LINKED_DIGEST_FACILITY"].fillna("").astype(str).str.strip().ne("")
    org_y = crop_df["ORG_FLAG"].eq("Y")
    crop_df["ORG_N_DEMAND_KG"] = np.where(org_y, demand_n, n_gap)
    if is_alt_scenario(ctx.scenario.scenario_name):
        is_digest = pd.Series(False, index=crop_df.index)
        crop_df["ACTIVE_ORG_SOURCE"] = "COMPOST"
        digest_supply_n_by_year = {int(y): 0.0 for y in pd.unique(crop_df["YEAR"])}
        digest_supply_c_by_year = {int(y): 0.0 for y in pd.unique(crop_df["YEAR"])}

    compost_supply_n_by_year: dict[int, float] = {}
    compost_supply_n_by_fac_year: dict[tuple[str, int], float] = {}
    digest_supply_n_by_year: dict[int, float] = {}
    digest_supply_c_by_year: dict[int, float] = {}
    if manure_inventory is not None and not manure_inventory.empty:
        mi = manure_inventory.copy()
        mi["FACILITY"] = mi.get("FACILITY", "").astype(str)
        mi["YEAR"] = pd.to_numeric(mi["YEAR"], errors="coerce").fillna(0).astype(int)
        mi["TYPE"] = mi["TYPE"].astype(str).str.upper().str.strip()
        mi["COMPOST_N"] = pd.to_numeric(mi.get("COMPOST_N", 0.0), errors="coerce").fillna(0.0)
        mi["DIGEST_N"] = pd.to_numeric(mi.get("DIGEST_N", 0.0), errors="coerce").fillna(0.0)
        compost_rows = mi.loc[mi["TYPE"].eq("COMPOST")].copy()
        compost_supply_n_by_year = compost_rows.groupby("YEAR")["COMPOST_N"].sum().to_dict()
        compost_supply_n_by_fac_year = {
            (str(r["FACILITY"]), int(r["YEAR"])): float(r["COMPOST_N"])
            for _, r in compost_rows.groupby(["FACILITY", "YEAR"], as_index=False)["COMPOST_N"].sum().iterrows()
        }
        digest_supply_n_by_year = mi.loc[mi["TYPE"].eq("MWAD")].groupby("YEAR")["DIGEST_N"].sum().to_dict()
        digest_supply_c_by_year = mi.loc[mi["TYPE"].eq("MWAD")].groupby("YEAR")["DIGEST_C"].sum().to_dict()

    alloc_comp_n_loc = np.zeros(len(crop_df), dtype=float)
    alloc_comp_n_imp = np.zeros(len(crop_df), dtype=float)
    alloc_dig_n = np.zeros(len(crop_df), dtype=float)
    local_compost_alloc_detail: list[dict[str, float]] = [dict() for _ in range(len(crop_df))]
    remaining_compost_by_fac_year: dict[tuple[str, int], float] = {}

    def _target_organic_n_for_row(i: int) -> float:
        if crop_df.at[i, "ORG_FLAG"] == "Y":
            return max(float(crop_df.at[i, "ORG_N_DEMAND_KG"]), 0.0)
        extent_ha_i = max(float(crop_df.at[i, "EXTENT_HA"]), 0.0)
        n_req_total = max(float(crop_df.at[i, "N_REQ"]), 0.0) * 100.0 * extent_ha_i
        syn_n = max(float(crop_df.at[i, "NIT"]), 0.0) * 100.0 * extent_ha_i
        return max(n_req_total - syn_n, 0.0)

    for year in sorted(pd.unique(crop_df["YEAR"])):
        mask_year = crop_df["YEAR"].eq(year)
        year_idx = crop_df.index[mask_year].tolist()
        if not year_idx:
            continue
        rem_comp_by_fac = {fac: float(val) for (fac, y), val in compost_supply_n_by_fac_year.items() if int(y) == int(year)}
        rem_comp_year = float(compost_supply_n_by_year.get(int(year), 0.0))
        rem_dig = float(digest_supply_n_by_year.get(int(year), 0.0))

        def _take_local_compost_n(i: int, need_n: float) -> float:
            nonlocal rem_comp_year
            need = max(float(need_n), 0.0)
            if need <= 0:
                return 0.0
            taken_total = 0.0
            if rem_comp_by_fac:
                linked = str(crop_df.at[i, "LINKED_COMPOST_FACILITY"]).strip() if "LINKED_COMPOST_FACILITY" in crop_df.columns else ""
                ordered_facs: list[str] = []
                if linked and rem_comp_by_fac.get(linked, 0.0) > 1e-12:
                    ordered_facs.append(linked)
                ordered_facs.extend(
                    fac for fac, val in sorted(rem_comp_by_fac.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
                    if fac not in ordered_facs and float(val) > 1e-12
                )
                for fac in ordered_facs:
                    if need <= 1e-12:
                        break
                    available = max(float(rem_comp_by_fac.get(fac, 0.0)), 0.0)
                    if available <= 0:
                        continue
                    take = min(need, available)
                    rem_comp_by_fac[fac] = available - take
                    local_compost_alloc_detail[i][fac] = local_compost_alloc_detail[i].get(fac, 0.0) + take
                    taken_total += take
                    need -= take
                return taken_total

            take = min(need, max(rem_comp_year, 0.0))
            rem_comp_year -= take
            taken_total += take
            return taken_total

        year_y_dig_idx = crop_df.index[mask_year & crop_df["ORG_FLAG"].eq("Y") & is_digest].tolist()
        year_y_dig_idx = crop_df.loc[year_y_dig_idx].assign(
            _demand=crop_df.loc[year_y_dig_idx, "ORG_N_DEMAND_KG"]
        ).sort_values(["_demand", "KEY", "CROP", "CULTIVAR"], ascending=[False, True, True, True]).index.tolist()
        for i in year_y_dig_idx:
            target_n = _target_organic_n_for_row(i)
            if target_n <= 0 or rem_dig <= 0:
                continue
            take_n = min(target_n, rem_dig)
            alloc_dig_n[i] = take_n
            rem_dig -= take_n

        year_comp_idx = crop_df.index[mask_year].tolist()
        year_comp_idx = crop_df.loc[year_comp_idx].assign(
            _priority=crop_df.loc[year_comp_idx, "ORG_FLAG"].eq("Y").astype(int),
            _demand=crop_df.loc[year_comp_idx, "ORG_N_DEMAND_KG"],
        ).sort_values(["_priority", "_demand", "KEY", "CROP", "CULTIVAR"], ascending=[False, False, True, True, True]).index.tolist()
        for i in year_comp_idx:
            target_n = _target_organic_n_for_row(i)
            current_org_n = max(float(alloc_dig_n[i]), 0.0)
            take_n = _take_local_compost_n(i, max(target_n - current_org_n, 0.0))
            alloc_comp_n_loc[i] += take_n

        for i in year_comp_idx:
            target_n = _target_organic_n_for_row(i)
            current_org_n = (
                max(float(alloc_comp_n_loc[i]), 0.0)
                + max(float(alloc_comp_n_imp[i]), 0.0)
                + max(float(alloc_dig_n[i]), 0.0)
            )
            add_import_comp = max(target_n - current_org_n, 0.0)
            if add_import_comp > 0:
                alloc_comp_n_imp[i] += add_import_comp

        for fac, val in rem_comp_by_fac.items():
            remaining_compost_by_fac_year[(str(fac), int(year))] = max(float(val), 0.0)
    if is_alt_scenario(ctx.scenario.scenario_name):
        alloc_dig_n[:] = 0.0

    crop_df["LOCAL_COMPOST_ALLOC_N_JSON"] = [
        json.dumps({str(k): float(v) for k, v in detail.items() if float(v) > 1e-9}, ensure_ascii=False, sort_keys=True)
        for detail in local_compost_alloc_detail
    ]

    crop_df["COMPOST_N_KG"] = alloc_comp_n_loc + alloc_comp_n_imp
    crop_df["DIGEST_N_KG"] = alloc_dig_n
    digest_c_per_n_by_year = {
        int(year): (float(digest_supply_c_by_year.get(int(year), 0.0)) / float(digest_supply_n_by_year.get(int(year), 0.0)))
        if float(digest_supply_n_by_year.get(int(year), 0.0)) > 0 else 0.0
        for year in pd.unique(crop_df["YEAR"])
    }
    crop_df["DIGEST_C_KG"] = crop_df["DIGEST_N_KG"] * crop_df["YEAR"].map(digest_c_per_n_by_year).fillna(0.0)
    crop_df["COMPOST_LOC_KG"] = np.where(crop_df["COMPOST_N_CONC"] > 0, alloc_comp_n_loc / crop_df["COMPOST_N_CONC"], 0.0)
    crop_df["COMPOST_IMP_KG"] = alloc_comp_n_imp / 0.037
    crop_df["COMPOST_KG"] = crop_df["COMPOST_LOC_KG"] + crop_df["COMPOST_IMP_KG"]
    crop_df["COMPOST_LOC_KM"] = np.where(crop_df["COMPOST_LOC_KG"] > 0, crop_df["COMPOST_LOC_KM"], 0.0)
    crop_df["COMPOST_IMP_KM"] = np.where(crop_df["COMPOST_IMP_KG"] > 0, crop_df["COMPOST_IMP_KM"], 0.0)
    crop_df["DIGEST_KG"] = np.where(crop_df["DIGEST_N_CONC"] > 0, crop_df["DIGEST_N_KG"] / crop_df["DIGEST_N_CONC"], 0.0)
    crop_df["COMPOST_P_KG"] = (
        crop_df["COMPOST_LOC_KG"] * np.where(pd.notna(crop_df["COMPOST_P_CONC"]), crop_df["COMPOST_P_CONC"], 0.0)
        + crop_df["COMPOST_IMP_KG"] * np.where(pd.notna(crop_df["COMPOST_P_CONC"]), crop_df["COMPOST_P_CONC"], 0.0)
    )
    crop_df["DIGEST_P_KG"] = crop_df["DIGEST_KG"] * np.where(pd.notna(crop_df["DIGEST_P_CONC"]), crop_df["DIGEST_P_CONC"], 0.0)
    if is_alt_scenario(ctx.scenario.scenario_name):
        crop_df["DIGEST_N_KG"] = 0.0
        crop_df["DIGEST_P_KG"] = 0.0
        crop_df["DIGEST_C_KG"] = 0.0
        crop_df["DIGEST_KG"] = 0.0
        crop_df["DIGEST_KM"] = 0.0
    crop_df = crop_df.drop(columns=["ORG_N_DEMAND_KG"], errors="ignore")

    crop_df["AGROCHEM_KG"] = np.where(crop_df["ORG_FLAG"] == "X", crop_df["AGROCHEM"] * 100.0 * crop_df["EXTENT_HA"], 0.0)
    crop_df["AG_DIESEL_L"] = crop_df["DIESEL"] * 100.0 * crop_df["EXTENT_HA"] + crop_df["DIGEST_KG"] * 0.001 * 60.3 * 15/60 * 3.6 * 0.0262868597
    crop_df["AG_DIESEL_MJ"] = crop_df["AG_DIESEL_L"] / DIESEL_L_PER_MJ
    crop_df["AG_GASOL_L"] = crop_df["GASOL"] * 100.0 * crop_df["EXTENT_HA"]
    crop_df["AG_GASOL_MJ"] = crop_df["AG_GASOL_L"] / GASOL_L_PER_MJ
    crop_df["AG_KEROS_L"] = crop_df["KEROS"] * 100.0 * crop_df["EXTENT_HA"]
    crop_df["AG_KEROS_MJ"] = crop_df["AG_KEROS_L"] / KEROS_L_PER_MJ
    crop_df["AG_ELEC_KWH"] = crop_df["ELEC"] * 100.0 * crop_df["EXTENT_HA"]

    conv_mask = crop_df["ORG_FLAG"].eq("X")
    org_mask = crop_df["ORG_FLAG"].eq("Y")
    n_req_kg = np.maximum(crop_df["N_REQ"], 0.0) * 100.0 * crop_df["EXTENT_HA"]
    p_req_kg = np.maximum(crop_df["PHO"], 0.0) * 100.0 * crop_df["EXTENT_HA"]
    k_req_kg = np.maximum(crop_df["KAL"], 0.0) * 100.0 * crop_df["EXTENT_HA"]

    if "DIGEST_K_CONC" not in crop_df.columns:
        crop_df["DIGEST_K_CONC"] = 0.0

    comp_p_conc_eff = pd.to_numeric(crop_df["COMPOST_P_CONC"], errors="coerce").fillna(0.0).clip(lower=0.0)
    comp_k_conc_eff = pd.to_numeric(crop_df["COMPOST_K_CONC"], errors="coerce").fillna(0.0).clip(lower=0.0)
    dig_p_conc_eff = pd.to_numeric(crop_df["DIGEST_P_CONC"], errors="coerce").fillna(0.0).clip(lower=0.0)
    dig_k_conc_eff = pd.to_numeric(crop_df["DIGEST_K_CONC"], errors="coerce").fillna(0.0).clip(lower=0.0)

    crop_df["COMPOST_K_KG"] = crop_df["COMPOST_KG"] * comp_k_conc_eff
    crop_df["DIGEST_K_KG"] = crop_df["DIGEST_KG"] * dig_k_conc_eff

    org_comp_mask = org_mask
    org_dig_mask = org_mask & is_digest

    org_p_def = np.maximum(p_req_kg - crop_df["COMPOST_P_KG"] - crop_df["DIGEST_P_KG"], 0.0)
    org_k_def = np.maximum(k_req_kg - crop_df["COMPOST_K_KG"] - crop_df["DIGEST_K_KG"], 0.0)
    add_comp_kg_for_p = np.where(comp_p_conc_eff > 0, org_p_def / comp_p_conc_eff, 0.0)
    add_comp_kg_for_k = np.where(comp_k_conc_eff > 0, org_k_def / comp_k_conc_eff, 0.0)
    add_comp_kg_raw = np.where(org_comp_mask, np.maximum(add_comp_kg_for_p, add_comp_kg_for_k), 0.0)

    add_comp_loc_kg = np.zeros(len(crop_df), dtype=float)
    add_comp_imp_kg = np.zeros(len(crop_df), dtype=float)
    for i in np.where(add_comp_kg_raw > 1e-12)[0]:
        year_i = int(crop_df.at[i, "YEAR"])
        need_kg = max(float(add_comp_kg_raw[i]), 0.0)
        remaining_n_allowance_i = max(float(n_req_kg.iloc[i] if hasattr(n_req_kg, "iloc") else n_req_kg[i]) - float(crop_df.at[i, "COMPOST_N_KG"]) - float(crop_df.at[i, "DIGEST_N_KG"]), 0.0)
        comp_n_conc_i = max(float(crop_df.at[i, "COMPOST_N_CONC"]), 0.0)
        if need_kg <= 0 or remaining_n_allowance_i <= 0:
            continue

        linked = str(crop_df.at[i, "LINKED_COMPOST_FACILITY"]).strip() if "LINKED_COMPOST_FACILITY" in crop_df.columns else ""
        facs_y = {fac: val for (fac, y), val in remaining_compost_by_fac_year.items() if int(y) == year_i and val > 1e-12}
        ordered_facs: list[str] = []
        if linked and facs_y.get(linked, 0.0) > 1e-12:
            ordered_facs.append(linked)
        ordered_facs.extend(
            fac for fac, val in sorted(facs_y.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
            if fac not in ordered_facs and float(val) > 1e-12
        )
        if comp_n_conc_i > 0:
            for fac in ordered_facs:
                if need_kg <= 1e-12 or remaining_n_allowance_i <= 1e-12:
                    break
                available_n = max(float(remaining_compost_by_fac_year.get((fac, year_i), 0.0)), 0.0)
                take_kg = min(need_kg, available_n / comp_n_conc_i, remaining_n_allowance_i / comp_n_conc_i)
                if take_kg <= 0:
                    continue
                take_n = take_kg * comp_n_conc_i
                remaining_compost_by_fac_year[(fac, year_i)] = available_n - take_n
                local_compost_alloc_detail[i][fac] = local_compost_alloc_detail[i].get(fac, 0.0) + take_n
                add_comp_loc_kg[i] += take_kg
                need_kg -= take_kg
                remaining_n_allowance_i -= take_n

        take_imp_kg = min(need_kg, remaining_n_allowance_i / 0.037 if 0.037 > 0 else 0.0)
        if take_imp_kg > 0:
            add_comp_imp_kg[i] += take_imp_kg

    if np.any(add_comp_loc_kg > 0) or np.any(add_comp_imp_kg > 0):
        crop_df["COMPOST_LOC_KG"] = crop_df["COMPOST_LOC_KG"] + add_comp_loc_kg
        crop_df["COMPOST_IMP_KG"] = crop_df["COMPOST_IMP_KG"] + add_comp_imp_kg
        crop_df["COMPOST_KG"] = crop_df["COMPOST_LOC_KG"] + crop_df["COMPOST_IMP_KG"]
        crop_df["COMPOST_N_KG"] = crop_df["COMPOST_N_KG"] + add_comp_loc_kg * crop_df["COMPOST_N_CONC"] + add_comp_imp_kg * 0.037
        crop_df["COMPOST_P_KG"] = crop_df["COMPOST_P_KG"] + (add_comp_loc_kg + add_comp_imp_kg) * comp_p_conc_eff
        crop_df["COMPOST_K_KG"] = crop_df["COMPOST_K_KG"] + (add_comp_loc_kg + add_comp_imp_kg) * comp_k_conc_eff
        crop_df["COMPOST_LOC_KM"] = np.where(crop_df["COMPOST_LOC_KG"] > 0, crop_df["COMPOST_LOC_KM"], 0.0)
        crop_df["COMPOST_IMP_KM"] = np.where(crop_df["COMPOST_IMP_KG"] > 0, crop_df["COMPOST_IMP_KM"], 0.0)
        crop_df["LOCAL_COMPOST_ALLOC_N_JSON"] = [
            json.dumps({str(k): float(v) for k, v in detail.items() if float(v) > 1e-9}, ensure_ascii=False, sort_keys=True)
            for detail in local_compost_alloc_detail
        ]

    org_p_def = np.maximum(p_req_kg - crop_df["COMPOST_P_KG"] - crop_df["DIGEST_P_KG"], 0.0)
    org_k_def = np.maximum(k_req_kg - crop_df["COMPOST_K_KG"] - crop_df["DIGEST_K_KG"], 0.0)
    add_dig_kg_for_p = np.where(dig_p_conc_eff > 0, org_p_def / dig_p_conc_eff, 0.0)
    add_dig_kg_for_k = np.where(dig_k_conc_eff > 0, org_k_def / dig_k_conc_eff, 0.0)
    add_dig_kg = np.where(org_dig_mask, np.maximum(add_dig_kg_for_p, add_dig_kg_for_k), 0.0)
    if np.any(add_dig_kg > 0):
        old_digest_kg = crop_df["DIGEST_KG"].copy()
        old_digest_c_per_kg = safe_div(crop_df["DIGEST_C_KG"], old_digest_kg, 0.0)
        crop_df["DIGEST_KG"] = crop_df["DIGEST_KG"] + add_dig_kg
        crop_df["DIGEST_N_KG"] = crop_df["DIGEST_N_KG"] + add_dig_kg * crop_df["DIGEST_N_CONC"]
        crop_df["DIGEST_P_KG"] = crop_df["DIGEST_P_KG"] + add_dig_kg * dig_p_conc_eff
        crop_df["DIGEST_K_KG"] = crop_df["DIGEST_K_KG"] + add_dig_kg * dig_k_conc_eff
        crop_df["DIGEST_C_KG"] = crop_df["DIGEST_C_KG"] + add_dig_kg * old_digest_c_per_kg
        crop_df["DIGEST_KM"] = np.where(crop_df["DIGEST_KG"] > 0, crop_df["DIGEST_KM"], 0.0)

    crop_df["SYN_N_KG"] = np.where(
        conv_mask,
        np.maximum(n_req_kg - crop_df["COMPOST_N_KG"] - crop_df["DIGEST_N_KG"], 0.0),
        0.0,
    )
    crop_df["SYN_P_KG"] = np.where(
        conv_mask,
        np.maximum(p_req_kg - crop_df["COMPOST_P_KG"] - crop_df["DIGEST_P_KG"], 0.0),
        0.0,
    )
    crop_df["SYN_K_KG"] = np.where(
        conv_mask,
        np.maximum(k_req_kg - crop_df["COMPOST_K_KG"] - crop_df["DIGEST_K_KG"], 0.0),
        0.0,
    )
    crop_df["SYN_KG"] = crop_df["SYN_N_KG"] + crop_df["SYN_P_KG"] + crop_df["SYN_K_KG"]

    # emissions
    env_cols = ["KEY", "AIRTEMP", "WIND", *NH3_ENV_MONTH_COLS, "PRECIP", "TN", "SOILTEMP", "pH", "CEC", "SAND"]
    crop_df = crop_df.merge(env[env_cols], on="KEY", how="left")
    for col in ["AIRTEMP", "WIND", *NH3_ENV_MONTH_COLS, "PRECIP", "TN", "SOILTEMP", "pH", "CEC", "SAND"]:
        crop_df[col] = pd.to_numeric(crop_df[col], errors="coerce")
    pH = crop_df["pH"].fillna(6.0)
    cec_cmol = crop_df["CEC"].fillna(0.0) / 10.0
    fpH = 0.067 * (pH ** 2) - 0.69 * pH + 0.68
    fCEC = np.select([cec_cmol <= 6, (cec_cmol > 6) & (cec_cmol <= 24), (cec_cmol > 24) & (cec_cmol <= 32), cec_cmol > 32], [0.088, 0.012, 0.163, 0.0], default=0.0)
    air = crop_df["AIRTEMP"].fillna(15.0)
    wind = crop_df["WIND"].fillna(2.0)

    rice_air = crop_df["AIRTEMP_M05"].fillna(air) if "AIRTEMP_M05" in crop_df.columns else air
    rice_wind = crop_df["WIND_M05"].fillna(wind) if "WIND_M05" in crop_df.columns else wind
    alpha_rice = np.exp(0.233 * rice_air + 0.0419 * rice_wind) / np.exp(0.233 * air + 0.0419 * wind)
    syn_ef_rice = np.clip(alpha_rice * np.exp(fpH + fCEC + 0.014 - 1.895), 0.0, 0.95)
    dig_ef_rice = np.clip(alpha_rice * np.exp(fpH + fCEC + 0.995 - 1.895), 0.0, 0.95)

    tea_air = (crop_df[[c for c in ["AIRTEMP_M03", "AIRTEMP_M06", "AIRTEMP_M09"] if c in crop_df.columns]].fillna(air)).mean(axis=1)
    tea_wind = (crop_df[[c for c in ["WIND_M03", "WIND_M06", "WIND_M09"] if c in crop_df.columns]].fillna(wind)).mean(axis=1) if all(c in crop_df.columns for c in ["WIND_M03", "WIND_M06", "WIND_M09"]) else wind
    alpha_tea = np.exp(0.233 * tea_air + 0.0419 * tea_wind) / np.exp(0.233 * air + 0.0419 * wind)
    syn_ef_tea = np.clip(alpha_tea * np.exp(fpH + fCEC - 0.045 + 0.014 - 1.895), 0.0, 0.95)
    dig_ef_tea = np.clip(alpha_tea * np.exp(fpH + fCEC - 0.045 + 0.995 - 1.895), 0.0, 0.95)

    veg_air_vals = []
    veg_wind_vals = []

    def _row_weather_value(row: pd.Series, col: str, fallback_col: str, default: float) -> float:
        value = pd.to_numeric(row.get(col, np.nan), errors="coerce")
        if pd.notna(value) and np.isfinite(float(value)):
            return float(value)
        fallback = pd.to_numeric(row.get(fallback_col, default), errors="coerce")
        if pd.notna(fallback) and np.isfinite(float(fallback)):
            return float(fallback)
        return float(default)

    for _, row in crop_df.iterrows():
        cultivar = str(row["CULTIVAR"]).lower().strip()
        if cultivar == "spring":
            months = ["M03"]
        elif cultivar == "summer":
            months = ["M05"]
        elif cultivar == "autumn":
            months = ["M09"]
        elif cultivar == "winter":
            months = ["M11"]
        else:
            months = ["M05"]
        a = []
        w = []
        for m in months:
            a.append(_row_weather_value(row, f"AIRTEMP_{m}", "AIRTEMP", AIRTEMP_FALLBACK))
            w.append(_row_weather_value(row, f"WIND_{m}", "WIND", 2.0))
        veg_air_vals.append(float(np.mean(a)))
        veg_wind_vals.append(float(np.mean(w)))
    veg_air = np.asarray(veg_air_vals, dtype=float)
    veg_wind = np.asarray(veg_wind_vals, dtype=float)
    alpha_veg = np.exp(0.233 * veg_air + 0.0419 * veg_wind) / np.exp(0.233 * air + 0.0419 * wind)
    syn_ef_veg = np.clip(alpha_veg * np.exp(fpH + fCEC - 0.045 + 0.014 - 1.895), 0.0, 0.95)
    dig_ef_veg = np.clip(alpha_veg * np.exp(fpH + fCEC - 0.045 + 0.995 - 1.895), 0.0, 0.95)

    crop_df["NH3_KG"] = 0.0
    rice_mask = crop_df["CROP"] == "RICE"
    tea_mask = crop_df["CROP"] == "TEA"
    veg_mask = crop_df["CROP"] == "VEG"
    crop_df.loc[rice_mask, "NH3_KG"] = crop_df.loc[rice_mask, "SYN_N_KG"] * syn_ef_rice[rice_mask] + crop_df.loc[rice_mask, "COMPOST_N_KG"] * 0.0155 + crop_df.loc[rice_mask, "DIGEST_N_KG"] * dig_ef_rice[rice_mask]
    crop_df.loc[tea_mask, "NH3_KG"] = crop_df.loc[tea_mask, "SYN_N_KG"] * syn_ef_tea[tea_mask] + crop_df.loc[tea_mask, "COMPOST_N_KG"] * 0.013175 + crop_df.loc[tea_mask, "DIGEST_N_KG"] * dig_ef_tea[tea_mask]
    crop_df.loc[veg_mask, "NH3_KG"] = crop_df.loc[veg_mask, "SYN_N_KG"] * syn_ef_veg[veg_mask] + (0.002 * crop_df.loc[veg_mask, "COMPOST_N_KG"] / crop_df.loc[veg_mask, "EXTENT_HA"] + 0.64) * crop_df.loc[veg_mask, "EXTENT_HA"] + crop_df.loc[veg_mask, "DIGEST_N_KG"] * dig_ef_veg[veg_mask]

    total_n = crop_df["SYN_N_KG"] + crop_df["COMPOST_N_KG"] + crop_df["DIGEST_N_KG"]
    crop_df["NH3_KG"] = np.minimum(crop_df["NH3_KG"], total_n)

    crop_df["NO_KG"] = 0.0
    crop_df.loc[rice_mask, "NO_KG"] = total_n[rice_mask] * 0.0012
    crop_df.loc[tea_mask, "NO_KG"] = total_n[tea_mask] * 0.0154
    crop_df.loc[veg_mask, "NO_KG"] = crop_df.loc[veg_mask, "EXTENT_HA"] * 0.1228 * np.exp(
        0.3869
        * crop_df.loc[veg_mask, "SOILTEMP"].fillna(15.0)
        * safe_div(total_n[veg_mask] * 0.001, crop_df.loc[veg_mask, "EXTENT_HA"], 0.0)
    )

    crop_df["NRUNOFF_KG"] = 0.0
    crop_df["NLEACH_KG"] = 0.0

    extent_ha = crop_df["EXTENT_HA"].replace(0, np.nan)
    N_deposit_rice = crop_df["EXTENT_HA"] * 100.0 * 0.062
    irrig_term = np.maximum(26.1 * 113 - crop_df["PRECIP"].fillna(0.0), 0.0)
    irrig_vol = irrig_term * 0.001 * crop_df["EXTENT_HA"] * 10000.0
    irrig_N_input = irrig_vol * crop_df["TN"].fillna(0.0) * 0.001
    unit_N_rice = safe_div(total_n + N_deposit_rice + irrig_N_input - crop_df["NH3_KG"], crop_df["EXTENT_HA"], 0.0)
    residue_kg_for_leach = pd.to_numeric(crop_df["RESIDUE_KG"], errors="coerce").fillna(0.0) if "RESIDUE_KG" in crop_df.columns else pd.Series(0.0, index=crop_df.index)
    runleach_rice = np.where(
        residue_kg_for_leach > 0,
        np.exp(-11.73 + 0.79 * crop_df["SOILTEMP"].fillna(15.0) + 0.00793 * unit_N_rice + 0.000220 * irrig_term + 0.43),
        np.exp(-11.73 + 0.79 * crop_df["SOILTEMP"].fillna(15.0) + 0.00793 * unit_N_rice + 0.000220 * irrig_term),
    ) * crop_df["EXTENT_HA"]
    crop_df.loc[rice_mask, "NRUNOFF_KG"] = runleach_rice[rice_mask] * (1.0 / 7.0)
    crop_df.loc[rice_mask, "NLEACH_KG"] = runleach_rice[rice_mask] * (6.0 / 7.0)

    unit_N_tea = safe_div(total_n, crop_df["EXTENT_HA"], 0.0)
    runoff_tea = 1.07 + 0.00335 * unit_N_tea
    leach_tea = np.where(unit_N_tea < 178, 1.51 + 0.03292 * unit_N_tea, 7.37 + 0.25204 * (unit_N_tea - 178))
    crop_df.loc[tea_mask, "NRUNOFF_KG"] = runoff_tea[tea_mask] * crop_df.loc[tea_mask, "EXTENT_HA"]
    crop_df.loc[tea_mask, "NLEACH_KG"] = leach_tea[tea_mask] * crop_df.loc[tea_mask, "EXTENT_HA"]
    crop_df.loc[veg_mask, "NLEACH_KG"] = total_n[veg_mask] * 0.0007
    crop_df.loc[veg_mask, "NRUNOFF_KG"] = total_n[veg_mask] * 0.0007 * 11.7 / 56.1

    crop_df.loc[tea_mask | veg_mask, ["NRUNOFF_KG", "NLEACH_KG"]] = crop_df.loc[tea_mask | veg_mask, ["NRUNOFF_KG", "NLEACH_KG"]].clip(lower=0.0)

    rice_available_n = total_n + N_deposit_rice + irrig_N_input
    crop_df.loc[rice_mask, "NH3_KG"], crop_df.loc[rice_mask, "NO_KG"], crop_df.loc[rice_mask, "NRUNOFF_KG"] = cap_n_loss_pathways(
        crop_df.loc[rice_mask, "NH3_KG"],
        crop_df.loc[rice_mask, "NO_KG"],
        crop_df.loc[rice_mask, "NRUNOFF_KG"],
        rice_available_n[rice_mask] - crop_df.loc[rice_mask, "NLEACH_KG"],
    )
    crop_df.loc[rice_mask, "NH3_KG"], crop_df.loc[rice_mask, "NO_KG"], crop_df.loc[rice_mask, "NLEACH_KG"] = cap_n_loss_pathways(
        crop_df.loc[rice_mask, "NH3_KG"],
        crop_df.loc[rice_mask, "NO_KG"],
        crop_df.loc[rice_mask, "NLEACH_KG"],
        rice_available_n[rice_mask] - crop_df.loc[rice_mask, "NRUNOFF_KG"],
    )
    crop_df.loc[tea_mask | veg_mask, "NH3_KG"], crop_df.loc[tea_mask | veg_mask, "NO_KG"], crop_df.loc[tea_mask | veg_mask, "NRUNOFF_KG"] = cap_n_loss_pathways(
        crop_df.loc[tea_mask | veg_mask, "NH3_KG"],
        crop_df.loc[tea_mask | veg_mask, "NO_KG"],
        crop_df.loc[tea_mask | veg_mask, "NRUNOFF_KG"],
        total_n[tea_mask | veg_mask] - crop_df.loc[tea_mask | veg_mask, "NLEACH_KG"],
    )
    crop_df.loc[tea_mask | veg_mask, "NH3_KG"], crop_df.loc[tea_mask | veg_mask, "NO_KG"], crop_df.loc[tea_mask | veg_mask, "NLEACH_KG"] = cap_n_loss_pathways(
        crop_df.loc[tea_mask | veg_mask, "NH3_KG"],
        crop_df.loc[tea_mask | veg_mask, "NO_KG"],
        crop_df.loc[tea_mask | veg_mask, "NLEACH_KG"],
        total_n[tea_mask | veg_mask] - crop_df.loc[tea_mask | veg_mask, "NRUNOFF_KG"],
    )
    crop_df["IRRIG_TERM"] = irrig_term
    crop_df["SAND"] = pd.to_numeric(crop_df.get("SAND", 0.0), errors="coerce").fillna(0.0)

    crop_df = crop_df.sort_values(["KEY", "CROP", "CULTIVAR", "YEAR"]).copy()
    net_n = safe_div(total_n - crop_df["NH3_KG"], crop_df["EXTENT_HA"], 0.0)
    net_n_adj = np.where(crop_df["CULTIVAR"].eq("other"), net_n * (0.62 / 0.85), net_n)
    crop_df["RESIDUE_KG"] = 0.0
    crop_df.loc[rice_mask, "RESIDUE_KG"] = 1.075 * crop_df.loc[rice_mask, "EXTENT_HA"] * (-0.291 * (net_n_adj[rice_mask] ** 2) + 60.981 * net_n_adj[rice_mask] + 3636.4)
    crop_df["RESIDUE_KG"] = crop_df["RESIDUE_KG"].clip(lower=0.0)
    crop_df["RESIDUE_C_KG"] = crop_df["RESIDUE_KG"] * 0.374
    crop_df["RESIDUE_N_KG"] = crop_df["RESIDUE_KG"] * (0.835/1.075 * 0.00541 + 0.240/0.835 * 0.009)

    crop_df = crop_df.merge(manure_key_species, on=["KEY", "YEAR"], how="left")
    for col in ["MANURE_DAIRY", "MANURE_CATTLE", "MANURE_SWINE", "MANURE_CHICKEN", "MANURE_BROILER"]:
        crop_df[col] = pd.to_numeric(crop_df.get(col, 0.0), errors="coerce").fillna(0.0)

    out = crop_df.copy()
    for col in CROP_OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out["YEAR"] = pd.to_numeric(out["YEAR"], errors="coerce").fillna(0).astype(int)
    return out

KYOTO_CITIES = {101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111}
KYOTO_CHOICES = ["KYOTO_NORTH", "KYOTO_NE", "KYOTO_SOUTH"]
CITY_TO_FAC = {
    "FUKUCHIYAMA": [201], "MAIZURU": [202], "AYABE": [203],
    "SAKURAZUKA": [206, 213, 407], "KYOTANABE": [211], "MINEYAMA": [212],
    "ORII": [204, 207, 210, 322, 343, 344, 364, 365, 367],
    "HASEYAMA": [204, 207, 210, 322, 343, 344, 364, 365, 367],
    "KIZUGAWA": [214, 366], "OTOKUNI": [208, 209, 303],
    "MIYAZUYOSA": [205, 463, 465],
}
GROUP_A = {101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 206, 213, 407}
GROUP_B = {201, 202, 203, 205, 212, 463, 465}
GROUP_C = {204, 207, 208, 209, 210, 211, 214, 303, 322, 343, 344, 364, 365, 366, 367}
FAC_GROUPS = {
    "A": ["KYOTO_NORTH", "KYOTO_NE", "KYOTO_SOUTH", "SAKURAZUKA"],
    "B": ["FUKUCHIYAMA", "MAIZURU", "AYABE", "MIYAZUYOSA", "MINEYAMA"],
    "C": ["ORII", "HASEYAMA", "KIZUGAWA", "OTOKUNI", "KYOTANABE"],
}
BOUNDARY = {
    "KYOTO_NORTH": KYOTO_CITIES, "KYOTO_NE": KYOTO_CITIES, "KYOTO_SOUTH": KYOTO_CITIES,
    "FUKUCHIYAMA": {201}, "MAIZURU": {202}, "AYABE": {203},
    "SAKURAZUKA": {206, 213, 407}, "KYOTANABE": {211}, "MINEYAMA": {212},
    "ORII": {204, 207, 210, 322, 343, 344, 364, 365, 367},
    "HASEYAMA": {204, 207, 210, 322, 343, 344, 364, 365, 367},
    "KIZUGAWA": {214, 366}, "OTOKUNI": {208, 209, 303},
    "MIYAZUYOSA": {205, 463, 465},
}
RDF_CONV = 2.404
INC_ENERGY_L_per_KG = 0.00373
INC_ELEC_KWH_per_KG = 0.0264
DEFAULT_ASH_RATIO = 0.10


def linear_func(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * t


def fit_linear_ab(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) == 0:
        return 0.0, 0.0
    if len(np.unique(x_arr)) < 2:
        return float(np.nanmean(y_arr)), 0.0
    slope, intercept = np.polyfit(x_arr, y_arr, 1)
    return float(intercept), float(slope)


def fac_base(name: str) -> str:
    if not isinstance(name, str):
        return ""
    parts = name.split("_")
    while parts and (parts[-1] == "R" or parts[-1] == "ALT" or parts[-1] == "OPT" or parts[-1].isdigit() or parts[-1] in {"INC", "RDF", "TDAD", "MWAD", "COMPOST", "AD"}):
        parts = parts[:-1]
    return "_".join(parts)


def facilities_in_group(gname: str) -> list[str]:
    return FAC_GROUPS.get(gname, [])


def combine_energy_use(s: pd.Series) -> str:
    vals = [v for v in s.dropna().astype(str).str.upper().unique().tolist() if v and v != "NAN"]
    if not vals:
        return ""
    if "ELEC" in vals:
        return "ELEC"
    return vals[0]


def ceil_to_1000(x: float) -> float:
    if not np.isfinite(x) or x <= 0:
        return 0.0
    return float(int(np.ceil(x / 1000.0) * 1000.0))


def build_waste_projection(ctx: InventoryContext) -> pd.DataFrame:
    shared = build_shared_inputs(ctx)
    wastegen = shared["wastegen"].copy()
    pop = shared["population"].copy()
    for c in ["YEAR", "UNIT_WASTE", "FW_RATIO", "PAPER_RATIO", "PLASTIC_RATIO", "SYNTEX_RATIO", "NATTEX_RATIO", "WOOD_RATIO"]:
        if c in wastegen.columns:
            wastegen[c] = pd.to_numeric(wastegen[c], errors="coerce")
    for c in ["YEAR", "POPULATION", "CITY", "KCITY", "RCOM", "PREF"]:
        if c in pop.columns:
            pop[c] = pd.to_numeric(pop[c], errors="coerce")
    pop["KEY"] = pop["KEY"].map(normalize_code)

    uw_params = {}
    for city_name, g in wastegen.dropna(subset=["CITY_NAME"]).groupby("CITY_NAME", dropna=False):
        a, b = fit_linear_ab(g["YEAR"], g["UNIT_WASTE"])
        uw_params[str(city_name)] = (a, b)

    ratio_map = {}
    for city_name, g in wastegen.dropna(subset=["CITY_NAME"]).groupby("CITY_NAME", dropna=False):
        ratio_map[str(city_name)] = {
            "FW_RATIO": float(pd.to_numeric(g["FW_RATIO"], errors="coerce").mean()),
            "PAPER_RATIO": float(pd.to_numeric(g["PAPER_RATIO"], errors="coerce").mean()),
            "PLASTIC_RATIO": float(pd.to_numeric(g["PLASTIC_RATIO"], errors="coerce").mean()),
            "SYNTEX_RATIO": float(pd.to_numeric(g["SYNTEX_RATIO"], errors="coerce").mean()),
            "NATTEX_RATIO": float(pd.to_numeric(g["NATTEX_RATIO"], errors="coerce").mean()),
            "WOOD_RATIO": float(pd.to_numeric(g["WOOD_RATIO"], errors="coerce").mean()),
        }

    pop_rc = pop[(pop["CITY"] != 0) & (pop["RCOM"] != 0)].copy()
    rows = []
    for keys, g in pop_rc.groupby(["KEY", "PREF", "CITY", "KCITY", "RCOM", "CITY_NAME", "KCITY_NAME"], dropna=False):
        key, pref, city, kcity, rcom, city_name, kcity_name = keys
        a_p, b_p = fit_linear_ab(g["YEAR"], g["POPULATION"])
        pop_future = linear_func(np.asarray(ctx.years, dtype=float), a_p, b_p)
        pop_future = np.round(np.maximum(pop_future, 0.0)).astype(int)
        a_uw, b_uw = uw_params.get(str(city_name), (0.0, 0.0))
        uw_future = np.maximum(linear_func(np.asarray(ctx.years, dtype=float), a_uw, b_uw), 0.0)
        ratio = ratio_map.get(str(city_name), {k: 0.0 for k in ["FW_RATIO", "PAPER_RATIO", "PLASTIC_RATIO", "SYNTEX_RATIO", "NATTEX_RATIO", "WOOD_RATIO"]})
        total = pop_future * uw_future
        food = total * ratio["FW_RATIO"]
        paper = total * ratio["PAPER_RATIO"]
        plastic = total * ratio["PLASTIC_RATIO"]
        syntex = total * ratio["SYNTEX_RATIO"]
        nattex = total * ratio["NATTEX_RATIO"]
        wood = total * ratio["WOOD_RATIO"]
        for i, year in enumerate(ctx.years):
            rows.append({
                "KEY": str(key), "PREF": int(pref), "CITY": int(city), "KCITY": int(kcity), "RCOM": int(rcom),
                "CITY_NAME": city_name, "KCITY_NAME": kcity_name, "YEAR": int(year), "POPULATION": int(pop_future[i]),
                "WASTE_KG": float(np.round(total[i])), "FOOD_KG": float(np.round(food[i])), "PAPER_KG": float(np.round(paper[i])),
                "PLASTIC_KG": float(np.round(plastic[i])), "SYNTEX_KG": float(np.round(syntex[i])), "NATTEX_KG": float(np.round(nattex[i])),
                "WOOD_KG": float(np.round(wood[i])),
            })
    return pd.DataFrame(rows)



def build_alt_mwad_event_years(mswfac: pd.DataFrame) -> dict[str, int]:
    fac = mswfac.copy()
    fac["FACILITY"] = fac["FACILITY"].astype(str)
    fac["TYPE"] = fac["TYPE"].astype(str).str.upper().str.strip()
    fac["BASE"] = fac["FACILITY"].map(fac_base)
    out: dict[str, int] = {}
    for base, grp in fac.groupby("BASE", dropna=False):
        if str(base).upper() == "MIYAZUYOSA":
            continue
        non_mwad = grp[~grp["TYPE"].eq("MWAD")].copy()
        if non_mwad.empty:
            continue
        ev_years: list[int] = []
        for col in ["REBUILD_YEAR", "RENOV_YEAR"]:
            if col in non_mwad.columns:
                vals = pd.to_numeric(non_mwad[col], errors="coerce").dropna().astype(int)
                ev_years.extend([int(v) for v in vals if 2030 <= int(v) <= 2050])
        if ev_years:
            out[str(base)] = min(ev_years)
    return out


def build_base_xy_from_mswfac(mswfac: pd.DataFrame) -> dict[str, tuple[float, float]]:
    fac = mswfac.copy()
    fac["FACILITY"] = fac["FACILITY"].astype(str)
    fac["BASE"] = fac["FACILITY"].map(fac_base)
    fac["LON"] = pd.to_numeric(fac["LON"], errors="coerce")
    fac["LAT"] = pd.to_numeric(fac["LAT"], errors="coerce")
    grp = fac.dropna(subset=["LON", "LAT"]).groupby("BASE", as_index=False)[["LON", "LAT"]].mean()
    return {str(r["BASE"]): (float(r["LON"]), float(r["LAT"])) for _, r in grp.iterrows()}


def map_city_to_alt_base(
    city: int,
    candidate_bases: list[str],
    base_xy: dict[str, tuple[float, float]],
    lon: float = np.nan,
    lat: float = np.nan,
) -> str | None:
    if not candidate_bases:
        return None
    if len(candidate_bases) == 1:
        return candidate_bases[0]
    if np.isfinite(lon) and np.isfinite(lat):
        best, best_km = None, np.inf
        for b in candidate_bases:
            if b not in base_xy:
                continue
            blon, blat = base_xy[b]
            km = haversine_km(lon, lat, blon, blat)
            if km < best_km:
                best, best_km = b, km
        if best is not None:
            return best
    return candidate_bases[0]


def build_alt_boundary_manure_2050(
    ctx: InventoryContext,
    manure_inventory: pd.DataFrame | None,
    event_years: dict[str, int],
    base_xy: dict[str, tuple[float, float]],
) -> dict[str, float]:
    if manure_inventory is None or manure_inventory.empty:
        return {}
    shared = build_shared_inputs(ctx)
    fac_info = shared["manurefac"][["FACILITY", "CITY", "LON", "LAT"]].drop_duplicates().copy()
    fac_info["FACILITY"] = fac_info["FACILITY"].astype(str)
    fac_info["CITY"] = pd.to_numeric(fac_info["CITY"], errors="coerce")
    fac_info["LON"] = pd.to_numeric(fac_info["LON"], errors="coerce")
    fac_info["LAT"] = pd.to_numeric(fac_info["LAT"], errors="coerce")

    mi = manure_inventory.copy()
    mi["FACILITY"] = mi["FACILITY"].astype(str)
    mi["YEAR"] = pd.to_numeric(mi.get("YEAR", 0), errors="coerce").fillna(0).astype(int)
    mi["TYPE"] = mi.get("TYPE", "").astype(str).str.upper().str.strip() if "TYPE" in mi.columns else ""
    mi["MANURE_KG"] = pd.to_numeric(mi.get("MANURE_KG", 0.0), errors="coerce").fillna(0.0)
    mi = mi[mi["YEAR"] == 2050].copy()
    if mi.empty:
        return {}

    out_rows = []


    if "TYPE" in mi.columns:
        direct_alt = mi[(mi["TYPE"] == "MWAD") & (mi["FACILITY"].isin(list(event_years.keys())))].copy()
        if not direct_alt.empty:
            direct_alt = direct_alt.groupby("FACILITY", as_index=False)["MANURE_KG"].sum()
            for _, r in direct_alt.iterrows():
                out_rows.append({"ALT_BASE": str(r["FACILITY"]), "MANURE_KG": float(r["MANURE_KG"])})

        remaining = mi.loc[~((mi["TYPE"] == "MWAD") & (mi["FACILITY"].isin(list(event_years.keys()))))].copy()
    else:
        remaining = mi.copy()

    if not remaining.empty:
        remaining = remaining.groupby("FACILITY", as_index=False)["MANURE_KG"].sum()
        df = remaining.merge(fac_info, on="FACILITY", how="left")
        for _, r in df.iterrows():
            city = pd.to_numeric(r.get("CITY", np.nan), errors="coerce")
            if pd.isna(city):
                continue
            cands = [b for b, cities in BOUNDARY.items() if int(city) in {int(x) for x in cities} and b in event_years]
            if not cands:
                continue
            alt_base = map_city_to_alt_base(
                int(city), cands, base_xy,
                lon=float(r["LON"]) if pd.notna(r["LON"]) else np.nan,
                lat=float(r["LAT"]) if pd.notna(r["LAT"]) else np.nan,
            )
            if alt_base is not None:
                out_rows.append({"ALT_BASE": alt_base, "MANURE_KG": float(r["MANURE_KG"])})

    if not out_rows:
        return {}
    out_df = pd.DataFrame(out_rows)
    out_df["MANURE_KG"] = pd.to_numeric(out_df["MANURE_KG"], errors="coerce").fillna(0.0)
    return out_df.groupby("ALT_BASE", as_index=False)["MANURE_KG"].sum().set_index("ALT_BASE")["MANURE_KG"].to_dict()


def apply_alt_manure_rerouting(ctx: InventoryContext, manure_df: pd.DataFrame, mwad_module) -> pd.DataFrame:
    if manure_df.empty or not is_alt_scenario(ctx.scenario.scenario_name):
        return manure_df
    shared = build_shared_inputs(ctx)
    mswfac = shared["mswfac"].copy()
    manurefac = shared["manurefac"].copy()
    event_years = build_alt_mwad_event_years(mswfac)
    if not event_years:
        return manure_df
    base_xy = build_base_xy_from_mswfac(mswfac)
    fac_info = manurefac[["FACILITY", "CITY", "LON", "LAT"]].drop_duplicates().copy()
    fac_info["FACILITY"] = fac_info["FACILITY"].astype(str)
    fac_info["CITY"] = pd.to_numeric(fac_info["CITY"], errors="coerce")
    fac_info["LON"] = pd.to_numeric(fac_info["LON"], errors="coerce")
    fac_info["LAT"] = pd.to_numeric(fac_info["LAT"], errors="coerce")
    fac_to_alt_base: dict[str, str | None] = {}
    for _, r in fac_info.iterrows():
        city = pd.to_numeric(r["CITY"], errors="coerce")
        if pd.isna(city):
            fac_to_alt_base[str(r["FACILITY"])] = None
            continue
        cands = [b for b, cities in BOUNDARY.items() if int(city) in {int(x) for x in cities} and b in event_years]
        fac_to_alt_base[str(r["FACILITY"])] = map_city_to_alt_base(
            int(city), cands, base_xy,
            lon=float(r["LON"]) if pd.notna(r["LON"]) else np.nan,
            lat=float(r["LAT"]) if pd.notna(r["LAT"]) else np.nan,
        )
    df = manure_df.copy()
    df["FACILITY"] = df["FACILITY"].astype(str)
    df["TYPE"] = df["TYPE"].astype(str).str.upper().str.strip()
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").fillna(0).astype(int)
    moved_rows = []
    for idx, row in df.iterrows():
        if row["TYPE"] != "COMPOST":
            continue
        target_base = fac_to_alt_base.get(str(row["FACILITY"]))
        if not target_base:
            continue
        event_year = event_years.get(target_base)

        if event_year is None or int(row["YEAR"]) < int(event_year) + 2:
            continue
        moved = row.copy()
        moved["FACILITY"] = target_base
        moved["TYPE"] = "MWAD"
        for col in ["COMPOST_KG","COMPOST_N","COMPOST_P2O5","COMPOST_USED_KG","RESIDUE_REQ_KG","RESIDUE_SUPPLY_KG","RESIDUE_IMPORT_KG","RESIDUE_SUPPLIED_KM","RESIDUE_IMPORT_KM"]:
            if col in moved.index:
                moved[col] = 0.0
        moved_rows.append(moved)
        for col in [
            "MANURE_KG","MANURE_N","DAIRY_MANURE_KG","CATTLE_MANURE_KG","SWINE_MANURE_KG","CHICKEN_MANURE_KG","BROILER_MANURE_KG",
            "RESIDUE_REQ_KG","RESIDUE_SUPPLY_KG","RESIDUE_IMPORT_KG","RESIDUE_SUPPLIED_KM","RESIDUE_IMPORT_KM",
            "COMPOST_KG","COMPOST_N","COMPOST_P2O5","DIGEST_KG","DIGEST_C","DIGEST_N","DIGEST_P2O5","BIOGAS_CH4_NM3",
            "MWAD_ELEC_USE_KWH","MWAD_HEAT_USE_MJ","MWAD_WATER_USE_M3","MWAD_REFUSED_KG"
        ]:
            if col in df.columns:
                df.at[idx, col] = 0.0
    if moved_rows:
        moved_df = pd.DataFrame(moved_rows)
        group_cols = ["FACILITY","YEAR","TYPE"]
        num_cols = [c for c in moved_df.columns if c not in group_cols]
        moved_df[num_cols] = moved_df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        moved_df = moved_df.groupby(group_cols, as_index=False)[num_cols].sum()
        if mwad_module is not None:
            for idx in moved_df.index:
                manure_kg = float(moved_df.at[idx, "MANURE_KG"])
                if manure_kg <= 0:
                    continue
                try:
                    res = mwad_module.mwad_run(food_waste_amount=manure_kg, capacity_amount=manure_kg)
                except Exception:
                    res = {}
                moved_df.at[idx, "DIGEST_KG"] = float(res.get("MWAD_DIGESTATE", 0.0))
                moved_df.at[idx, "DIGEST_C"] = float(res.get("MWAD_DIG_ORGC", 0.0))
                moved_df.at[idx, "DIGEST_N"] = float(res.get("MWAD_DIG_N", 0.0))
                moved_df.at[idx, "DIGEST_P2O5"] = float(res.get("MWAD_DIG_P", 0.0))
                moved_df.at[idx, "BIOGAS_CH4_NM3"] = float(res.get("MWAD_BIOGAS", 0.0))
                moved_df.at[idx, "MWAD_ELEC_USE_KWH"] = float(res.get("MWAD_ELEC_REQ", 0.0))
                moved_df.at[idx, "MWAD_HEAT_USE_MJ"] = float(res.get("MWAD_HEAT_REQ", 0.0))
                moved_df.at[idx, "MWAD_WATER_USE_M3"] = float(res.get("MWAD_WATER_REQ", 0.0))
                moved_df.at[idx, "MWAD_REFUSED_KG"] = float(res.get("MWAD_REFUSED", 0.0))
        df = pd.concat([df, moved_df], ignore_index=True)
    group_cols = ["FACILITY","YEAR","TYPE"]
    num_cols = [c for c in df.columns if c not in group_cols]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df = df.groupby(group_cols, as_index=False)[num_cols].sum()
    zero_manure = pd.to_numeric(df.get("MANURE_KG", 0.0), errors="coerce").fillna(0.0) <= 0
    for col in ["RESIDUE_REQ_KG", "RESIDUE_SUPPLY_KG", "RESIDUE_IMPORT_KG", "RESIDUE_SUPPLIED_KM", "RESIDUE_IMPORT_KM"]:
        if col in df.columns:
            df.loc[zero_manure, col] = 0.0
    df = zero_mwad_compost_residue_fields(df)
    return finalize_output_schema(clean_small_numeric(df), MANURE_OUTPUT_COLUMNS)

def prepare_msw_facilities(ctx: InventoryContext, waste_proj: pd.DataFrame | None = None, manure_inventory: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    shared = build_shared_inputs(ctx)
    df_fac = shared["mswfac"].copy()
    for c in ["LAT", "LON", "CAPACITY", "RENOV_YEAR", "REBUILD_YEAR", "CAP_YEAR", "ASH_RATIO", "CONST_YEAR", "MWAD_YEAR", "GEN_EFF"]:
        if c in df_fac.columns:
            df_fac[c] = pd.to_numeric(df_fac[c], errors="coerce")
    if "ENERGY_USE" not in df_fac.columns:
        df_fac["ENERGY_USE"] = ""
    df_fac["TYPE"] = df_fac["TYPE"].astype(str).str.upper().str.strip()
    df_fac["FACILITY"] = df_fac["FACILITY"].astype(str)
    df_fac["BASE"] = df_fac["FACILITY"].map(fac_base)
    df_fac["ASH_RATIO"] = pd.to_numeric(df_fac.get("ASH_RATIO", DEFAULT_ASH_RATIO), errors="coerce").fillna(DEFAULT_ASH_RATIO)

    if is_alt_scenario(ctx.scenario.scenario_name) and waste_proj is not None and manure_inventory is not None:
        event_years = build_alt_mwad_event_years(df_fac)
        base_xy_map = build_base_xy_from_mswfac(df_fac)
        manure2050_by_base = build_alt_boundary_manure_2050(ctx, manure_inventory, event_years, base_xy_map)
        base_xy_tmp = pd.DataFrame([{"FACILITY_BASE": k, "LON": v[0], "LAT": v[1]} for k, v in base_xy_map.items()])
        key_meta_map = build_waste_key_meta(ctx, waste_proj, base_xy_tmp) if not base_xy_tmp.empty else {}
        mwad_rows = []
        for base, event_year in event_years.items():
            if base == "MIYAZUYOSA":
                continue
            base_facs = df_fac[df_fac["BASE"] == base]
            if base_facs.empty:
                continue
            ref = base_facs.iloc[0].copy()
            wy = waste_proj[waste_proj["YEAR"] == int(event_year)].copy()
            if wy.empty:
                food_kg = 0.0
            else:
                if base in {"KYOTO_NORTH", "KYOTO_NE", "KYOTO_SOUTH", "ORII", "HASEYAMA"}:
                    wy["PRIMARY_BASE"] = wy["KEY"].astype(str).map(lambda k: key_meta_map.get(str(k), {}).get("PRIMARY_BASE"))
                    food_kg = float(wy.loc[wy["PRIMARY_BASE"] == base, "FOOD_KG"].sum())
                else:
                    cities = {int(c) for c in BOUNDARY.get(base, set())}
                    food_kg = float(wy.loc[wy["CITY"].astype(int).isin(cities), "FOOD_KG"].sum())
            manure_kg = float(manure2050_by_base.get(base, 0.0))
            capacity = ceil_to_1000(food_kg + manure_kg)
            new = ref.copy()
            new["FACILITY"] = f"{base}_ALT_AD"
            new["TYPE"] = "MWAD"
            new["CAPACITY"] = capacity
            new["CONST_YEAR"] = int(event_year) + 2
            new["REBUILD_YEAR"] = np.nan
            new["RENOV_YEAR"] = np.nan
            new["ENERGY_USE"] = os.environ.get("MWAD_BIOGAS_UTIL", "ELEC").strip().upper()
            new["BASE"] = base
            mwad_rows.append(new)
        if mwad_rows:
            df_fac = pd.concat([df_fac, pd.DataFrame(mwad_rows)], ignore_index=True)

    base_xy = (df_fac.dropna(subset=["LON", "LAT"]).groupby("BASE")[["LON", "LAT"]].first().reset_index().rename(columns={"BASE": "FACILITY_BASE"}))

    r_rows = []
    for _, r in df_fac.iterrows():
        unit_name = str(r["FACILITY"])
        ry = pd.to_numeric(r.get("REBUILD_YEAR", np.nan), errors="coerce")
        rny = pd.to_numeric(r.get("RENOV_YEAR", np.nan), errors="coerce")
        evy, ev = np.nan, ""
        if np.isfinite(ry) and (2030 <= int(ry) <= 2050):
            evy, ev = int(ry), "REBUILD"
        elif np.isfinite(rny) and (2030 <= int(rny) <= 2050):
            evy, ev = int(rny), "RENEW"
        else:
            continue
        r_rows.append({"FACILITY": f"{unit_name}_R", "TYPE": r["TYPE"], "CAPACITY": r["CAPACITY"], "EVENT_YEAR": evy, "EVENT": ev, "ENERGY_USE": "ELEC", "BASE": r["BASE"], "LAT": r["LAT"], "LON": r["LON"], "ASH_RATIO": r["ASH_RATIO"], "GEN_EFF": r.get("GEN_EFF", np.nan), "CONST_YEAR": r.get("CONST_YEAR", np.nan), "RENOV_YEAR": np.nan, "REBUILD_YEAR": np.nan})

    def unit_shutdown_years(row: pd.Series) -> set[int]:
        ys: set[int] = set()
        for ev in ["RENOV_YEAR", "REBUILD_YEAR"]:
            val = pd.to_numeric(row.get(ev, np.nan), errors="coerce")
            if np.isfinite(val):
                y = int(val)
                if 2030 <= y <= 2050:
                    ys |= {y, y + 1}
        return ys

    df_fac_unit = df_fac.copy()
    df_fac_unit["SHUT_SET"] = df_fac_unit.apply(unit_shutdown_years, axis=1)
    if df_fac_unit["FACILITY"].duplicated().any():
        dup = df_fac_unit.loc[df_fac_unit["FACILITY"].duplicated(), "FACILITY"].astype(str).unique().tolist()[:10]
        log(f"Duplicate MSW unit names found after overlay; keeping first occurrence for routing map: {dup}")
        df_fac_unit = df_fac_unit.drop_duplicates("FACILITY", keep="first").copy()
    fac_unit_map = df_fac_unit.set_index("FACILITY", drop=False).to_dict("index")
    return df_fac, base_xy, fac_unit_map


def build_waste_key_meta(ctx: InventoryContext, waste_proj: pd.DataFrame, base_xy: pd.DataFrame) -> dict[str, dict[str, Any]]:
    shared = build_shared_inputs(ctx)
    key_cent = shared["key_centroids"].copy()
    key_cent["KEY"] = key_cent["KEY"].map(normalize_code)
    cent_map = {str(r["KEY"]): (float(r["CENT_LON"]), float(r["CENT_LAT"])) for _, r in key_cent.iterrows()}
    base_loc = {str(r["FACILITY_BASE"]): (float(r["LON"]), float(r["LAT"])) for _, r in base_xy.iterrows()}
    city2cands: dict[int, set[str]] = {}
    for b, cities in CITY_TO_FAC.items():
        for c in cities:
            city2cands.setdefault(int(c), set()).add(b)

    def dist_key_to_base(key: str, base: str) -> float:
        if key not in cent_map or base not in base_loc:
            return np.nan
        lon, lat = cent_map[key]
        blon, blat = base_loc[base]
        return get_distance_km(ctx, f"KEY:{key}", lon, lat, f"BASE:{base}", blon, blat)

    def primary_base_for_key(key: str, city: int) -> str | None:
        if int(city) in KYOTO_CITIES:
            best, bestd = None, np.inf
            for b in KYOTO_CHOICES:
                d = dist_key_to_base(key, b)
                if np.isfinite(d) and d < bestd:
                    best, bestd = b, d
            return best
        cands = sorted(city2cands.get(int(city), set()))
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        best, bestd = None, np.inf
        for b in cands:
            d = dist_key_to_base(key, b)
            if np.isfinite(d) and d < bestd:
                best, bestd = b, d
        return best

    meta = {}
    for _, r in waste_proj[["KEY", "PREF", "CITY"]].drop_duplicates().iterrows():
        key = str(r["KEY"])
        city = int(r["CITY"])
        prim = primary_base_for_key(key, city)
        dkp = dist_key_to_base(key, prim) if prim else np.nan
        meta[key] = {"PREF": int(r["PREF"]), "CITY": city, "PRIMARY_BASE": prim, "D_KEY_PRIM": dkp}
    return meta


def build_rebuilt_alloc_unit(ctx: InventoryContext, df_fac: pd.DataFrame, waste_proj: pd.DataFrame, key_meta_map: dict[str, dict[str, Any]]) -> dict[str, float]:
    def primary_series(df: pd.DataFrame) -> pd.Series:
        return df["KEY"].astype(str).map(lambda k: key_meta_map.get(str(k), {}).get("PRIMARY_BASE", None))

    if is_alt_scenario(ctx.scenario.scenario_name):
        pass

    rebuilt_base_target = {}
    for base in df_fac["BASE"].unique():
        sub = df_fac[df_fac["BASE"] == base]
        rebuild_year = pd.to_numeric(sub.get("REBUILD_YEAR", np.nan), errors="coerce").min()
        cap_year = pd.to_numeric(sub.get("CAP_YEAR", np.nan), errors="coerce").min()
        if not (np.isfinite(rebuild_year) and np.isfinite(cap_year)):
            continue
        cy = int(cap_year)
        wcy = waste_proj[waste_proj["YEAR"] == cy].copy()
        if wcy.empty:
            continue
        wcy["PRIMARY_BASE"] = primary_series(wcy)
        if is_alt_scenario(ctx.scenario.scenario_name):
            shared_boundary_bases = {"KYOTO_NORTH", "KYOTO_NE", "KYOTO_SOUTH", "HASEYAMA", "ORII"}
            if base in shared_boundary_bases:
                base_waste = (wcy.loc[wcy["PRIMARY_BASE"] == base, "WASTE_KG"] - wcy.loc[wcy["PRIMARY_BASE"] == base, "FOOD_KG"]).sum()
            else:
                cities = BOUNDARY.get(base, set())
                if cities:
                    m = wcy["CITY"].astype(int).isin({int(c) for c in cities})
                    base_waste = (wcy.loc[m, "WASTE_KG"] - wcy.loc[m, "FOOD_KG"]).sum()
                else:
                    base_waste = (wcy.loc[wcy["PRIMARY_BASE"] == base, "WASTE_KG"] - wcy.loc[wcy["PRIMARY_BASE"] == base, "FOOD_KG"]).sum()
        else:
            base_waste = wcy.loc[wcy["PRIMARY_BASE"] == base, "WASTE_KG"].sum()
        rebuilt_base_target[base] = float(max(base_waste, 0.0))

    rebuilt_alloc_unit = {}
    for base, target in rebuilt_base_target.items():
        sub_all = df_fac[df_fac["BASE"] == base].copy()
        sub = sub_all[sub_all["TYPE"].astype(str).str.upper().isin(["INC", "RDF"])].copy()
        if sub.empty:
            sub = sub_all.copy()
        total = pd.to_numeric(sub["CAPACITY"], errors="coerce").fillna(0.0).sum()
        if total <= 0:
            for _, r in sub.iterrows():
                rebuilt_alloc_unit[str(r["FACILITY"])] = ceil_to_1000(float(target) / max(len(sub), 1))
        else:
            for _, r in sub.iterrows():
                rebuilt_alloc_unit[str(r["FACILITY"])] = ceil_to_1000(float(target) * float(r.get("CAPACITY", 0.0)) / float(total))
    return rebuilt_alloc_unit


def build_facility_year_rows(ctx: InventoryContext, df_fac: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_fac.iterrows():
        fac = str(r["FACILITY"])
        base = str(r["BASE"])
        typ = str(r["TYPE"]).upper()
        energy_use = str(r.get("ENERGY_USE", "") or "").upper()
        renov_y = pd.to_numeric(r.get("RENOV_YEAR", np.nan), errors="coerce")
        rebuild_y = pd.to_numeric(r.get("REBUILD_YEAR", np.nan), errors="coerce")
        const_y = pd.to_numeric(r.get("CONST_YEAR", np.nan), errors="coerce")
        for y in ctx.years:
            event = ""
            effective = fac
            if np.isfinite(rebuild_y) and y == int(rebuild_y):
                event = "REBUILD"
            elif np.isfinite(renov_y) and y == int(renov_y):
                event = "RENEW"
            if np.isfinite(rebuild_y) and y >= int(rebuild_y) + 2:
                effective = f"{fac}_R"
                energy_use_y = "ELEC" if typ in {"INC", "RDF", "TDAD"} else energy_use
            elif np.isfinite(renov_y) and y >= int(renov_y) + 2:
                effective = f"{fac}_R"
                energy_use_y = "ELEC" if typ in {"INC", "RDF", "TDAD"} else energy_use
            else:
                energy_use_y = energy_use
            if typ == "MWAD" and np.isfinite(const_y) and y == int(const_y):
                event = "NEW"
            rows.append({
                "UNIT_NAME": fac, "FACILITY": effective, "BASE": base, "TYPE": typ, "YEAR": int(y), "EVENT": event,
                "ENERGY_USE": energy_use_y, "LAT": r.get("LAT", np.nan), "LON": r.get("LON", np.nan),
                "CAPACITY_RAW": pd.to_numeric(r.get("CAPACITY", 0.0), errors="coerce"),
                "CONST_YEAR": const_y, "RENOV_YEAR": renov_y, "REBUILD_YEAR": rebuild_y,
            })
    return pd.DataFrame(rows)


def build_waste_routing(ctx: InventoryContext, manure_inventory: pd.DataFrame | None = None) -> pd.DataFrame:
    waste_proj = build_waste_projection(ctx)
    df_fac, base_xy, fac_unit_map = prepare_msw_facilities(ctx, waste_proj=waste_proj, manure_inventory=manure_inventory)
    key_meta_map = build_waste_key_meta(ctx, waste_proj, base_xy)
    rebuilt_alloc_unit = build_rebuilt_alloc_unit(ctx, df_fac, waste_proj, key_meta_map)

    def unit_shutdown_years(row: dict[str, Any], year: int) -> bool:
        shut = set()
        for ev in ["RENOV_YEAR", "REBUILD_YEAR"]:
            val = pd.to_numeric(row.get(ev, np.nan), errors="coerce")
            if np.isfinite(val):
                y = int(val)
                if 2030 <= y <= 2050:
                    shut |= {y, y + 1}
        return int(year) in shut

    def unit_capacity_limit(row: dict[str, Any], year: int) -> float:
        cap = float(pd.to_numeric(row.get("CAPACITY", 0.0), errors="coerce")) if "CAPACITY" in row else 0.0
        ry = pd.to_numeric(row.get("REBUILD_YEAR", np.nan), errors="coerce")
        cy = pd.to_numeric(row.get("CAP_YEAR", np.nan), errors="coerce")
        fac = str(row.get("FACILITY", ""))
        if np.isfinite(ry) and np.isfinite(cy) and year >= int(cy) and fac in rebuilt_alloc_unit:
            cap = float(rebuilt_alloc_unit[fac])
        typ = str(row.get("TYPE", "")).upper()
        if typ == "INC":
            cap *= 0.90
        return max(cap, 0.0)

    base_units: dict[str, list[dict[str, Any]]] = {}
    for _, r in df_fac.iterrows():
        base_units.setdefault(str(r["BASE"]), []).append(r.to_dict())

    def _event_year_from_unit(row: dict[str, Any]) -> int | None:
        years_ev: list[int] = []
        for col in ["REBUILD_YEAR", "RENOV_YEAR"]:
            val = pd.to_numeric(row.get(col, np.nan), errors="coerce")
            if np.isfinite(val) and 2030 <= int(val) <= 2050:
                years_ev.append(int(val))
        return min(years_ev) if years_ev else None

    def build_event_city_year_lookup_from_units() -> dict[int, int]:
        out: dict[int, int] = {}
        for u in df_fac.to_dict("records"):
            ey = _event_year_from_unit(u)
            if ey is None:
                continue
            base = str(u.get("BASE", fac_base(str(u.get("FACILITY", "")))))
            cities = BOUNDARY.get(base, None)
            if not cities:
                city_val = pd.to_numeric(u.get("CITY", np.nan), errors="coerce")
                cities = {int(city_val)} if np.isfinite(city_val) else set()
            for city in cities:
                city_i = int(city)
                out[city_i] = min(out.get(city_i, int(ey)), int(ey))
        return out

    event_city_year = build_event_city_year_lookup_from_units()

    def event_food_no_inc_active(city: Any, year: int) -> bool:
        try:
            city_i = int(float(city))
        except Exception:
            return False
        ey = event_city_year.get(city_i)
        return ey is not None and int(year) >= int(ey)

    base_loc = {str(r["FACILITY_BASE"]): (float(r["LON"]), float(r["LAT"])) for _, r in base_xy.iterrows()}

    def base_to_base_km(a: str, b: str) -> float:
        if a not in base_loc or b not in base_loc:
            return np.nan
        lon1, lat1 = base_loc[a]
        lon2, lat2 = base_loc[b]
        return get_distance_km(ctx, f"BASE:{a}", lon1, lat1, f"BASE:{b}", lon2, lat2)

    def _unit_available(u: dict[str, Any], year: int, typ: str) -> bool:
        if unit_shutdown_years(u, year):
            return False
        if typ in {"MWAD", "TDAD"}:
            cy = pd.to_numeric(u.get("CONST_YEAR", np.nan), errors="coerce")
            if np.isfinite(cy) and year < int(cy):
                return False
        return True

    def pick_unit(base: str, year: int, q: float, cap_rem: dict[str, float], typeset: set[str] | None = None) -> str | None:
        cand = []
        for u in base_units.get(base, []):
            typ = str(u.get("TYPE", "")).upper()
            if typeset is not None and typ not in typeset:
                continue
            if typeset is None and typ == "MWAD":
                continue
            if not _unit_available(u, year, typ):
                continue
            uname = str(u["FACILITY"])
            if cap_rem.get(uname, 0.0) >= q:
                cand.append(uname)
        if not cand:
            return None
        cand.sort(key=lambda n: cap_rem.get(n, 0.0), reverse=True)
        return cand[0]

    def pick_unit_any_capacity(base: str, year: int, cap_rem: dict[str, float], typeset: set[str]) -> str | None:
        cand = []
        for u in base_units.get(base, []):
            typ = str(u.get("TYPE", "")).upper()
            if typ not in typeset:
                continue
            if not _unit_available(u, year, typ):
                continue
            uname = str(u["FACILITY"])
            if cap_rem.get(uname, 0.0) > 1e-9:
                cand.append(uname)
        if not cand:
            return None
        cand.sort(key=lambda n: cap_rem.get(n, 0.0), reverse=True)
        return cand[0]

    def pick_nearest_unit_any_capacity(prim: str, group_name: str, year: int, cap_rem: dict[str, float], typeset: set[str]) -> tuple[str | None, float]:
        unit = pick_unit_any_capacity(prim, year, cap_rem, typeset)
        if unit is not None:
            return unit, 0.0
        best_d, best_u = np.inf, None
        for b in facilities_in_group(group_name):
            if b == prim:
                continue
            u = pick_unit_any_capacity(b, year, cap_rem, typeset)
            if u is None:
                continue
            d = base_to_base_km(prim, b)
            if np.isfinite(d) and d < best_d:
                best_d, best_u = d, u
        return best_u, (best_d if best_u is not None else 0.0)

    def pick_full_unit_by_type_priority(prim: str, group_name: str, year: int, q: float, cap_rem: dict[str, float], type_priority: list[str]) -> tuple[str | None, float]:
        for typ in type_priority:
            unit = pick_unit(prim, year, q, cap_rem, {typ})
            if unit is not None:
                return unit, 0.0
            best_d, best_u = np.inf, None
            for b in facilities_in_group(group_name):
                if b == prim:
                    continue
                u = pick_unit(b, year, q, cap_rem, {typ})
                if u is None:
                    continue
                d = base_to_base_km(prim, b)
                if np.isfinite(d) and d < best_d:
                    best_d, best_u = d, u
            if best_u is not None:
                return best_u, best_d
        return None, 0.0

    rows = []
    for y in ctx.years:
        wy = waste_proj[waste_proj["YEAR"] == y].copy()
        if wy.empty:
            continue
        cap_rem = {str(r["FACILITY"]): unit_capacity_limit(r.to_dict(), y) for _, r in df_fac.iterrows()}
        for gname, gset in [("A", GROUP_A), ("B", GROUP_B), ("C", GROUP_C)]:
            wgy = wy[wy["CITY"].isin(list(gset))].copy().sort_values("WASTE_KG", ascending=False)
            for _, r in wgy.iterrows():
                key = str(r["KEY"])
                prim = key_meta_map.get(key, {}).get("PRIMARY_BASE")
                dkp = float(key_meta_map.get(key, {}).get("D_KEY_PRIM", np.nan))
                if prim is None:
                    continue
                total = float(r["WASTE_KG"])
                food = float(r["FOOD_KG"])
                paper = float(r["PAPER_KG"])
                wood = float(r["WOOD_KG"])
                plastic = float(r["PLASTIC_KG"])
                syntex = float(r["SYNTEX_KG"])
                nattex = float(r["NATTEX_KG"])

                if food > 0:
                    if is_alt_scenario(ctx.scenario.scenario_name):
                        cand_unit = f"{prim}_ALT_AD"
                        if cand_unit in cap_rem:
                            urow_tmp = fac_unit_map.get(cand_unit, {})
                            if _unit_available(urow_tmp, y, "MWAD") and cap_rem.get(cand_unit, 0.0) > 1e-9:
                                mwad_unit, extra_dist = cand_unit, 0.0
                            else:
                                mwad_unit, extra_dist = None, 0.0
                        else:
                            mwad_unit, extra_dist = None, 0.0
                    else:
                        mwad_unit, extra_dist = pick_nearest_unit_any_capacity(prim, gname, y, cap_rem, {"MWAD"})
                    if mwad_unit is not None:
                        urow = fac_unit_map.get(mwad_unit, {})
                        take = min(food, cap_rem.get(mwad_unit, 0.0))
                        if take > 0:
                            cap_rem[mwad_unit] -= take
                            rows.append({
                                "FROM_KEY": key, "FROM_CITY": int(r["CITY"]),
                                "FACILITY": fac_base(mwad_unit), "UNIT_FACILITY": mwad_unit, "YEAR": y, "TYPE": "MWAD",
                                "CAPACITY": unit_capacity_limit(urow, y), "WASTE_KG": take, "WASTE_KM": dkp + extra_dist,
                                "PLASTIC_KG": 0.0, "SYNTEX_KG": 0.0, "FOOD_KG": take, "PAPER_KG": 0.0, "NATTEX_KG": 0.0, "WOOD_KG": 0.0
                            })
                            total -= take
                            food -= take

                tdad_demand = max(food, 0.0) + max(paper, 0.0) + max(wood, 0.0)
                tdad_unit = None
                if tdad_demand > 0:
                    tdad_unit, tdad_extra_dist = pick_nearest_unit_any_capacity(prim, gname, y, cap_rem, {"TDAD"})
                    tdad_dist = dkp + tdad_extra_dist
                    if tdad_unit is not None:
                        cap_avail = cap_rem.get(tdad_unit, 0.0)
                        take_food = min(food, cap_avail)
                        rem = cap_avail - take_food
                        take_paper = min(paper, rem)
                        rem -= take_paper
                        take_wood = min(wood, rem)
                        tdad_take = take_food + take_paper + take_wood
                        if tdad_take > 0:
                            cap_rem[tdad_unit] -= tdad_take
                            urow = fac_unit_map.get(tdad_unit, {})
                            rows.append({
                                "FROM_KEY": key, "FROM_CITY": int(r["CITY"]),
                                "FACILITY": fac_base(tdad_unit), "UNIT_FACILITY": tdad_unit, "YEAR": y, "TYPE": "TDAD",
                                "CAPACITY": unit_capacity_limit(urow, y), "WASTE_KG": tdad_take, "WASTE_KM": tdad_dist,
                                "PLASTIC_KG": 0.0, "SYNTEX_KG": 0.0, "FOOD_KG": take_food, "PAPER_KG": take_paper, "NATTEX_KG": 0.0, "WOOD_KG": take_wood
                            })
                            total -= tdad_take
                            food -= take_food
                            paper -= take_paper
                            wood -= take_wood

                if food > 1e-9 and event_food_no_inc_active(r.get("CITY", np.nan), y):
                    rows.append({
                        "FROM_KEY": key, "FROM_CITY": int(r["CITY"]),
                        "FACILITY": f"{prim}_FOOD_COMPOST", "UNIT_FACILITY": f"{prim}_FOOD_COMPOST",
                        "YEAR": y, "TYPE": "COMPOST",
                        "CAPACITY": food, "WASTE_KG": food, "WASTE_KM": dkp,
                        "PLASTIC_KG": 0.0, "SYNTEX_KG": 0.0, "FOOD_KG": food,
                        "PAPER_KG": 0.0, "NATTEX_KG": 0.0, "WOOD_KG": 0.0,
                    })
                    total -= food
                    food = 0.0

                q = total
                if q <= 0:
                    continue
                local_unit = pick_unit(prim, y, q, cap_rem, {"INC", "RDF"})
                if local_unit is not None:
                    unit, extra_dist = local_unit, 0.0
                else:
                    unit, extra_dist = pick_full_unit_by_type_priority(prim, gname, y, q, cap_rem, ["INC", "RDF"])
                dist = dkp + extra_dist
                if unit is None:
                    continue
                cap_rem[unit] -= q
                urow = fac_unit_map.get(unit, {})
                rows.append({
                    "FROM_KEY": key, "FROM_CITY": int(r["CITY"]),
                    "FACILITY": fac_base(unit), "UNIT_FACILITY": unit, "YEAR": y, "TYPE": str(urow.get("TYPE", "")).upper(),
                    "CAPACITY": unit_capacity_limit(urow, y), "WASTE_KG": q, "WASTE_KM": dist,
                    "PLASTIC_KG": plastic, "SYNTEX_KG": syntex, "FOOD_KG": food, "PAPER_KG": paper, "NATTEX_KG": nattex, "WOOD_KG": wood
                })

    if not rows:
        try:
            write_waste_link_from_raw(ctx, pd.DataFrame(), manure_inventory)
        except Exception as exc:
            log(f"Could not write empty 02_waste_link.csv: {exc}")
        return finalize_output_schema(pd.DataFrame(), WASTE_OUTPUT_COLUMNS)

    df = pd.DataFrame(rows)
    try:
        write_waste_link_from_raw(ctx, df, manure_inventory)
    except Exception as exc:
        log(f"Could not write 02_waste_link.csv: {exc}")
    df["WT_KM"] = pd.to_numeric(df["WASTE_KG"], errors="coerce").fillna(0.0) * pd.to_numeric(df["WASTE_KM"], errors="coerce").fillna(0.0)
    agg = df.groupby(["FACILITY", "YEAR", "TYPE"], as_index=False).agg({
        "CAPACITY": "sum",
        "WASTE_KG": "sum", "WT_KM": "sum", "PLASTIC_KG": "sum", "SYNTEX_KG": "sum", "FOOD_KG": "sum", "PAPER_KG": "sum", "NATTEX_KG": "sum", "WOOD_KG": "sum",
    })
    agg["WASTE_KM"] = np.where(agg["WASTE_KG"] > 0, agg["WT_KM"] / agg["WASTE_KG"], 0.0)
    agg = agg.drop(columns=["WT_KM"])

    schedule_rows = []
    df_fac_sched = df_fac[~df_fac["TYPE"].astype(str).str.upper().eq("LANDFILL")].copy()
    grouped = df_fac_sched.groupby(["BASE", "TYPE"], dropna=False)
    for (base, typ), grp in grouped:
        typ = str(typ).upper()
        base = str(base)

        event_candidates: list[tuple[int, str]] = []
        new_years: list[int] = []
        for _, rr in grp.iterrows():
            ry = pd.to_numeric(rr.get("REBUILD_YEAR", np.nan), errors="coerce")
            ny = pd.to_numeric(rr.get("RENOV_YEAR", np.nan), errors="coerce")
            cy = pd.to_numeric(rr.get("CONST_YEAR", np.nan), errors="coerce")
            if np.isfinite(ry) and 2030 <= int(ry) <= 2050:
                event_candidates.append((int(ry), "REBUILD"))
            if np.isfinite(ny) and 2030 <= int(ny) <= 2050:
                event_candidates.append((int(ny), "RENEW"))
            if typ == "MWAD" and np.isfinite(cy) and 2030 <= int(cy) <= 2050:
                new_years.append(int(cy))

        event_candidates = sorted(event_candidates, key=lambda x: (x[0], 0 if x[1] == "REBUILD" else 1))
        earliest_event_year = event_candidates[0][0] if event_candidates else None
        earliest_event_name = event_candidates[0][1] if event_candidates else ""
        earliest_new = min(new_years) if new_years else None

        base_energy = combine_energy_use(pd.Series(grp["ENERGY_USE"]))

        for y in ctx.years:
            active_caps = []
            for _, rr in grp.iterrows():
                ry = pd.to_numeric(rr.get("REBUILD_YEAR", np.nan), errors="coerce")
                ny = pd.to_numeric(rr.get("RENOV_YEAR", np.nan), errors="coerce")
                cy = pd.to_numeric(rr.get("CONST_YEAR", np.nan), errors="coerce")

                shut = False
                if np.isfinite(ry) and y in {int(ry), int(ry) + 1}:
                    shut = True
                if np.isfinite(ny) and y in {int(ny), int(ny) + 1}:
                    shut = True
                if typ == "MWAD" and np.isfinite(cy) and y < int(cy):
                    shut = True

                if not shut:
                    active_caps.append(unit_capacity_limit(rr.to_dict(), y))

            if typ in {"INC", "RDF", "TDAD"}:
                if base_energy == "ELEC":
                    energy = "ELEC"
                elif earliest_event_year is not None and y >= earliest_event_year + 2:
                    energy = "ELEC"
                else:
                    energy = base_energy
            else:
                energy = base_energy

            event = ""
            if typ == "MWAD" and earliest_new is not None and y == earliest_new:
                event = "NEW"
            elif earliest_event_year is not None and y == earliest_event_year:
                event = earliest_event_name

            schedule_rows.append({
                "FACILITY": base,
                "YEAR": int(y),
                "TYPE": typ,
                "CAPACITY_META": float(np.sum(active_caps)) if active_caps else 0.0,
                "EVENT_META": event,
                "ENERGY_USE_META": energy,
            })

    fac_year_summary = pd.DataFrame(schedule_rows)
    if not fac_year_summary.empty:
        agg = agg.merge(fac_year_summary, on=["FACILITY", "YEAR", "TYPE"], how="outer")
        for col in ["WASTE_KG", "WASTE_KM", "PLASTIC_KG", "SYNTEX_KG", "FOOD_KG", "PAPER_KG", "NATTEX_KG", "WOOD_KG"]:
            if col in agg.columns:
                agg[col] = pd.to_numeric(agg[col], errors="coerce").fillna(0.0)
        agg["CAPACITY"] = pd.to_numeric(agg["CAPACITY_META"], errors="coerce").fillna(pd.to_numeric(agg.get("CAPACITY", 0.0), errors="coerce").fillna(0.0))
        agg["EVENT"] = pd.Series(agg.get("EVENT_META", "")).fillna("").astype(str)
        agg["ENERGY_USE"] = pd.Series(agg.get("ENERGY_USE_META", "")).fillna("").astype(str)
        agg = agg.drop(columns=["CAPACITY_META", "EVENT_META", "ENERGY_USE_META"], errors="ignore")
    else:
        agg["EVENT"] = ""
        agg["ENERGY_USE"] = ""
    for col in [c for c in WASTE_OUTPUT_COLUMNS if c not in agg.columns]:
        agg[col] = 0.0 if col not in {"FACILITY", "TYPE", "EVENT", "ENERGY_USE", "YEAR"} else ("" if col in {"FACILITY", "TYPE", "EVENT", "ENERGY_USE"} else 0)
    agg["YEAR"] = pd.to_numeric(agg["YEAR"], errors="coerce").fillna(0).astype(int)
    agg["EVENT"] = agg["EVENT"].fillna("").astype(str)
    agg["ENERGY_USE"] = agg["ENERGY_USE"].fillna("").astype(str)
    agg = clean_small_numeric(agg)
    return finalize_output_schema(agg, WASTE_OUTPUT_COLUMNS)

def load_ad_lookup_for_inventory(ctx: InventoryContext) -> pd.DataFrame:
    for p in [ctx.paths.output_dir / "06_ad_lookup.csv", ctx.paths.dataset_dir / "06_ad_lookup.csv"]:
        if p.exists():
            try:
                df = pd.read_csv(p)
                if not df.empty and {"TYPE", "SCALE_T_DAY", "CAPACITY_KG_YEAR"}.issubset(df.columns):
                    return df
            except Exception:
                continue
    return pd.DataFrame()


def lookup_ad_reaction(ad_lookup: pd.DataFrame, typ: str, feed_kg: float, capacity_kg: float, shares: dict[str, float] | None = None) -> dict[str, float]:
    if ad_lookup.empty or feed_kg <= 0:
        return {}
    sub = ad_lookup[ad_lookup["TYPE"].astype(str).str.upper().eq(str(typ).upper())].copy()
    if sub.empty:
        return {}
    scale_target = max(float(capacity_kg), float(feed_kg)) / 365000.0
    sub["_SCALE_DIFF"] = (pd.to_numeric(sub.get("SCALE_T_DAY", 0.0), errors="coerce").fillna(0.0) - scale_target).abs()
    sub["_MIX_DIFF"] = 0.0
    if shares:
        for c in ["FOOD_SHARE", "PAPER_SHARE", "WOOD_SHARE", "MANURE_SHARE"]:
            if c not in sub.columns:
                sub[c] = 0.0
            sub["_MIX_DIFF"] += (pd.to_numeric(sub[c], errors="coerce").fillna(0.0) - float(shares.get(c, 0.0))) ** 2
    r = sub.sort_values(["_SCALE_DIFF", "_MIX_DIFF"]).iloc[0]
    base_cap = float(pd.to_numeric(r.get("CAPACITY_KG_YEAR", 0.0), errors="coerce") or 0.0)
    ratio = float(feed_kg) / base_cap if base_cap > 0 else 0.0
    def val(col: str) -> float:
        return float(pd.to_numeric(r.get(col, 0.0), errors="coerce") or 0.0) * ratio
    prefix = str(typ).upper()
    return {
        f"{prefix}_ELEC_REQ": val("ELEC_KWH"),
        f"{prefix}_HEAT_REQ": val("HEAT_MJ"),
        f"{prefix}_WATER_REQ": val("WATER_M3"),
        f"{prefix}_BIOGAS": val("BIOGAS_NM3"),
        f"{prefix}_DIGESTATE": val("DIGEST_KG"),
        f"{prefix}_DIG_ORGC": val("DIGEST_C_KG"),
        f"{prefix}_DIG_N": val("DIGEST_N_KG"),
        f"{prefix}_DIG_P2O5": val("DIGEST_P2O5_KG"),
        f"{prefix}_DIG_K2O": val("DIGEST_K2O_KG"),
        f"{prefix}_REFUSED": val("REFUSE_KG"),
        f"{prefix}_CAPEX": val("CAPEX"),
    }

def build_waste_inventory(ctx: InventoryContext, manure_inventory: pd.DataFrame | None = None) -> pd.DataFrame:
    direct = build_waste_routing(ctx, manure_inventory=manure_inventory)
    direct = direct[~direct["TYPE"].astype(str).str.upper().eq("LANDFILL")].copy()
    if direct.empty:
        return finalize_output_schema(pd.DataFrame(), WASTE_OUTPUT_COLUMNS)

    shared = build_shared_inputs(ctx)
    df_fac, base_xy, fac_unit_map = prepare_msw_facilities(ctx, waste_proj=None if manure_inventory is None else build_waste_projection(ctx), manure_inventory=manure_inventory)
    fac_master = shared["mswfac"].copy()
    fac_master["FACILITY"] = fac_master["FACILITY"].astype(str)
    fac_master["TYPE"] = fac_master["TYPE"].astype(str).str.upper().str.strip()
    for c in ["LAT", "LON", "CAPACITY", "RENOV_YEAR", "REBUILD_YEAR", "CAP_YEAR", "GEN_EFF", "ASH_RATIO", "CONST_YEAR", "MWAD_YEAR"]:
        if c in fac_master.columns:
            fac_master[c] = pd.to_numeric(fac_master[c], errors="coerce")
    if "ENERGY_USE" not in fac_master.columns:
        fac_master["ENERGY_USE"] = ""
    fac_master["BASE"] = fac_master["FACILITY"].map(fac_base)
    fac_master["ASH_RATIO"] = pd.to_numeric(fac_master.get("ASH_RATIO", DEFAULT_ASH_RATIO), errors="coerce").fillna(DEFAULT_ASH_RATIO)
    base_loc = {str(r["FACILITY_BASE"]): (float(r["LON"]), float(r["LAT"])) for _, r in base_xy.iterrows()}

    def load_tdad_module(script_dir: Path | None):
        module_path = _resolve_module_path(script_dir, "TDAD_module.py")
        if module_path is None:
            return None
        spec = importlib.util.spec_from_file_location("TDAD_module_dynamic", module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def unit_shutdown_years(row: dict[str, Any], year: int) -> bool:
        shut = set()
        for ev in ["RENOV_YEAR", "REBUILD_YEAR"]:
            val = pd.to_numeric(row.get(ev, np.nan), errors="coerce")
            if np.isfinite(val):
                y = int(val)
                if 2030 <= y <= 2050:
                    shut |= {y, y + 1}
        return int(year) in shut

    def unit_capacity_limit(row: dict[str, Any], year: int) -> float:
        cap = float(pd.to_numeric(row.get("CAPACITY", 0.0), errors="coerce")) if "CAPACITY" in row else 0.0
        return max(cap, 0.0)

    def available_inc_bases(year: int) -> dict[str, float]:
        caps: dict[str, float] = {}
        for _, r in df_fac.iterrows():
            typ = str(r.get("TYPE", "")).upper()
            if typ != "INC":
                continue
            rd = r.to_dict()
            if unit_shutdown_years(rd, year):
                continue
            base = str(r.get("BASE", fac_base(str(r.get("FACILITY", "")))))
            caps[base] = caps.get(base, 0.0) + unit_capacity_limit(rd, year) * 0.90
        return caps

    def nearest_base_km(from_base: str, to_bases: list[str]) -> tuple[str | None, float]:
        if from_base not in base_loc:
            return None, np.nan
        lon1, lat1 = base_loc[from_base]
        best_base, best_km = None, np.inf
        for b in to_bases:
            if b not in base_loc:
                continue
            lon2, lat2 = base_loc[b]
            km = get_distance_km(ctx, f"BASE:{from_base}", lon1, lat1, f"BASE:{b}", lon2, lat2)
            if np.isfinite(km) and km < best_km:
                best_base, best_km = b, km
        return best_base, (float(best_km) if np.isfinite(best_km) else np.nan)

    direct = direct.copy()
    numeric_direct = [c for c in direct.columns if c not in {"FACILITY", "TYPE", "EVENT", "ENERGY_USE", "YEAR"}]
    direct[numeric_direct] = direct[numeric_direct].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    tdad_module = load_tdad_module(ctx.paths.script_dir)
    mwad_module = load_mwad_module(ctx.paths.script_dir)
    ad_lookup = load_ad_lookup_for_inventory(ctx)

    for c in ["INC_INPUT_KG", "INC_KEROS_L", "INC_ELEC_KWH", "ASH_KG", "ASH_KM",
              "TDAD_ELEC_KWH", "TDAD_HEAT_MJ", "TDAD_BIOGAS_NM3", "TDAD_DIGEST_KG", "TDAD_REFUSE_KG", "TDAD_REFUSE_KM", "TDAD_DIGSOLID_KG",
              "RDF_KG", "RDF_KM", "RDF_REFUSE_KG", "RDF_REFUSE_KM",
              "MWAD_ELEC_KWH", "MWAD_HEAT_MJ", "MWAD_WATER_M3", "MWAD_BIOGAS_NM3", "MWAD_DIGEST_KG", "MWAD_DIGSOLID_KG",
              "MWAD_DIGEST_ORGC_KG", "MWAD_DIGEST_N_KG", "MWAD_DIGEST_P2O5_KG", "MWAD_DIGEST_K2O_KG", "MWAD_REFUSE_KG", "MWAD_REFUSE_KM",
              "INC_CAPEX", "TDAD_CAPEX", "RDF_CAPEX", "MWAD_CAPEX",
              "ELEC_INC_KWH", "ELEC_TDAD_KWH", "ELEC_RDF_KWH", "ELEC_MWAD_KWH"]:
        if c not in direct.columns:
            direct[c] = 0.0

    transfer_rows: list[dict[str, Any]] = []

    for idx, row in direct.iterrows():
        typ = str(row["TYPE"]).upper()
        fac = str(row["FACILITY"])
        year = int(row["YEAR"])
        base = fac_base(fac)
        if typ == "TDAD":
            wt = float(row["WASTE_KG"])
            food_kg = max(float(row.get("FOOD_KG", 0.0)), 0.0)
            paper_kg = max(float(row.get("PAPER_KG", 0.0)), 0.0)
            wood_kg = max(float(row.get("WOOD_KG", 0.0)), 0.0)
            cap_amount = max(float(row.get("CAPACITY", wt)), wt)
            energy_use = str(row.get("ENERGY_USE", "")).upper().strip()

            if tdad_module is not None and hasattr(tdad_module, "tdad_run"):
                mix = {}
                if food_kg > 0:
                    mix["FOOD_WASTE"] = food_kg
                if paper_kg > 0:
                    mix["PAPER_WASTE"] = paper_kg
                if wood_kg > 0:
                    mix["WOOD_WASTE"] = wood_kg
                try:
                    res = tdad_module.tdad_run(
                        mix if mix else wt,
                        avg_air_temp=AIRTEMP_FALLBACK,
                        capacity_amount=cap_amount,
                        Biogas_griduse=energy_use in {"GRIDGAS", "GASGRID"},
                    )
                except Exception:
                    res = {}

                refuse_kg = float(res.get("TDAD_REFUSED", 0.0))
                digest_kg = float(res.get("TDAD_DIGESTATE", 0.0))
                digts_kg = float(res.get("TDAD_DIG_TS", 0.0))
                digsolid_kg = float(res.get("TDAD_digsolid", 0.0)) * 1000.0
                if digsolid_kg <= 0.0 and digts_kg > 0.0:
                    digsolid_kg = digts_kg * TDAD_DIGSOLID_TS_RECOVERY / (1.0 - TDAD_DIGSOLID_WATER)
                biogas = float(res.get("TDAD_BIOGAS", 0.0))
                elec = float(res.get("TDAD_ELEC_REQ", 0.0))
                heat = float(res.get("TDAD_HEAT_REQ", 0.0))
                water = float(res.get("TDAD_WATER_REQ", 0.0))

                direct.at[idx, "TDAD_ELEC_KWH"] = elec
                direct.at[idx, "TDAD_HEAT_MJ"] = heat
                direct.at[idx, "TDAD_BIOGAS_NM3"] = biogas
                direct.at[idx, "TDAD_DIGEST_KG"] = digest_kg
                direct.at[idx, "TDAD_REFUSE_KG"] = refuse_kg
                direct.at[idx, "TDAD_DIGSOLID_KG"] = digsolid_kg
                if "TDAD_WATER_M3" in direct.columns:
                    direct.at[idx, "TDAD_WATER_M3"] = water

                if energy_use not in {"GRIDGAS", "GASGRID"}:
                    direct.at[idx, "ELEC_TDAD_KWH"] = BIOGAS_HV * biogas * 0.35 / KJ_PER_KWH

                transfer_rows.append({"YEAR": year, "FROM_FACILITY": fac, "FROM_BASE": base, "KG": refuse_kg, "KM_FIELD": "TDAD_REFUSE_KM", "COMP": {"FOOD": refuse_kg}})
                transfer_rows.append({"YEAR": year, "FROM_FACILITY": fac, "FROM_BASE": base, "KG": digsolid_kg, "KM_FIELD": None, "COMP": {"FOOD": digsolid_kg}})
                continue

            if tdad_module is None:
                total_mix = max(food_kg + paper_kg + wood_kg, 0.0)
                shares = {"FOOD_SHARE": float(safe_div(food_kg, total_mix)), "PAPER_SHARE": float(safe_div(paper_kg, total_mix)), "WOOD_SHARE": float(safe_div(wood_kg, total_mix)), "MANURE_SHARE": 0.0}
                res_lu = lookup_ad_reaction(ad_lookup, "TDAD", wt, cap_amount, shares)
                if res_lu:
                    direct.at[idx, "TDAD_ELEC_KWH"] = float(res_lu.get("TDAD_ELEC_REQ", 0.0))
                    direct.at[idx, "TDAD_HEAT_MJ"] = float(res_lu.get("TDAD_HEAT_REQ", 0.0))
                    direct.at[idx, "TDAD_BIOGAS_NM3"] = float(res_lu.get("TDAD_BIOGAS", 0.0))
                    direct.at[idx, "TDAD_DIGEST_KG"] = float(res_lu.get("TDAD_DIGESTATE", 0.0))
                    direct.at[idx, "TDAD_REFUSE_KG"] = float(res_lu.get("TDAD_REFUSED", 0.0))
                    direct.at[idx, "TDAD_DIGSOLID_KG"] = float(res_lu.get("TDAD_DIGESTATE", 0.0)) * 0.05
                    if energy_use not in {"GRIDGAS", "GASGRID"}:
                        direct.at[idx, "ELEC_TDAD_KWH"] = float(res_lu.get("TDAD_BIOGAS", 0.0)) * BIOGAS_HV * 0.35 / KJ_PER_KWH
                continue

            refuse_kg = float(tdad_module.tdad_refused(wt)) if hasattr(tdad_module, "tdad_refused") else 0.0
            digest_t = float(tdad_module.tdad_digestate(wt)) if hasattr(tdad_module, "tdad_digestate") else 0.0
            digts_t = float(tdad_module.tdad_digTS(digest_t)) if hasattr(tdad_module, "tdad_digTS") else 0.0
            digsolid_t = float(tdad_module.tdad_digsolid(digts_t)) if hasattr(tdad_module, "tdad_digsolid") else 0.0
            biogas = float(tdad_module.tdad_ch4(wt)) if hasattr(tdad_module, "tdad_ch4") else 0.0
            elec = float(tdad_module.tdad_elec(wt)) if hasattr(tdad_module, "tdad_elec") else 0.0
            heat = float(tdad_module.tdad_energy_MJ(wt, AIRTEMP_FALLBACK, cap_amount)) if hasattr(tdad_module, "tdad_energy_MJ") else 0.0
            direct.at[idx, "TDAD_ELEC_KWH"] = elec
            direct.at[idx, "TDAD_HEAT_MJ"] = heat
            direct.at[idx, "TDAD_BIOGAS_NM3"] = biogas
            direct.at[idx, "TDAD_DIGEST_KG"] = digest_t * 1000.0
            direct.at[idx, "TDAD_REFUSE_KG"] = refuse_kg
            direct.at[idx, "TDAD_DIGSOLID_KG"] = digsolid_t * 1000.0
            transfer_rows.append({"YEAR": year, "FROM_FACILITY": fac, "FROM_BASE": base, "KG": refuse_kg, "KM_FIELD": "TDAD_REFUSE_KM", "COMP": {"FOOD": refuse_kg}})
            transfer_rows.append({"YEAR": year, "FROM_FACILITY": fac, "FROM_BASE": base, "KG": digsolid_t * 1000.0, "KM_FIELD": None, "COMP": {"FOOD": digsolid_t * 1000.0}})
            if energy_use not in {"GRIDGAS", "GASGRID"}:
                direct.at[idx, "ELEC_TDAD_KWH"] = BIOGAS_HV * biogas * 0.35 / KJ_PER_KWH
        elif typ == "RDF":
            wt = float(row["WASTE_KG"])
            rdf_kg = wt * (1.0 / RDF_CONV)
            refuse_kg = wt * (1.0 - 1.0 / RDF_CONV)
            direct.at[idx, "RDF_KG"] = rdf_kg
            direct.at[idx, "RDF_REFUSE_KG"] = refuse_kg
            total_comp = float(row["PLASTIC_KG"] + row["SYNTEX_KG"] + row["FOOD_KG"] + row["PAPER_KG"] + row["NATTEX_KG"] + row["WOOD_KG"])
            if total_comp <= 0:
                shares = {"PLASTIC":0.0,"SYNTEX":0.0,"FOOD":0.0,"PAPER":0.0,"NATTEX":0.0,"WOOD":0.0}
            else:
                shares = {
                    "PLASTIC": float(row["PLASTIC_KG"]) / total_comp,
                    "SYNTEX": float(row["SYNTEX_KG"]) / total_comp,
                    "FOOD": float(row["FOOD_KG"]) / total_comp,
                    "PAPER": float(row["PAPER_KG"]) / total_comp,
                    "NATTEX": float(row["NATTEX_KG"]) / total_comp,
                    "WOOD": float(row["WOOD_KG"]) / total_comp,
                }
            comp_rdf = {k: rdf_kg * v for k, v in shares.items()}
            comp_ref = {k: refuse_kg * v for k, v in shares.items()}
            transfer_rows.append({"YEAR": year, "FROM_FACILITY": fac, "FROM_BASE": base, "KG": rdf_kg, "KM_FIELD": "RDF_KM", "COMP": comp_rdf})
            transfer_rows.append({"YEAR": year, "FROM_FACILITY": fac, "FROM_BASE": base, "KG": refuse_kg, "KM_FIELD": "RDF_REFUSE_KM", "COMP": comp_ref})
            gen_eff = 0.30
            direct.at[idx, "ELEC_RDF_KWH"] = RDF_HV * rdf_kg * gen_eff / KJ_PER_KWH
        elif typ == "MWAD":
            wt = float(row["WASTE_KG"])
            if mwad_module is None:
                cap = float(row.get("CAPACITY", wt))
                use_grid = str(row.get("ENERGY_USE", "")).upper() in {"GRIDGAS", "GASGRID"}
                res_lu = lookup_ad_reaction(ad_lookup, "MWAD", wt, cap if cap > 0 else wt, {"FOOD_SHARE": 1.0, "PAPER_SHARE": 0.0, "WOOD_SHARE": 0.0, "MANURE_SHARE": 0.0})
                if res_lu:
                    direct.at[idx, "MWAD_ELEC_KWH"] = float(res_lu.get("MWAD_ELEC_REQ", 0.0))
                    direct.at[idx, "MWAD_HEAT_MJ"] = float(res_lu.get("MWAD_HEAT_REQ", 0.0))
                    direct.at[idx, "MWAD_WATER_M3"] = float(res_lu.get("MWAD_WATER_REQ", 0.0))
                    direct.at[idx, "MWAD_BIOGAS_NM3"] = float(res_lu.get("MWAD_BIOGAS", 0.0))
                    direct.at[idx, "MWAD_DIGEST_KG"] = float(res_lu.get("MWAD_DIGESTATE", 0.0))
                    direct.at[idx, "MWAD_DIGSOLID_KG"] = float(res_lu.get("MWAD_DIGESTATE", 0.0)) * 0.05
                    direct.at[idx, "MWAD_DIGEST_ORGC_KG"] = float(res_lu.get("MWAD_DIG_ORGC", 0.0))
                    direct.at[idx, "MWAD_DIGEST_N_KG"] = float(res_lu.get("MWAD_DIG_N", 0.0))
                    direct.at[idx, "MWAD_DIGEST_P2O5_KG"] = float(res_lu.get("MWAD_DIG_P2O5", 0.0))
                    direct.at[idx, "MWAD_DIGEST_K2O_KG"] = float(res_lu.get("MWAD_DIG_K2O", 0.0))
                    direct.at[idx, "MWAD_REFUSE_KG"] = float(res_lu.get("MWAD_REFUSED", 0.0))
                    if not use_grid:
                        direct.at[idx, "ELEC_MWAD_KWH"] = float(res_lu.get("MWAD_BIOGAS", 0.0)) * BIOGAS_HV * 0.35 / KJ_PER_KWH
                continue
            cap = float(row["CAPACITY"])
            use_grid = str(row.get("ENERGY_USE", "")).upper() in {"GRIDGAS", "GASGRID"}
            res = mwad_module.mwad_run(wt, cap if cap > 0 else wt, air_temp=AIRTEMP_FALLBACK, Biogas_griduse=use_grid)
            if isinstance(res, dict):
                dig_ts = float(res.get("MWAD_DIG_TS", 0.0))
                digsolid = dig_ts * 0.95 / (1.0 - 0.75)
                direct.at[idx, "MWAD_ELEC_KWH"] = float(res.get("MWAD_ELEC_REQ", 0.0))
                direct.at[idx, "MWAD_HEAT_MJ"] = float(res.get("MWAD_HEAT_REQ", 0.0))
                direct.at[idx, "MWAD_WATER_M3"] = float(res.get("MWAD_WATER_REQ", 0.0))
                direct.at[idx, "MWAD_BIOGAS_NM3"] = float(res.get("MWAD_BIOGAS", 0.0))
                direct.at[idx, "MWAD_DIGEST_KG"] = float(res.get("MWAD_DIGESTATE", 0.0))
                direct.at[idx, "MWAD_DIGSOLID_KG"] = digsolid
                direct.at[idx, "MWAD_DIGEST_ORGC_KG"] = float(res.get("MWAD_DIG_ORGC", 0.0))
                direct.at[idx, "MWAD_DIGEST_N_KG"] = float(res.get("MWAD_DIG_N", 0.0))
                direct.at[idx, "MWAD_DIGEST_P2O5_KG"] = float(res.get("MWAD_DIG_P2O5", 0.0))
                direct.at[idx, "MWAD_DIGEST_K2O_KG"] = float(res.get("MWAD_DIG_K2O", 0.0))
                direct.at[idx, "MWAD_REFUSE_KG"] = float(res.get("MWAD_REFUSED", 0.0))
                transfer_rows.append({"YEAR": year, "FROM_FACILITY": fac, "FROM_BASE": base, "KG": float(res.get("MWAD_REFUSED", 0.0)), "KM_FIELD": "MWAD_REFUSE_KM", "COMP": {"FOOD": float(res.get("MWAD_REFUSED", 0.0))}})
                transfer_rows.append({"YEAR": year, "FROM_FACILITY": fac, "FROM_BASE": base, "KG": digsolid, "KM_FIELD": None, "COMP": {"FOOD": digsolid}})
                if not use_grid:
                    biogas = float(res.get("MWAD_BIOGAS", 0.0))
                    direct.at[idx, "ELEC_MWAD_KWH"] = biogas * BIOGAS_HV * 0.35 / KJ_PER_KWH

    inc_rows = direct[direct["TYPE"].astype(str).str.upper().eq("INC")].copy()
    inc_headroom: dict[tuple[int, str], float] = {}
    inc_comp_add: dict[tuple[int, str], dict[str, float]] = {}
    for year in ctx.years:
        caps = available_inc_bases(year)
        direct_year = inc_rows[inc_rows["YEAR"].astype(int) == int(year)]
        direct_map = direct_year.groupby("FACILITY")["WASTE_KG"].sum().to_dict()
        for base_name, capv in caps.items():
            inc_headroom[(int(year), base_name)] = max(float(capv) - float(direct_map.get(base_name, 0.0)), 0.0)
            inc_comp_add[(int(year), base_name)] = {"PLASTIC":0.0,"SYNTEX":0.0,"FOOD":0.0,"PAPER":0.0,"NATTEX":0.0,"WOOD":0.0}

    km_acc: dict[tuple[str, int, str], list[float]] = {}
    inc_input_add: dict[tuple[int, str], float] = {}
    for tr in transfer_rows:
        kg_left = float(tr["KG"])
        if kg_left <= 0:
            continue
        year = int(tr["YEAR"])
        from_base = str(tr["FROM_BASE"])
        cands = [b for (yy, b), rem in inc_headroom.items() if yy == year and rem > 0 and b in base_loc]
        while kg_left > 1e-9 and cands:
            ordered = []
            for b in cands:
                _, km = nearest_base_km(from_base, [b])
                if np.isfinite(km):
                    ordered.append((km, b))
            if not ordered:
                break
            ordered.sort(key=lambda x: x[0])
            km, dest = ordered[0]
            rem = inc_headroom.get((year, dest), 0.0)
            if rem <= 0:
                cands = [b for b in cands if b != dest]
                continue
            take = min(kg_left, rem)
            inc_headroom[(year, dest)] = rem - take
            inc_input_add[(year, dest)] = inc_input_add.get((year, dest), 0.0) + take
            total_comp = sum(float(v) for v in tr["COMP"].values())
            for k, v in tr["COMP"].items():
                inc_comp_add[(year, dest)][k] += take * (float(v) / total_comp) if total_comp > 0 else (take if k == "FOOD" else 0.0)
            if tr["KM_FIELD"] is not None:
                km_acc.setdefault((str(tr["FROM_FACILITY"]), year, tr["KM_FIELD"]), []).append((take, km))
            kg_left -= take
            cands = [b for b in cands if inc_headroom.get((year, b), 0.0) > 1e-9]

    for (fac, year, km_field), vals in km_acc.items():
        tot = sum(w for w, _ in vals)
        wkm = sum(w * km for w, km in vals)
        mask = (direct["FACILITY"].astype(str) == fac) & (direct["YEAR"].astype(int) == year)
        direct.loc[mask, km_field] = (wkm / tot) if tot > 0 else 0.0

    landfill_bases = fac_master[fac_master["TYPE"].eq("LANDFILL")]["BASE"].dropna().astype(str).unique().tolist()
    for idx, row in direct[direct["TYPE"].astype(str).str.upper().eq("INC")].iterrows():
        year = int(row["YEAR"])
        fac = str(row["FACILITY"])
        direct_input = float(row["WASTE_KG"])
        added = float(inc_input_add.get((year, fac), 0.0))
        inc_input = direct_input + added
        direct.at[idx, "INC_INPUT_KG"] = inc_input
        direct.at[idx, "INC_KEROS_L"] = inc_input * INC_ENERGY_L_PER_KG
        direct.at[idx, "INC_ELEC_KWH"] = inc_input * INC_ELEC_KWH_PER_KG
        ash_ratio = float(fac_master.loc[(fac_master["BASE"] == fac) & (fac_master["TYPE"] == "INC"), "ASH_RATIO"].mean()) if not fac_master.loc[(fac_master["BASE"] == fac) & (fac_master["TYPE"] == "INC"), "ASH_RATIO"].empty else DEFAULT_ASH_RATIO
        direct.at[idx, "ASH_KG"] = inc_input * ash_ratio
        _, ash_km = nearest_base_km(fac, landfill_bases)
        direct.at[idx, "ASH_KM"] = 0.0 if not np.isfinite(ash_km) else ash_km

        comp_add = inc_comp_add.get((year, fac), {"PLASTIC":0.0,"SYNTEX":0.0,"FOOD":0.0,"PAPER":0.0,"NATTEX":0.0,"WOOD":0.0})
        food = float(row["FOOD_KG"]) + comp_add["FOOD"]
        paper = float(row["PAPER_KG"]) + comp_add["PAPER"]
        plastic = float(row["PLASTIC_KG"]) + comp_add["PLASTIC"]
        syntex = float(row["SYNTEX_KG"]) + comp_add["SYNTEX"]
        nattex = float(row["NATTEX_KG"]) + comp_add["NATTEX"]
        wood = float(row["WOOD_KG"]) + comp_add["WOOD"]
        if inc_input > 0:
            food_ratio = 100.0 * food / inc_input
            paper_ratio = 100.0 * paper / inc_input
            pla_ratio = 100.0 * plastic / inc_input
            tex_ratio = 100.0 * (syntex + nattex) / inc_input
            wood_ratio = 100.0 * wood / inc_input
            lhv = -68.06 * food_ratio + 91.44 * paper_ratio + 52.65 * pla_ratio + 30.73 * tex_ratio + 34.91 * wood_ratio + 7342.79
            gen_eff_series = pd.to_numeric(fac_master.loc[fac_master["TYPE"] == "INC", "GEN_EFF"], errors="coerce").dropna()
            gen_eff = float(gen_eff_series.max()) if not gen_eff_series.empty else 0.20
            if str(row.get("ENERGY_USE", "")).upper() == "ELEC":
                direct.at[idx, "ELEC_INC_KWH"] = lhv * inc_input * gen_eff / KJ_PER_KWH

    for idx, row in direct.iterrows():
        typ = str(row["TYPE"]).upper()
        energy_use = str(row.get("ENERGY_USE", "")).upper()
        if energy_use != "ELEC":
            continue
        if typ == "INC":
            inc_input = float(pd.to_numeric(row.get("INC_INPUT_KG", 0.0), errors="coerce") or 0.0)
            if inc_input > 0 and float(pd.to_numeric(row.get("ELEC_INC_KWH", 0.0), errors="coerce") or 0.0) <= 0:
                food = float(pd.to_numeric(row.get("FOOD_KG", 0.0), errors="coerce") or 0.0)
                paper = float(pd.to_numeric(row.get("PAPER_KG", 0.0), errors="coerce") or 0.0)
                plastic = float(pd.to_numeric(row.get("PLASTIC_KG", 0.0), errors="coerce") or 0.0)
                syntex = float(pd.to_numeric(row.get("SYNTEX_KG", 0.0), errors="coerce") or 0.0)
                nattex = float(pd.to_numeric(row.get("NATTEX_KG", 0.0), errors="coerce") or 0.0)
                wood = float(pd.to_numeric(row.get("WOOD_KG", 0.0), errors="coerce") or 0.0)
                food_ratio = 100.0 * food / inc_input
                paper_ratio = 100.0 * paper / inc_input
                pla_ratio = 100.0 * plastic / inc_input
                tex_ratio = 100.0 * (syntex + nattex) / inc_input
                wood_ratio = 100.0 * wood / inc_input
                lhv = -68.06 * food_ratio + 91.44 * paper_ratio + 52.65 * pla_ratio + 30.73 * tex_ratio + 34.91 * wood_ratio + 7342.79
                gen_eff_series = pd.to_numeric(fac_master.loc[fac_master["TYPE"] == "INC", "GEN_EFF"], errors="coerce").dropna()
                gen_eff = float(gen_eff_series.max()) if not gen_eff_series.empty else 0.20
                direct.at[idx, "ELEC_INC_KWH"] = lhv * inc_input * gen_eff / KJ_PER_KWH
        elif typ == "TDAD":
            biogas = float(pd.to_numeric(row.get("TDAD_BIOGAS_NM3", 0.0), errors="coerce") or 0.0)
            if biogas > 0 and float(pd.to_numeric(row.get("ELEC_TDAD_KWH", 0.0), errors="coerce") or 0.0) <= 0:
                direct.at[idx, "ELEC_TDAD_KWH"] = BIOGAS_HV * biogas * 0.35 / KJ_PER_KWH
        elif typ == "RDF":
            rdf_kg = float(pd.to_numeric(row.get("RDF_KG", 0.0), errors="coerce") or 0.0)
            if rdf_kg > 0 and float(pd.to_numeric(row.get("ELEC_RDF_KWH", 0.0), errors="coerce") or 0.0) <= 0:
                direct.at[idx, "ELEC_RDF_KWH"] = RDF_HV * rdf_kg * 0.30 / KJ_PER_KWH
        elif typ == "MWAD":
            biogas = float(pd.to_numeric(row.get("MWAD_BIOGAS_NM3", 0.0), errors="coerce") or 0.0)
            if biogas > 0 and float(pd.to_numeric(row.get("ELEC_MWAD_KWH", 0.0), errors="coerce") or 0.0) <= 0:
                direct.at[idx, "ELEC_MWAD_KWH"] = biogas * BIOGAS_HV * 0.35 / KJ_PER_KWH

    cost = load_csv(ctx.paths.cost_csv)
    validate_required_columns(cost, ["YEAR", "REBUILD", "RENOV"], "00_cost.csv")
    for c in ["YEAR", "REBUILD", "RENOV"]:
        cost[c] = pd.to_numeric(cost[c], errors="coerce")

    years_target = np.asarray(ctx.years, dtype=int)

    def _extrap(x: np.ndarray, y: np.ndarray, xnew: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 2:
            return np.full(len(xnew), float(y[-1]) if len(y) else 1.0, dtype=float)
        a, b = np.polyfit(x, y, 1)
        return a * xnew + b

    x_reb = cost.loc[cost["REBUILD"].notna(), "YEAR"].to_numpy(dtype=float)
    y_reb = cost.loc[cost["REBUILD"].notna(), "REBUILD"].to_numpy(dtype=float)
    x_ren = cost.loc[cost["RENOV"].notna(), "YEAR"].to_numpy(dtype=float)
    y_ren = cost.loc[cost["RENOV"].notna(), "RENOV"].to_numpy(dtype=float)
    reb_2024 = float(_extrap(x_reb, y_reb, np.asarray([2024.0]))[0]) if len(y_reb) else 1.0
    ren_2024 = float(_extrap(x_ren, y_ren, np.asarray([2024.0]))[0]) if len(y_ren) else 1.0
    if (not np.isfinite(reb_2024)) or reb_2024 == 0:
        reb_2024 = 1.0
    if (not np.isfinite(ren_2024)) or ren_2024 == 0:
        ren_2024 = 1.0
    ratio_df = pd.DataFrame({
        "YEAR": years_target,
        "REBUILD_RATIO": _extrap(x_reb, y_reb, years_target.astype(float)) / reb_2024 if len(y_reb) else np.ones(len(years_target)),
        "RENOV_RATIO": _extrap(x_ren, y_ren, years_target.astype(float)) / ren_2024 if len(y_ren) else np.ones(len(years_target)),
    })
    ratio_df["REBUILD_RATIO"] = pd.to_numeric(ratio_df["REBUILD_RATIO"], errors="coerce").fillna(1.0)
    ratio_df["RENOV_RATIO"] = pd.to_numeric(ratio_df["RENOV_RATIO"], errors="coerce").fillna(1.0)
    ratio_map_reb = dict(zip(ratio_df["YEAR"].astype(int), ratio_df["REBUILD_RATIO"].astype(float)))
    ratio_map_ren = dict(zip(ratio_df["YEAR"].astype(int), ratio_df["RENOV_RATIO"].astype(float)))

    cap2050_src = direct[direct["YEAR"].astype(int) == 2050].copy()
    if cap2050_src.empty:
        cap2050_src = direct.copy()
    cap2050 = (
        cap2050_src.groupby(["FACILITY", "TYPE"], as_index=False)["CAPACITY"]
        .sum()
        .rename(columns={"CAPACITY": "CAPACITY_2050"})
    )
    cap2050["CAPACITY_2050"] = pd.to_numeric(cap2050["CAPACITY_2050"], errors="coerce").fillna(0.0)

    ev_rows: list[dict[str, Any]] = []
    for fac, typ in cap2050[["FACILITY", "TYPE"]].drop_duplicates().itertuples(index=False):
        fac = str(fac)
        typ = str(typ).upper()
        orig = fac_master[(fac_master["BASE"] == fac) & (fac_master["TYPE"] == typ)]
        if typ == "MWAD":
            if orig.empty:
                event_years = pd.to_numeric(
                    direct.loc[(direct["FACILITY"] == fac) & (direct["TYPE"] == typ) & (direct["EVENT"].astype(str) == "NEW"), "YEAR"],
                    errors="coerce",
                ).dropna().astype(int).tolist()
            else:
                vals = pd.to_numeric(orig.get("CONST_YEAR", orig.get("MWAD_YEAR", np.nan)), errors="coerce").dropna().astype(int).tolist()
                event_years = vals
            for ey in sorted(set(event_years)):
                if 2030 <= int(ey) <= 2050:
                    ev_rows.append({"FACILITY": fac, "TYPE": typ, "EVENT": "NEW", "EVENT_YEAR": int(ey)})
            continue
        if orig.empty:
            continue
        rebuild_years = pd.to_numeric(orig.get("REBUILD_YEAR", np.nan), errors="coerce").dropna().astype(int).tolist()
        renov_years = pd.to_numeric(orig.get("RENOV_YEAR", np.nan), errors="coerce").dropna().astype(int).tolist()
        for ey in sorted(set(rebuild_years)):
            if 2030 <= int(ey) <= 2050:
                ev_rows.append({"FACILITY": fac, "TYPE": typ, "EVENT": "REBUILD", "EVENT_YEAR": int(ey)})
        for ey in sorted(set(renov_years)):
            if 2030 <= int(ey) <= 2050:
                ev_rows.append({"FACILITY": fac, "TYPE": typ, "EVENT": "RENEW", "EVENT_YEAR": int(ey)})
    ev_df = pd.DataFrame(ev_rows).drop_duplicates(["FACILITY", "TYPE", "EVENT", "EVENT_YEAR"]) if ev_rows else pd.DataFrame(columns=["FACILITY", "TYPE", "EVENT", "EVENT_YEAR"])

    capex_records: list[dict[str, Any]] = []
    cap2050_map = {(str(r["FACILITY"]), str(r["TYPE"]).upper()): float(r["CAPACITY_2050"]) for _, r in cap2050.iterrows()}
    fac_type_event_costs: dict[tuple[str, str], list[tuple[int, float]]] = {}

    for _, ev in ev_df.iterrows():
        fac = str(ev["FACILITY"])
        typ = str(ev["TYPE"]).upper()
        event = str(ev["EVENT"]).upper()
        event_year = int(ev["EVENT_YEAR"])
        cap_2050 = float(cap2050_map.get((fac, typ), 0.0))
        capex_event = 0.0
        cap_scale = (cap_2050 * 0.001) / 300.0 if cap_2050 > 0 else 0.0

        if cap_scale > 0 and typ in {"INC", "RDF", "TDAD"}:
            if event == "RENEW":
                unit_capex = np.exp(5.242 - 0.500 * np.log(cap_scale))
                capex_event = cap_scale * unit_capex * 10**6 * float(ratio_map_ren.get(event_year, 1.0))
            elif event == "REBUILD" and typ == "INC":
                unit_capex = np.exp(5.578 - 0.323 * np.log(cap_scale) + 0.19)
                capex_event = cap_scale * unit_capex * 10**6 * float(ratio_map_reb.get(event_year, 1.0))
            elif event == "REBUILD" and typ == "RDF":
                capex_event = 336.61 * (cap_scale ** 0.7999) * 10**6 * float(ratio_map_reb.get(event_year, 1.0))
            elif event == "REBUILD" and typ == "TDAD":
                capex_event = 112.86 * (cap_scale ** 0.7999) * 10**6 * float(ratio_map_reb.get(event_year, 1.0))
        elif typ == "MWAD" and event == "NEW" and mwad_module is not None and cap_2050 > 0:
            try:
                res_cap = mwad_module.mwad_run(cap_2050, cap_2050, air_temp=AIRTEMP_FALLBACK, Biogas_griduse=False)
                capex_event = float(res_cap.get("MWAD_CAPEX", 0.0)) * float(ratio_map_reb.get(event_year, 1.0))
            except Exception:
                capex_event = 0.0

        if capex_event > 0:
            fac_type_event_costs.setdefault((fac, typ), []).append((event_year, capex_event))

    for (fac, typ), event_costs in fac_type_event_costs.items():
        start_year = min(y for y, _ in event_costs)
        total_capex = sum(v for _, v in event_costs)
        n_years = max(2050 - int(start_year) + 1, 1)
        annual = total_capex / n_years
        for y in ctx.years:
            if y >= start_year:
                capex_records.append({"FACILITY": fac, "TYPE": typ, "YEAR": int(y), "ANNUAL_CAPEX": annual})

    capex_df = pd.DataFrame(capex_records)
    if not capex_df.empty:
        capex_df = capex_df.groupby(["FACILITY", "TYPE", "YEAR"], as_index=False)["ANNUAL_CAPEX"].sum()
        direct = direct.merge(capex_df, on=["FACILITY", "TYPE", "YEAR"], how="left")
        direct["ANNUAL_CAPEX"] = pd.to_numeric(direct["ANNUAL_CAPEX"], errors="coerce").fillna(0.0)
        direct.loc[direct["TYPE"].eq("INC"), "INC_CAPEX"] = direct.loc[direct["TYPE"].eq("INC"), "ANNUAL_CAPEX"]
        direct.loc[direct["TYPE"].eq("TDAD"), "TDAD_CAPEX"] = direct.loc[direct["TYPE"].eq("TDAD"), "ANNUAL_CAPEX"]
        direct.loc[direct["TYPE"].eq("RDF"), "RDF_CAPEX"] = direct.loc[direct["TYPE"].eq("RDF"), "ANNUAL_CAPEX"]
        direct.loc[direct["TYPE"].eq("MWAD"), "MWAD_CAPEX"] = direct.loc[direct["TYPE"].eq("MWAD"), "ANNUAL_CAPEX"]
        direct = direct.drop(columns=["ANNUAL_CAPEX"])

    if manure_inventory is not None and not manure_inventory.empty:
        mi = manure_inventory.copy()
        mi["FACILITY"] = mi["FACILITY"].astype(str)
        mi["YEAR"] = pd.to_numeric(mi.get("YEAR", 0), errors="coerce").fillna(0).astype(int)
        manure_cols = ["MANURE_KG", "DAIRY_MANURE_KG", "CATTLE_MANURE_KG", "SWINE_MANURE_KG", "CHICKEN_MANURE_KG", "BROILER_MANURE_KG", "MANURE_KM"]
        for c in manure_cols:
            mi[c] = pd.to_numeric(mi.get(c, 0.0), errors="coerce").fillna(0.0)
        mi = mi.groupby(["FACILITY", "YEAR"], as_index=False)[manure_cols].sum()

        direct = direct.drop(columns=[c for c in manure_cols if c in direct.columns], errors="ignore")
        direct = direct.merge(mi, on=["FACILITY", "YEAR"], how="left")
        for c in manure_cols:
            if c not in direct.columns:
                direct[c] = 0.0
            direct[c] = pd.to_numeric(direct[c], errors="coerce").fillna(0.0)
        non_mwad = ~direct["TYPE"].astype(str).str.upper().eq("MWAD")
        for c in manure_cols:
            direct.loc[non_mwad, c] = 0.0

        mwad_mask = direct["TYPE"].astype(str).str.upper().eq("MWAD")
        if mwad_mask.any():
            total_feed = (
                pd.to_numeric(direct.loc[mwad_mask, "WASTE_KG"], errors="coerce").fillna(0.0)
                + pd.to_numeric(direct.loc[mwad_mask, "MANURE_KG"], errors="coerce").fillna(0.0)
            )
            direct.loc[mwad_mask, "CAPACITY"] = [
                max(float(cap), ceil_to_1000(float(feed)))
                for cap, feed in zip(
                    pd.to_numeric(direct.loc[mwad_mask, "CAPACITY"], errors="coerce").fillna(0.0),
                    total_feed,
                    strict=False,
                )
            ]
            if mwad_module is not None or not ad_lookup.empty:
                for idx, feed, cap in zip(direct.index[mwad_mask], total_feed, direct.loc[mwad_mask, "CAPACITY"], strict=False):
                    feed = float(feed)
                    cap = float(pd.to_numeric(cap, errors="coerce") or 0.0)
                    use_grid = str(direct.at[idx, "ENERGY_USE"]).upper() in {"GRIDGAS", "GASGRID"}
                    if feed <= 0:
                        for col in [
                            "MWAD_ELEC_KWH", "MWAD_HEAT_MJ", "MWAD_WATER_M3", "MWAD_BIOGAS_NM3",
                            "MWAD_DIGEST_KG", "MWAD_DIGSOLID_KG", "MWAD_DIGEST_ORGC_KG", "MWAD_DIGEST_N_KG",
                            "MWAD_DIGEST_P2O5_KG", "MWAD_DIGEST_K2O_KG", "MWAD_REFUSE_KG", "ELEC_MWAD_KWH",
                        ]:
                            direct.at[idx, col] = 0.0
                        continue
                    if mwad_module is not None:
                        try:
                            res = mwad_module.mwad_run(feed, cap if cap > 0 else feed, air_temp=AIRTEMP_FALLBACK, Biogas_griduse=use_grid)
                        except Exception:
                            res = {}
                    else:
                        manure_share = float(pd.to_numeric(direct.at[idx, "MANURE_KG"], errors="coerce") or 0.0) / feed if feed > 0 else 0.0
                        food_share = max(1.0 - manure_share, 0.0)
                        res = lookup_ad_reaction(ad_lookup, "MWAD", feed, cap if cap > 0 else feed, {"FOOD_SHARE": food_share, "PAPER_SHARE": 0.0, "WOOD_SHARE": 0.0, "MANURE_SHARE": manure_share})
                    dig_ts = float(res.get("MWAD_DIG_TS", 0.0))
                    direct.at[idx, "MWAD_ELEC_KWH"] = float(res.get("MWAD_ELEC_REQ", 0.0))
                    direct.at[idx, "MWAD_HEAT_MJ"] = float(res.get("MWAD_HEAT_REQ", 0.0))
                    direct.at[idx, "MWAD_WATER_M3"] = float(res.get("MWAD_WATER_REQ", 0.0))
                    direct.at[idx, "MWAD_BIOGAS_NM3"] = float(res.get("MWAD_BIOGAS", 0.0))
                    direct.at[idx, "MWAD_DIGEST_KG"] = float(res.get("MWAD_DIGESTATE", 0.0))
                    direct.at[idx, "MWAD_DIGSOLID_KG"] = dig_ts * MWAD_DIGSOLID_TS_RECOVERY / (1.0 - MWAD_DIGSOLID_WATER)
                    direct.at[idx, "MWAD_DIGEST_ORGC_KG"] = float(res.get("MWAD_DIG_ORGC", 0.0))
                    direct.at[idx, "MWAD_DIGEST_N_KG"] = float(res.get("MWAD_DIG_N", 0.0))
                    direct.at[idx, "MWAD_DIGEST_P2O5_KG"] = float(res.get("MWAD_DIG_P2O5", 0.0))
                    direct.at[idx, "MWAD_DIGEST_K2O_KG"] = float(res.get("MWAD_DIG_K2O", 0.0))
                    direct.at[idx, "MWAD_REFUSE_KG"] = float(res.get("MWAD_REFUSED", 0.0))
                    direct.at[idx, "ELEC_MWAD_KWH"] = 0.0 if use_grid else float(res.get("MWAD_BIOGAS", 0.0)) * BIOGAS_HV * 0.35 / KJ_PER_KWH

    for col in [c for c in WASTE_OUTPUT_COLUMNS if c not in direct.columns]:
        direct[col] = 0.0 if col not in {"FACILITY", "TYPE", "EVENT", "ENERGY_USE", "YEAR"} else ("" if col in {"FACILITY", "TYPE", "EVENT", "ENERGY_USE"} else 0)
    direct["YEAR"] = pd.to_numeric(direct["YEAR"], errors="coerce").fillna(0).astype(int)
    direct["EVENT"] = direct["EVENT"].fillna("").astype(str)
    direct["ENERGY_USE"] = direct["ENERGY_USE"].fillna("").astype(str)
    num_cols = [c for c in direct.columns if c not in {"FACILITY", "TYPE", "EVENT", "ENERGY_USE"}]
    for c in num_cols:
        direct[c] = pd.to_numeric(direct[c], errors="coerce").fillna(0.0)
    direct = enforce_component_sum_limit(
        clean_small_numeric(direct),
        ["PLASTIC_KG", "SYNTEX_KG", "FOOD_KG", "PAPER_KG", "NATTEX_KG", "WOOD_KG"],
        "WASTE_KG",
    )
    direct = enforce_species_manure_mass_balance(direct)
    return finalize_output_schema(direct, WASTE_OUTPUT_COLUMNS)




def _num_value(value: Any, default: float = 0.0) -> float:
    val = pd.to_numeric(value, errors="coerce")
    try:
        if pd.isna(val):
            return float(default)
    except Exception:
        return float(default)
    try:
        return float(val)
    except Exception:
        return float(default)


def _int_value(value: Any, default: int = 0) -> int:
    return int(round(_num_value(value, float(default))))


def _biowaste_n_per_kg_lookup(ctx: InventoryContext) -> dict[str, float]:
    """Return wet-kg N fractions used for link-table N conversion."""
    try:
        bw = build_biowaste_lookup(build_shared_inputs(ctx)["biowaste"])
    except Exception:
        bw = pd.DataFrame()

    def get_n(*names: str, default: float = 0.0) -> float:
        if bw.empty:
            return default
        for name in names:
            key = str(name).upper().strip()
            if key in bw.index:
                row = bw.loc[key]
                vs = float(pd.to_numeric(row.get("VS", 1.0), errors="coerce") or 0.0)
                nit = float(pd.to_numeric(row.get("NIT", 0.0), errors="coerce") or 0.0)
                return vs * nit
        return default

    out = {
        "FOOD": get_n("FOOD", "FOOD_WASTE", "FW", default=0.0),
        "PAPER": get_n("PAPER", "PAPER_WASTE", default=0.0),
        "WOOD": get_n("WOOD", "WOOD_WASTE", default=0.0),
        "RESIDUE": RESIDUE_N_PER_KG,
    }
    for species in ["DAIRY", "CATTLE", "SWINE", "CHICKEN", "BROILER"]:
        try:
            out[species] = get_species_n_per_kg(bw, species) / sum(MANURE_UNIT_RATES[species].values()) if sum(MANURE_UNIT_RATES[species].values()) > 0 else 0.0
        except Exception:
            out[species] = 0.0
    return out


def _empty_waste_link() -> pd.DataFrame:
    return pd.DataFrame(columns=WASTE_LINK_COLUMNS)


def _empty_fertilizer_link() -> pd.DataFrame:
    return pd.DataFrame(columns=FERTILIZER_LINK_COLUMNS)


def _finalize_link(df: pd.DataFrame, columns: list[str], value_cols: list[str]) -> pd.DataFrame:
    out = df.copy() if df is not None else pd.DataFrame()
    for col in columns:
        if col not in out.columns:
            out[col] = 0.0 if col in value_cols or col == "YEAR" else ""
    out = out[columns].copy()
    if "YEAR" in out.columns:
        out["YEAR"] = pd.to_numeric(out["YEAR"], errors="coerce").fillna(0).astype(int)
        out = out[out["YEAR"] >= OUTPUT_START_YEAR].copy()
    for col in value_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    group_cols = [c for c in columns if c not in value_cols]
    if out.empty:
        return pd.DataFrame(columns=columns)
    out = out.groupby(group_cols, as_index=False, dropna=False)[value_cols].sum()
    out = out[out[value_cols].sum(axis=1).abs() > 1e-9].copy()
    out = clean_small_numeric(out)
    return out[columns].sort_values(group_cols).reset_index(drop=True)


def write_waste_link_from_raw(
    ctx: InventoryContext,
    raw_routes: pd.DataFrame,
    manure_inventory: pd.DataFrame | None,
) -> pd.DataFrame:
    """Create 02_waste_link.csv from source-level routing rows and manure/residue links."""
    nfac = _biowaste_n_per_kg_lookup(ctx)
    rows: list[dict[str, Any]] = []

    if raw_routes is not None and not raw_routes.empty:
        rr = raw_routes.copy()
        for c in ["FROM_KEY", "FROM_CITY", "FACILITY", "YEAR", "TYPE", "FOOD_KG", "PAPER_KG", "WOOD_KG"]:
            if c not in rr.columns:
                rr[c] = "" if c in {"FROM_KEY", "FROM_CITY", "FACILITY", "TYPE"} else 0.0
        for _, r in rr.iterrows():
            rows.append({
                "YEAR": _int_value(r.get("YEAR", 0)),
                "FROM_KEY": normalize_code(r.get("FROM_KEY", "")),
                "FROM_CITY": normalize_code(r.get("FROM_CITY", "")),
                "TO_FACILITY": str(r.get("FACILITY", "")),
                "TYPE": str(r.get("TYPE", "")).upper().strip(),
                "FOOD_N_KG": _num_value(r.get("FOOD_KG", 0.0)) * nfac["FOOD"],
                "PAPER_N_KG": _num_value(r.get("PAPER_KG", 0.0)) * nfac["PAPER"],
                "WOOD_N_KG": _num_value(r.get("WOOD_KG", 0.0)) * nfac["WOOD"],
                "DAIRY_N_KG": 0.0,
                "CATTLE_N_KG": 0.0,
                "SWINE_N_KG": 0.0,
                "CHICKEN_N_KG": 0.0,
                "BROILER_N_KG": 0.0,
                "RESIDUE_N_KG": 0.0,
            })

    shared = build_shared_inputs(ctx)
    assignments = shared.get("_manure_assignments", pd.DataFrame())
    if isinstance(assignments, pd.DataFrame) and not assignments.empty:
        ass = assignments.copy()
        for c in ["YEAR", "KEY", "FACILITY", "TYPE", "SPECIES", "ASSIGNED_N"]:
            if c not in ass.columns:
                ass[c] = 0.0 if c in {"YEAR", "ASSIGNED_N"} else ""
        key_city = shared.get("key_centroids", pd.DataFrame())[["KEY", "CITY"]].drop_duplicates().copy()
        key_city["KEY"] = key_city["KEY"].map(normalize_code)
        ass["KEY"] = ass["KEY"].map(normalize_code)
        ass = ass.merge(key_city, on="KEY", how="left")
        for _, r in ass.iterrows():
            species = normalize_species(r.get("SPECIES", ""))
            if species not in {"DAIRY", "CATTLE", "SWINE", "CHICKEN", "BROILER"}:
                continue
            rec = {c: 0.0 for c in WASTE_LINK_COLUMNS if c.endswith("_N_KG")}
            rec.update({
                "YEAR": _int_value(r.get("YEAR", 0)),
                "FROM_KEY": normalize_code(r.get("KEY", "")),
                "FROM_CITY": normalize_code(r.get("CITY", "")),
                "TO_FACILITY": str(r.get("FACILITY", "")),
                "TYPE": str(r.get("TYPE", "")).upper().strip(),
            })
            rec[f"{species}_N_KG"] = _num_value(r.get("ASSIGNED_N", 0.0))
            rows.append(rec)

    residue_alloc = shared.get("_residue_allocations", pd.DataFrame())
    if isinstance(residue_alloc, pd.DataFrame) and not residue_alloc.empty:
        ra = residue_alloc.copy()
        key_city = shared.get("key_centroids", pd.DataFrame())[["KEY", "CITY"]].drop_duplicates().copy()
        key_city["KEY"] = key_city["KEY"].map(normalize_code)
        ra["KEY"] = ra["KEY"].map(normalize_code)
        ra = ra.merge(key_city, on="KEY", how="left")
        for _, r in ra.iterrows():
            fac = str(r.get("FACILITY", ""))
            year = _int_value(r.get("YEAR", 0))
            rec = {c: 0.0 for c in WASTE_LINK_COLUMNS if c.endswith("_N_KG")}
            rec.update({
                "YEAR": year,
                "FROM_KEY": normalize_code(r.get("KEY", "")),
                "FROM_CITY": normalize_code(r.get("CITY", "")),
                "TO_FACILITY": fac,
                "TYPE": "COMPOST",
                "RESIDUE_N_KG": _num_value(r.get("TAKE_N_KG", 0.0)),
            })
            rows.append(rec)

    if manure_inventory is not None and not manure_inventory.empty:
        mi = manure_inventory.copy()
        for c in ["RESIDUE_IMPORT_KG", "FACILITY", "YEAR", "TYPE"]:
            if c not in mi.columns:
                mi[c] = 0.0 if c in {"RESIDUE_IMPORT_KG", "YEAR"} else ""
        mi["TYPE"] = mi["TYPE"].astype(str).str.upper().str.strip()
        # Imported residue is also compost auxiliary material; never export it as MWAD input.
        mi = mi[mi["TYPE"].ne("MWAD")].copy()
        mi["RESIDUE_IMPORT_KG"] = pd.to_numeric(mi["RESIDUE_IMPORT_KG"], errors="coerce").fillna(0.0)
        for _, r in mi[mi["RESIDUE_IMPORT_KG"] > 0].iterrows():
            rec = {c: 0.0 for c in WASTE_LINK_COLUMNS if c.endswith("_N_KG")}
            rec.update({
                "YEAR": _int_value(r.get("YEAR", 0)),
                "FROM_KEY": "IMPORTED",
                "FROM_CITY": "IMPORTED",
                "TO_FACILITY": str(r.get("FACILITY", "")),
                "TYPE": str(r.get("TYPE", "")).upper().strip(),
                "RESIDUE_N_KG": float(r.get("RESIDUE_IMPORT_KG", 0.0)) * nfac["RESIDUE"],
            })
            rows.append(rec)

    out = _finalize_link(pd.DataFrame(rows), WASTE_LINK_COLUMNS, [c for c in WASTE_LINK_COLUMNS if c.endswith("_N_KG")])
    if not out.empty and "TYPE" in out.columns and "RESIDUE_N_KG" in out.columns:
        mwad_mask = out["TYPE"].astype(str).str.upper().str.strip().eq("MWAD")
        out.loc[mwad_mask, "RESIDUE_N_KG"] = 0.0
        out = out[out[[c for c in WASTE_LINK_COLUMNS if c.endswith("_N_KG")]].sum(axis=1).abs() > 1e-12].copy()
    ensure_dir(ctx.paths.output_dir)
    out.to_csv(ctx.paths.output_dir / "02_waste_link.csv", index=False)
    log(f"Wrote 02_waste_link.csv: {len(out):,} rows")
    return out


def write_fertilizer_link(
    ctx: InventoryContext,
    crop_df: pd.DataFrame,
    waste_df: pd.DataFrame,
    output_dir: Path,
    manure_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create 02_fertilizer_link.csv from crop-side fertilizer use and unused organic outputs.

    Local/ imported fertilizer rows are written as N flows. Unused digestate and
    unused local compost are represented by facility -> UNUSED rows, using the
    same DIGEST_N_KG and COMPOST_N_KG columns so the table remains an N-flow link.
    """
    rows: list[dict[str, Any]] = []
    crop = crop_df.copy() if crop_df is not None else pd.DataFrame()

    def _row(year, from_facility, to_key, to_city, crop_name, cultivar, syn=0.0, compost=0.0, digest=0.0, residue=0.0):
        return {
            "YEAR": int(year),
            "FROM_FACILITY": str(from_facility),
            "TO_KEY": str(to_key),
            "TO_CITY": str(to_city),
            "CROP": str(crop_name),
            "CULTIVAR": str(cultivar),
            "SYN_N_KG": float(syn),
            "COMPOST_N_KG": float(compost),
            "DIGEST_N_KG": float(digest),
            "RESIDUE_N_KG": float(residue),
        }

    if not crop.empty:
        for c in [
            "YEAR", "KEY", "CITY", "CROP", "CULTIVAR", "SYN_N_KG", "COMPOST_N_KG",
            "COMPOST_IMP_KG", "DIGEST_N_KG", "RESIDUE_N_KG",
            "LINKED_COMPOST_FACILITY", "LINKED_DIGEST_FACILITY", "LOCAL_COMPOST_ALLOC_N_JSON",
        ]:
            if c not in crop.columns:
                crop[c] = 0.0 if c.endswith("_KG") or c == "YEAR" else ""
        crop["YEAR"] = pd.to_numeric(crop["YEAR"], errors="coerce").fillna(0).astype(int)
        for _, r in crop.iterrows():
            year = int(r["YEAR"])
            to_key = normalize_code(r.get("KEY", ""))
            to_city = normalize_code(r.get("CITY", ""))
            crop_name = str(r.get("CROP", "")).upper().strip()
            cultivar = str(r.get("CULTIVAR", "")).lower().strip()
            syn_n = _num_value(r.get("SYN_N_KG", 0.0))
            compost_total_n = _num_value(r.get("COMPOST_N_KG", 0.0))
            compost_imp_kg = _num_value(r.get("COMPOST_IMP_KG", 0.0))
            compost_imp_n = compost_imp_kg * 0.037
            compost_loc_n = max(compost_total_n - compost_imp_n, 0.0)
            digest_n = _num_value(r.get("DIGEST_N_KG", 0.0))
            residue_n = _num_value(r.get("RESIDUE_N_KG", 0.0))
            comp_fac = str(r.get("LINKED_COMPOST_FACILITY", "") or "").strip()
            dig_fac = str(r.get("LINKED_DIGEST_FACILITY", "") or "").strip()
            if syn_n > 0:
                rows.append(_row(year, "IMPORTED", to_key, to_city, crop_name, cultivar, syn=syn_n))
            if compost_imp_n > 0:
                rows.append(_row(year, "IMPORTED", to_key, to_city, crop_name, cultivar, compost=compost_imp_n))
            if compost_loc_n > 0:
                detail_raw = str(r.get("LOCAL_COMPOST_ALLOC_N_JSON", "") or "").strip()
                detail: dict[str, float] = {}
                if detail_raw:
                    try:
                        parsed = json.loads(detail_raw)
                        if isinstance(parsed, dict):
                            detail = {str(k): float(v) for k, v in parsed.items() if float(v) > 1e-9}
                    except Exception:
                        detail = {}
                if detail:
                    for fac_name, comp_n_val in detail.items():
                        rows.append(_row(year, fac_name, to_key, to_city, crop_name, cultivar, compost=comp_n_val))
                else:
                    rows.append(_row(year, comp_fac if comp_fac else "UNKNOWN", to_key, to_city, crop_name, cultivar, compost=compost_loc_n))
            if digest_n > 0:
                rows.append(_row(year, dig_fac if dig_fac else "UNKNOWN", to_key, to_city, crop_name, cultivar, digest=digest_n))
            if residue_n > 0:
                rows.append(_row(year, "FIELD_RESIDUE", to_key, to_city, crop_name, cultivar, residue=residue_n))

    # Local compost not transported/used is represented as a facility -> UNUSED link.
    manure = manure_df.copy() if manure_df is not None else pd.DataFrame()
    if not manure.empty:
        for c in ["FACILITY", "YEAR", "TYPE", "COMPOST_N"]:
            if c not in manure.columns:
                manure[c] = 0.0 if c in {"YEAR", "COMPOST_N"} else ""
        used_comp = pd.DataFrame()
        if not crop.empty:
            used_rows: list[dict[str, Any]] = []
            for _, rr in crop.iterrows():
                year_rr_val = pd.to_numeric(rr.get("YEAR", 0), errors="coerce")
                year_rr = int(year_rr_val) if pd.notna(year_rr_val) else 0
                detail_raw = str(rr.get("LOCAL_COMPOST_ALLOC_N_JSON", "") or "").strip()
                parsed_detail: dict[str, float] = {}
                if detail_raw:
                    try:
                        parsed = json.loads(detail_raw)
                        if isinstance(parsed, dict):
                            parsed_detail = {str(k): float(v) for k, v in parsed.items() if float(v) > 1e-9}
                    except Exception:
                        parsed_detail = {}
                if parsed_detail:
                    for fac_name, comp_n_val in parsed_detail.items():
                        used_rows.append({"FACILITY": fac_name, "YEAR": year_rr, "USED_COMPOST_N_KG": comp_n_val})
                else:
                    comp_fac_rr = str(rr.get("LINKED_COMPOST_FACILITY", "") or "").strip()
                    comp_total_n_rr = _num_value(rr.get("COMPOST_N_KG", 0.0))
                    comp_imp_n_rr = _num_value(rr.get("COMPOST_IMP_KG", 0.0)) * 0.037
                    comp_loc_n_rr = max(comp_total_n_rr - comp_imp_n_rr, 0.0)
                    if comp_fac_rr and comp_loc_n_rr > 0:
                        used_rows.append({"FACILITY": comp_fac_rr, "YEAR": year_rr, "USED_COMPOST_N_KG": comp_loc_n_rr})
            if used_rows:
                used_comp = pd.DataFrame(used_rows).groupby(["FACILITY", "YEAR"], as_index=False)["USED_COMPOST_N_KG"].sum()
        comp = manure[manure["TYPE"].astype(str).str.upper().eq("COMPOST")].copy()
        comp["FACILITY"] = comp["FACILITY"].astype(str)
        comp["YEAR"] = pd.to_numeric(comp["YEAR"], errors="coerce").fillna(0).astype(int)
        comp["COMPOST_N"] = pd.to_numeric(comp["COMPOST_N"], errors="coerce").fillna(0.0)
        comp = comp.groupby(["FACILITY", "YEAR"], as_index=False)["COMPOST_N"].sum()
        if not used_comp.empty:
            comp = comp.merge(used_comp, on=["FACILITY", "YEAR"], how="left")
        else:
            comp["USED_COMPOST_N_KG"] = 0.0
        comp["USED_COMPOST_N_KG"] = pd.to_numeric(comp.get("USED_COMPOST_N_KG", 0.0), errors="coerce").fillna(0.0)
        comp["UNUSED_COMPOST_N_KG"] = np.maximum(comp["COMPOST_N"] - comp["USED_COMPOST_N_KG"], 0.0)
        for _, r in comp[comp["UNUSED_COMPOST_N_KG"] > 0].iterrows():
            rows.append(_row(int(r["YEAR"]), str(r.get("FACILITY", "")), "UNUSED", "UNUSED", "UNUSED_COMPOST", "UNUSED", compost=float(r["UNUSED_COMPOST_N_KG"])))

    # Digestate not transported/used is represented as a facility -> UNUSED link.
    waste = waste_df.copy() if waste_df is not None else pd.DataFrame()
    if not waste.empty:
        for c in ["FACILITY", "YEAR", "TYPE", "MWAD_DIGEST_N_KG"]:
            if c not in waste.columns:
                waste[c] = 0.0 if c in {"YEAR", "MWAD_DIGEST_N_KG"} else ""
        used = pd.DataFrame()
        if not crop.empty and "LINKED_DIGEST_FACILITY" in crop.columns:
            used = crop.copy()
            used["LINKED_DIGEST_FACILITY"] = used["LINKED_DIGEST_FACILITY"].fillna("").astype(str)
            used["DIGEST_N_KG"] = pd.to_numeric(used.get("DIGEST_N_KG", 0.0), errors="coerce").fillna(0.0)
            used = used[(used["LINKED_DIGEST_FACILITY"] != "") & (used["DIGEST_N_KG"] > 0)].groupby(["LINKED_DIGEST_FACILITY", "YEAR"], as_index=False)["DIGEST_N_KG"].sum().rename(columns={"LINKED_DIGEST_FACILITY": "FACILITY", "DIGEST_N_KG": "USED_DIGEST_N_KG"})
        mw = waste[waste["TYPE"].astype(str).str.upper().eq("MWAD")].copy()
        mw["YEAR"] = pd.to_numeric(mw["YEAR"], errors="coerce").fillna(0).astype(int)
        mw["MWAD_DIGEST_N_KG"] = pd.to_numeric(mw["MWAD_DIGEST_N_KG"], errors="coerce").fillna(0.0)
        if not used.empty:
            mw = mw.merge(used, on=["FACILITY", "YEAR"], how="left")
        else:
            mw["USED_DIGEST_N_KG"] = 0.0
        mw["USED_DIGEST_N_KG"] = pd.to_numeric(mw.get("USED_DIGEST_N_KG", 0.0), errors="coerce").fillna(0.0)
        mw["UNUSED_DIGEST_N_KG"] = np.maximum(mw["MWAD_DIGEST_N_KG"] - mw["USED_DIGEST_N_KG"], 0.0)
        for _, r in mw[mw["UNUSED_DIGEST_N_KG"] > 0].iterrows():
            rows.append(_row(int(r["YEAR"]), str(r.get("FACILITY", "")), "UNUSED", "UNUSED", "UNUSED_DIGEST", "UNUSED", digest=float(r["UNUSED_DIGEST_N_KG"])))

    out = _finalize_link(pd.DataFrame(rows), FERTILIZER_LINK_COLUMNS, ["SYN_N_KG", "COMPOST_N_KG", "DIGEST_N_KG", "RESIDUE_N_KG"])
    ensure_dir(output_dir)
    out.to_csv(output_dir / "02_fertilizer_link.csv", index=False)
    log(f"Wrote 02_fertilizer_link.csv: {len(out):,} rows")
    return out

def write_outputs(
    manure_df: pd.DataFrame,
    crop_df: pd.DataFrame,
    waste_df: pd.DataFrame,
    output_dir: Path,
    scenario_name: str = "BASE",
) -> None:
    ensure_dir(output_dir)

    manure_out = manure_df.copy()
    crop_out = crop_df.copy()
    waste_out = waste_df.copy()

    if "YEAR" in manure_out.columns:
        manure_out = manure_out[pd.to_numeric(manure_out["YEAR"], errors="coerce").fillna(0).astype(int) >= OUTPUT_START_YEAR].copy()
    if "YEAR" in crop_out.columns:
        crop_out = crop_out[pd.to_numeric(crop_out["YEAR"], errors="coerce").fillna(0).astype(int) >= OUTPUT_START_YEAR].copy()
    if "YEAR" in waste_out.columns:
        waste_out = waste_out[pd.to_numeric(waste_out["YEAR"], errors="coerce").fillna(0).astype(int) >= OUTPUT_START_YEAR].copy()

    if "LINKED_COMPOST_FACILITY" in crop_out.columns:
        use_df = crop_out.copy()
        use_df["YEAR"] = pd.to_numeric(use_df.get("YEAR", 0), errors="coerce").fillna(0).astype(int)
        use_df["COMPOST_LOC_KG"] = pd.to_numeric(use_df.get("COMPOST_LOC_KG", 0.0), errors="coerce").fillna(0.0)
        use_df["COMPOST_N_KG"] = pd.to_numeric(use_df.get("COMPOST_N_KG", 0.0), errors="coerce").fillna(0.0)
        use_df["COMPOST_IMP_KG"] = pd.to_numeric(use_df.get("COMPOST_IMP_KG", 0.0), errors="coerce").fillna(0.0)
        use_df["LINKED_COMPOST_FACILITY"] = use_df["LINKED_COMPOST_FACILITY"].fillna("").astype(str)
        if "LOCAL_COMPOST_ALLOC_N_JSON" not in use_df.columns:
            use_df["LOCAL_COMPOST_ALLOC_N_JSON"] = ""

        comp_conc = manure_out.copy()
        for c in ["FACILITY", "YEAR", "COMPOST_KG", "COMPOST_N"]:
            if c not in comp_conc.columns:
                comp_conc[c] = 0.0 if c in {"YEAR", "COMPOST_KG", "COMPOST_N"} else ""
        comp_conc["FACILITY"] = comp_conc["FACILITY"].astype(str)
        comp_conc["YEAR"] = pd.to_numeric(comp_conc["YEAR"], errors="coerce").fillna(0).astype(int)
        comp_conc["COMPOST_KG"] = pd.to_numeric(comp_conc["COMPOST_KG"], errors="coerce").fillna(0.0)
        comp_conc["COMPOST_N"] = pd.to_numeric(comp_conc["COMPOST_N"], errors="coerce").fillna(0.0)
        comp_conc = comp_conc.groupby(["FACILITY", "YEAR"], as_index=False)[["COMPOST_KG", "COMPOST_N"]].sum()
        conc_map = {
            (str(r["FACILITY"]), int(r["YEAR"])): (float(r["COMPOST_N"]) / float(r["COMPOST_KG"]))
            for _, r in comp_conc.iterrows()
            if float(r["COMPOST_KG"]) > 0 and float(r["COMPOST_N"]) > 0
        }

        use_rows: list[dict[str, Any]] = []
        for _, rr in use_df.iterrows():
            year_rr = int(rr["YEAR"])
            detail_raw = str(rr.get("LOCAL_COMPOST_ALLOC_N_JSON", "") or "").strip()
            parsed_detail: dict[str, float] = {}
            if detail_raw:
                try:
                    parsed = json.loads(detail_raw)
                    if isinstance(parsed, dict):
                        parsed_detail = {str(k): float(v) for k, v in parsed.items() if float(v) > 1e-9}
                except Exception:
                    parsed_detail = {}
            if parsed_detail:
                for fac_name, comp_n_val in parsed_detail.items():
                    conc = conc_map.get((str(fac_name), year_rr), 0.037)
                    use_rows.append({"FACILITY": str(fac_name), "YEAR": year_rr, "COMPOST_USED_KG": float(comp_n_val) / max(conc, 1e-12)})
            else:
                comp_fac_rr = str(rr.get("LINKED_COMPOST_FACILITY", "") or "").strip()
                comp_total_n_rr = float(rr.get("COMPOST_N_KG", 0.0))
                comp_imp_n_rr = float(rr.get("COMPOST_IMP_KG", 0.0)) * 0.037
                comp_loc_n_rr = max(comp_total_n_rr - comp_imp_n_rr, 0.0)
                if comp_fac_rr and comp_loc_n_rr > 0:
                    conc = conc_map.get((comp_fac_rr, year_rr), 0.037)
                    use_rows.append({"FACILITY": comp_fac_rr, "YEAR": year_rr, "COMPOST_USED_KG": comp_loc_n_rr / max(conc, 1e-12)})
        if use_rows:
            fac_use = pd.DataFrame(use_rows).groupby(["FACILITY", "YEAR"], as_index=False)["COMPOST_USED_KG"].sum()
        else:
            fac_use = pd.DataFrame(columns=["FACILITY", "YEAR", "COMPOST_USED_KG"])
        manure_out = manure_out.drop(columns=["COMPOST_USED_KG"], errors="ignore").merge(fac_use, on=["FACILITY", "YEAR"], how="left")
    else:
        manure_out["COMPOST_USED_KG"] = 0.0

    if "COMPOST_USED_KG" in manure_out.columns:
        manure_out["COMPOST_USED_KG"] = pd.to_numeric(manure_out["COMPOST_USED_KG"], errors="coerce").fillna(0.0)
    else:
        manure_out["COMPOST_USED_KG"] = 0.0
    manure_out["COMPOST_KG"] = pd.to_numeric(manure_out.get("COMPOST_KG", 0.0), errors="coerce").fillna(0.0)
    if "TYPE" in manure_out.columns:
        compost_mask = manure_out["TYPE"].astype(str).str.upper() == "COMPOST"
        manure_out.loc[~compost_mask, "COMPOST_USED_KG"] = 0.0
        manure_out.loc[compost_mask, "COMPOST_USED_KG"] = np.minimum(
            manure_out.loc[compost_mask, "COMPOST_USED_KG"],
            manure_out.loc[compost_mask, "COMPOST_KG"],
        )
        if is_alt_scenario(scenario_name):
            manure_out = manure_out[manure_out["TYPE"].astype(str).str.upper() != "MWAD"].copy()

    manure_out = zero_mwad_compost_residue_fields(manure_out)

    zero_manure_mask = pd.to_numeric(manure_out.get("MANURE_KG", 0.0), errors="coerce").fillna(0.0) <= 0
    for col in ["RESIDUE_REQ_KG", "RESIDUE_SUPPLY_KG", "RESIDUE_IMPORT_KG", "RESIDUE_SUPPLIED_KM", "RESIDUE_IMPORT_KM"]:
        if col in manure_out.columns:
            manure_out.loc[zero_manure_mask, col] = 0.0

    if "LINKED_DIGEST_FACILITY" in crop_out.columns:
        dig_use = crop_out.copy()
        dig_use["YEAR"] = pd.to_numeric(dig_use.get("YEAR", 0), errors="coerce").fillna(0).astype(int)
        dig_use["DIGEST_KG"] = pd.to_numeric(dig_use.get("DIGEST_KG", 0.0), errors="coerce").fillna(0.0)
        dig_use["LINKED_DIGEST_FACILITY"] = dig_use["LINKED_DIGEST_FACILITY"].fillna("").astype(str)
        fac_dig_use = (
            dig_use[(dig_use["LINKED_DIGEST_FACILITY"] != "") & (dig_use["DIGEST_KG"] > 0)]
            .groupby(["LINKED_DIGEST_FACILITY", "YEAR"], as_index=False)["DIGEST_KG"]
            .sum()
            .rename(columns={"LINKED_DIGEST_FACILITY": "FACILITY", "DIGEST_KG": "DIGEST_USED_KG"})
        )
        waste_out = waste_out.merge(fac_dig_use, on=["FACILITY", "YEAR"], how="left")
    else:
        waste_out["DIGEST_USED_KG"] = 0.0

    waste_out["DIGEST_USED_KG"] = pd.to_numeric(waste_out.get("DIGEST_USED_KG", 0.0), errors="coerce").fillna(0.0)
    waste_out["MWAD_DIGEST_KG"] = pd.to_numeric(waste_out.get("MWAD_DIGEST_KG", 0.0), errors="coerce").fillna(0.0)
    waste_out["DIGEST_UNUSED_KG"] = np.maximum(waste_out["MWAD_DIGEST_KG"] - waste_out["DIGEST_USED_KG"], 0.0)
    if "TYPE" in waste_out.columns:
        waste_out.loc[waste_out["TYPE"].astype(str).str.upper() != "MWAD", "DIGEST_UNUSED_KG"] = 0.0
    waste_out = waste_out.drop(columns=["DIGEST_USED_KG"], errors="ignore")

    try:
        cap_diag = waste_out.copy()
        if not cap_diag.empty and {"FACILITY", "YEAR", "TYPE", "CAPACITY"}.issubset(cap_diag.columns):
            cap_diag["FACILITY"] = cap_diag["FACILITY"].astype(str)
            cap_diag["YEAR"] = pd.to_numeric(cap_diag["YEAR"], errors="coerce").fillna(0).astype(int)
            cap_diag["TYPE"] = cap_diag["TYPE"].astype(str).str.upper().str.strip()
            cap_diag["BASE"] = cap_diag["FACILITY"].map(fac_base)
            for c in ["CAPACITY", "WASTE_KG", "INC_INPUT_KG", "RDF_KG", "MWAD_DIGEST_KG", "TDAD_DIGEST_KG"]:
                if c not in cap_diag.columns:
                    cap_diag[c] = 0.0
                cap_diag[c] = pd.to_numeric(cap_diag[c], errors="coerce").fillna(0.0)
            ad_by_base = (
                cap_diag[cap_diag["TYPE"].isin(["MWAD", "TDAD"])]
                .groupby(["BASE", "YEAR"], as_index=False)["WASTE_KG"].sum()
                .rename(columns={"WASTE_KG": "AD_DIVERTED_MSW_KG"})
            )
            base_total = (
                cap_diag[cap_diag["TYPE"].isin(["INC", "RDF", "MWAD", "TDAD"])]
                .groupby(["BASE", "YEAR"], as_index=False)["WASTE_KG"].sum()
                .rename(columns={"WASTE_KG": "TOTAL_MSW_FRAMEWORK_KG"})
            )
            cap_diag = cap_diag[cap_diag["TYPE"].isin(["INC", "RDF"])].copy()
            if not cap_diag.empty:
                before = (
                    cap_diag.groupby(["FACILITY", "TYPE"], as_index=False)["CAPACITY"].max()
                    .rename(columns={"CAPACITY": "CAPACITY_BEFORE_ADJUST_KG"})
                )
                cap_diag = cap_diag.merge(before, on=["FACILITY", "TYPE"], how="left")
                cap_diag = cap_diag.merge(ad_by_base, on=["BASE", "YEAR"], how="left")
                cap_diag = cap_diag.merge(base_total, on=["BASE", "YEAR"], how="left")
                cap_diag["AD_DIVERTED_MSW_KG"] = pd.to_numeric(cap_diag.get("AD_DIVERTED_MSW_KG", 0.0), errors="coerce").fillna(0.0)
                cap_diag["TOTAL_MSW_FRAMEWORK_KG"] = pd.to_numeric(cap_diag.get("TOTAL_MSW_FRAMEWORK_KG", 0.0), errors="coerce").fillna(0.0)
                cap_diag["RESIDUAL_MSW_KG"] = np.maximum(cap_diag["TOTAL_MSW_FRAMEWORK_KG"] - cap_diag["AD_DIVERTED_MSW_KG"], 0.0)
                cap_diag["CAPACITY_AFTER_ADJUST_KG"] = cap_diag["CAPACITY"]
                cap_diag["CAPACITY_REDUCTION_KG"] = np.maximum(cap_diag["CAPACITY_BEFORE_ADJUST_KG"] - cap_diag["CAPACITY_AFTER_ADJUST_KG"], 0.0)
                cap_diag["CAPACITY_REDUCTION_REASON"] = np.where(
                    cap_diag["AD_DIVERTED_MSW_KG"] > 0,
                    "AD adoption reduces residual INC/RDF capacity under ALT/OPT inventory logic",
                    "No AD diversion recorded for this base-year",
                )
                diag_cols = [
                    "FACILITY", "YEAR", "TYPE", "BASE",
                    "CAPACITY_BEFORE_ADJUST_KG", "AD_DIVERTED_MSW_KG", "RESIDUAL_MSW_KG",
                    "CAPACITY_AFTER_ADJUST_KG", "CAPACITY_REDUCTION_KG", "CAPACITY_REDUCTION_REASON",
                ]
                cap_diag[diag_cols].to_csv(output_dir / "02_capacity_adjustment.csv", index=False)
    except Exception as exc:
        log(f"Could not write 02_capacity_adjustment.csv: {exc}")

    manure_out = enforce_species_manure_mass_balance(clean_small_numeric(manure_out))
    manure_out = finalize_output_schema(manure_out, MANURE_OUTPUT_COLUMNS)
    crop_out = add_incremental_digest_machinery_units(crop_out)
    crop_out = crop_out.drop(columns=["LINKED_COMPOST_FACILITY", "LINKED_DIGEST_FACILITY"], errors="ignore")
    crop_out = finalize_output_schema(clean_small_numeric(crop_out), CROP_OUTPUT_COLUMNS)
    waste_out = enforce_species_manure_mass_balance(clean_small_numeric(waste_out))
    waste_out = finalize_output_schema(waste_out, WASTE_OUTPUT_COLUMNS)

    manure_out.to_csv(output_dir / "02_inventory_manure.csv", index=False)
    crop_out.to_csv(output_dir / "02_inventory_crop.csv", index=False)
    if len(WASTE_OUTPUT_COLUMNS) > 0:
        waste_out.to_csv(output_dir / "02_inventory_waste.csv", index=False)



def export_optimization_parameters(
    ctx: InventoryContext,
    crop_out: pd.DataFrame,
    manure_out: pd.DataFrame,
    waste_out: pd.DataFrame,
    output_dir: Path,
    *,
    skip: bool = False,
    distance_mode: str = "haversine",
    max_facilities_per_key: int = 5,
) -> None:

    if skip:
        log("Skipping optimization parameter export (--skip-opt-params).")
        return
    distance_mode = str(distance_mode).strip().lower()
    if distance_mode not in {"haversine", "osrm"}:
        raise ValueError("distance_mode must be 'haversine' or 'osrm'")
    max_facilities_per_key = int(max_facilities_per_key) if max_facilities_per_key is not None else 0
    ensure_dir(output_dir)
    shared = build_shared_inputs(ctx)
    key_cent = shared.get("key_centroids", pd.DataFrame()).copy()
    if key_cent.empty:
        log("Skipping optimization parameter export: key centroids unavailable")
        return
    key_cent["KEY"] = key_cent["KEY"].map(normalize_code)
    key_cent["CITY"] = pd.to_numeric(key_cent.get("CITY", np.nan), errors="coerce")

    # Feedstock availability: municipal waste projection by KEY/YEAR + projected manure by species.
    try:
        waste_proj = build_waste_projection(ctx)
    except Exception as exc:
        log(f"Could not build waste projection for optimization parameters: {exc}")
        waste_proj = pd.DataFrame()

    try:
        biowaste_lookup = build_biowaste_lookup(shared["biowaste"])
        city_proj = project_city_species_livestock(shared["livestock"], ctx.years)
        livestock_key = distribute_city_species_to_keys(city_proj, shared["rcom"])
        manure_sources = build_manure_sources(livestock_key, biowaste_lookup)
    except Exception as exc:
        log(f"Could not build manure projection for optimization parameters: {exc}")
        manure_sources = pd.DataFrame()

    keys_year = []
    for key in key_cent["KEY"].astype(str).unique().tolist():
        for year in ctx.years:
            keys_year.append({"KEY": key, "YEAR": int(year)})
    feed = pd.DataFrame(keys_year)
    if not waste_proj.empty:
        wp = waste_proj.copy()
        wp["KEY"] = wp["KEY"].map(normalize_code)
        keep = ["KEY", "YEAR", "FOOD_KG", "PAPER_KG", "WOOD_KG", "PLASTIC_KG", "SYNTEX_KG", "NATTEX_KG", "WASTE_KG"]
        for col in keep:
            if col not in wp.columns:
                wp[col] = 0.0
        wp = wp[keep]
        feed = feed.merge(wp, on=["KEY", "YEAR"], how="left")
    for col in ["FOOD_KG", "PAPER_KG", "WOOD_KG", "PLASTIC_KG", "SYNTEX_KG", "NATTEX_KG", "WASTE_KG"]:
        if col not in feed.columns:
            feed[col] = 0.0
        feed[col] = pd.to_numeric(feed[col], errors="coerce").fillna(0.0)
    feed["PLA_KG"] = feed["PLASTIC_KG"]
    feed["TEX_KG"] = feed["SYNTEX_KG"] + feed["NATTEX_KG"]
    feed["OTH_KG"] = np.maximum(feed["WASTE_KG"] - feed[["FOOD_KG", "PAPER_KG", "WOOD_KG", "PLASTIC_KG", "SYNTEX_KG", "NATTEX_KG"]].sum(axis=1), 0.0)

    if not manure_sources.empty:
        ms = manure_sources.copy()
        ms["KEY"] = ms["KEY"].map(normalize_code)
        ms["YEAR"] = pd.to_numeric(ms["YEAR"], errors="coerce").fillna(0).astype(int)
        ms["SPECIES"] = ms["SPECIES"].map(normalize_species)
        piv = ms.pivot_table(index=["KEY", "YEAR"], columns="SPECIES", values="MANURE_KG", aggfunc="sum", fill_value=0.0).reset_index()
        piv.columns.name = None
        rename = {sp: f"MANURE_{sp}_KG" for sp in ["DAIRY", "CATTLE", "SWINE", "CHICKEN", "BROILER"]}
        piv = piv.rename(columns=rename)
        feed = feed.merge(piv, on=["KEY", "YEAR"], how="left")
    for col in ["MANURE_DAIRY_KG", "MANURE_CATTLE_KG", "MANURE_SWINE_KG", "MANURE_CHICKEN_KG", "MANURE_BROILER_KG"]:
        if col not in feed.columns:
            feed[col] = 0.0
        feed[col] = pd.to_numeric(feed[col], errors="coerce").fillna(0.0)
    feed["MANURE_TOTAL_KG"] = feed[["MANURE_DAIRY_KG", "MANURE_CATTLE_KG", "MANURE_SWINE_KG", "MANURE_CHICKEN_KG", "MANURE_BROILER_KG"]].sum(axis=1)
    feed = feed.merge(key_cent[["KEY", "CITY", "CENT_LON", "CENT_LAT"]], on="KEY", how="left")
    # Optimization starts in 2030; do not export the internal 2029 carryover year.
    feed = feed[pd.to_numeric(feed["YEAR"], errors="coerce").fillna(0).astype(int) >= OUTPUT_START_YEAR].copy()
    feed.to_csv(output_dir / "05_param_feedstock.csv", index=False)
    log(f"Wrote 05_param_feedstock.csv: {len(feed):,} rows")

    manurefac = shared.get("manurefac", pd.DataFrame()).copy()
    mswfac = shared.get("mswfac", pd.DataFrame()).copy()
    rows: list[dict[str, Any]] = []
    if not manurefac.empty:
        manurefac["FACILITY"] = manurefac["FACILITY"].astype(str)
        manurefac["TYPE"] = manurefac["TYPE"].astype(str).str.upper().str.strip()
        for fac, g in manurefac.groupby("FACILITY", dropna=False):
            rows.append({
                "FACILITY": str(fac), "SOURCE": "MANUREFAC",
                "LAT": pd.to_numeric(g.get("LAT", pd.Series([np.nan])).dropna().iloc[0] if g.get("LAT", pd.Series(dtype=float)).notna().any() else np.nan, errors="coerce"),
                "LON": pd.to_numeric(g.get("LON", pd.Series([np.nan])).dropna().iloc[0] if g.get("LON", pd.Series(dtype=float)).notna().any() else np.nan, errors="coerce"),
                "CITY": pd.to_numeric(g.get("CITY", pd.Series([-1])).dropna().iloc[0] if g.get("CITY", pd.Series(dtype=float)).notna().any() else -1, errors="coerce"),
                "BASE_TYPE": "COMPOST" if (g["TYPE"] == "COMPOST").any() else str(g["TYPE"].iloc[0]),
                "OPEN_YEAR": int(pd.to_numeric(g.get("CONST_YEAR", pd.Series([2030])), errors="coerce").fillna(2030).min()) if "CONST_YEAR" in g.columns else 2030,
                "BASE_CAPACITY_KG": pd.to_numeric(g.get("CAPACITY", pd.Series([0.0])), errors="coerce").fillna(0.0).sum(),
                "IS_INC_SITE": 0,
                "ALLOW_COMPOST": 1, "ALLOW_MWAD": 1, "ALLOW_TDAD": 1, "ALLOW_INC": 0, "ALLOW_RDF": 0,
                "REBUILD_YEAR": np.nan, "RENOV_YEAR": np.nan, "EVENT": "",
            })
    if not mswfac.empty:
        mswfac["FACILITY"] = mswfac["FACILITY"].astype(str)
        mswfac["TYPE"] = mswfac["TYPE"].astype(str).str.upper().str.strip()
        for fac, g in mswfac.groupby("FACILITY", dropna=False):
            base_type = str(g["TYPE"].iloc[0])
            is_landfill = int((g["TYPE"] == "LANDFILL").any())
            rows.append({
                "FACILITY": str(fac), "SOURCE": "MSWFAC",
                "LAT": pd.to_numeric(g.get("LAT", pd.Series([np.nan])).dropna().iloc[0] if g.get("LAT", pd.Series(dtype=float)).notna().any() else np.nan, errors="coerce"),
                "LON": pd.to_numeric(g.get("LON", pd.Series([np.nan])).dropna().iloc[0] if g.get("LON", pd.Series(dtype=float)).notna().any() else np.nan, errors="coerce"),
                "CITY": pd.to_numeric(g.get("CITY", pd.Series([-1])).dropna().iloc[0] if g.get("CITY", pd.Series(dtype=float)).notna().any() else -1, errors="coerce"),
                "BASE_TYPE": base_type,
                "OPEN_YEAR": int(pd.to_numeric(g.get("CONST_YEAR", pd.Series([2030])), errors="coerce").fillna(2030).min()) if "CONST_YEAR" in g.columns else 2030,
                "BASE_CAPACITY_KG": pd.to_numeric(g.get("CAPACITY", pd.Series([0.0])), errors="coerce").fillna(0.0).sum(),
                "IS_INC_SITE": int((g["TYPE"] == "INC").any()),
                "ALLOW_COMPOST": 0, "ALLOW_MWAD": 0 if is_landfill else 1, "ALLOW_TDAD": 0 if is_landfill else 1,
                "ALLOW_INC": 1 if base_type == "INC" else 0, "ALLOW_RDF": 1 if base_type == "RDF" else 0,
                "REBUILD_YEAR": pd.to_numeric(g.get("REBUILD_YEAR", pd.Series([np.nan])), errors="coerce").dropna().min() if "REBUILD_YEAR" in g.columns and not pd.to_numeric(g.get("REBUILD_YEAR", pd.Series([np.nan])), errors="coerce").dropna().empty else np.nan,
                "RENOV_YEAR": pd.to_numeric(g.get("RENOV_YEAR", pd.Series([np.nan])), errors="coerce").dropna().min() if "RENOV_YEAR" in g.columns and not pd.to_numeric(g.get("RENOV_YEAR", pd.Series([np.nan])), errors="coerce").dropna().empty else np.nan,
                "EVENT": event_from_years(
                    pd.to_numeric(g.get("REBUILD_YEAR", pd.Series([np.nan])), errors="coerce").dropna().min() if "REBUILD_YEAR" in g.columns and not pd.to_numeric(g.get("REBUILD_YEAR", pd.Series([np.nan])), errors="coerce").dropna().empty else np.nan,
                    pd.to_numeric(g.get("RENOV_YEAR", pd.Series([np.nan])), errors="coerce").dropna().min() if "RENOV_YEAR" in g.columns and not pd.to_numeric(g.get("RENOV_YEAR", pd.Series([np.nan])), errors="coerce").dropna().empty else np.nan,
                ),
            })
    fac_param = pd.DataFrame(rows)
    if not fac_param.empty:
        fac_param = fac_param.groupby("FACILITY", as_index=False).agg({
            "SOURCE": lambda s: "+".join(sorted(set(map(str, s)))),
            "LAT": "first", "LON": "first", "CITY": "first", "BASE_TYPE": "first", "OPEN_YEAR": "min", "BASE_CAPACITY_KG": "sum",
            "IS_INC_SITE": "max", "ALLOW_COMPOST": "max", "ALLOW_MWAD": "max", "ALLOW_TDAD": "max", "ALLOW_INC": "max", "ALLOW_RDF": "max",
            "REBUILD_YEAR": "min", "RENOV_YEAR": "min", "EVENT": "first",
        })
        fac_param.loc[fac_param["ALLOW_COMPOST"].eq(1), "BASE_TYPE"] = "COMPOST"
        # Landfills are not optimization candidates.
        fac_param = fac_param[~fac_param["BASE_TYPE"].astype(str).str.upper().eq("LANDFILL")].copy()
        # Treat same-coordinate MSW units as one candidate for optimization parameters.
        msw_mask = fac_param["SOURCE"].astype(str).str.contains("MSWFAC", case=False, na=False)
        if msw_mask.any():
            non_msw = fac_param.loc[~msw_mask].copy()
            msw = fac_param.loc[msw_mask].copy()
            msw["_LAT_KEY"] = pd.to_numeric(msw["LAT"], errors="coerce").round(7)
            msw["_LON_KEY"] = pd.to_numeric(msw["LON"], errors="coerce").round(7)
            grouped_rows = []
            for _, g in msw.groupby(["_LAT_KEY", "_LON_KEY", "BASE_TYPE"], dropna=False):
                r = g.iloc[0].copy()
                r["FACILITY"] = fac_base(str(r["FACILITY"]))
                r["SOURCE"] = "+".join(sorted(set(map(str, g["SOURCE"]))))
                r["BASE_CAPACITY_KG"] = pd.to_numeric(g["BASE_CAPACITY_KG"], errors="coerce").fillna(0.0).sum()
                r["OPEN_YEAR"] = pd.to_numeric(g["OPEN_YEAR"], errors="coerce").fillna(2030).min()
                for col in ["IS_INC_SITE", "ALLOW_COMPOST", "ALLOW_MWAD", "ALLOW_TDAD", "ALLOW_INC", "ALLOW_RDF"]:
                    r[col] = pd.to_numeric(g[col], errors="coerce").fillna(0.0).max()
                for col in ["REBUILD_YEAR", "RENOV_YEAR"]:
                    if col in g.columns:
                        vals = pd.to_numeric(g[col], errors="coerce").dropna()
                        r[col] = vals.min() if not vals.empty else np.nan
                r["EVENT"] = event_from_years(r.get("REBUILD_YEAR", np.nan), r.get("RENOV_YEAR", np.nan))
                grouped_rows.append(r.drop(labels=["_LAT_KEY", "_LON_KEY"], errors="ignore"))
            fac_param = pd.concat([non_msw, pd.DataFrame(grouped_rows)], ignore_index=True)
        fac_param = fac_param.drop_duplicates(subset=["FACILITY"], keep="last").reset_index(drop=True)
    fac_param.to_csv(output_dir / "05_param_facility.csv", index=False)
    log(f"Wrote 05_param_facility.csv: {len(fac_param):,} rows")

    dist_rows: list[dict[str, Any]] = []
    city_office_map = shared.get("city_office_map", {})
    if not fac_param.empty:
        key_map = {
            str(r["KEY"]): (float(r["CENT_LON"]), float(r["CENT_LAT"]), r.get("CITY", np.nan))
            for _, r in key_cent.iterrows()
            if pd.notna(r.get("CENT_LON", np.nan)) and pd.notna(r.get("CENT_LAT", np.nan))
        }
        fac_rows = [r for _, r in fac_param.iterrows() if pd.notna(r.get("LON", np.nan)) and pd.notna(r.get("LAT", np.nan))]
        log(f"Building 05_param_distance.csv using {distance_mode} distance; keys={len(key_map):,}, facilities={len(fac_rows):,}, nearest={max_facilities_per_key or 'all'}")
        for ik, (key, (lon, lat, city)) in enumerate(key_map.items(), start=1):
            tmp = []
            for frow in fac_rows:
                if distance_mode == "osrm":
                    try:
                        d = get_city_relay_distance_km(
                            ctx, f"KEY:{key}", lon, lat, city,
                            f"FAC:{frow['FACILITY']}", float(frow["LON"]), float(frow["LAT"]), frow.get("CITY", np.nan), city_office_map,
                        )
                    except Exception:
                        d = haversine_km(lon, lat, float(frow["LON"]), float(frow["LAT"]))
                else:
                    d = haversine_km(lon, lat, float(frow["LON"]), float(frow["LAT"]))
                tmp.append({"KEY": key, "FACILITY": frow["FACILITY"], "DIST_KM": float(d)})
            tmp.sort(key=lambda r: r["DIST_KM"])
            if max_facilities_per_key and max_facilities_per_key > 0:
                tmp = tmp[:max_facilities_per_key]
            dist_rows.extend(tmp)
            if ik % 500 == 0:
                log(f"  distance parameter progress: {ik:,}/{len(key_map):,} keys")
    pd.DataFrame(dist_rows).to_csv(output_dir / "05_param_distance.csv", index=False)
    log(f"Wrote 05_param_distance.csv: {len(dist_rows):,} rows")

    crop = crop_out.copy()
    if not crop.empty:
        crop["KEY"] = crop["KEY"].map(normalize_code)
        opt_numeric_cols = [
            "YEAR", "EXTENT_HA",
            "SYN_KG", "SYN_KM", "SYN_N_KG", "SYN_P_KG", "SYN_K_KG",
            "COMPOST_IMP_KG", "COMPOST_IMP_KM", "COMPOST_LOC_KG", "COMPOST_LOC_KM",
            "COMPOST_N_KG", "COMPOST_P_KG",
            "DIGEST_KG", "DIGEST_KM", "DIGEST_N_KG", "DIGEST_P_KG", "DIGEST_C_KG",
            "AGROCHEM_KG", "AGROCHEM_KM",
            "AG_DIESEL_L", "AG_DIESEL_MJ", "AG_GASOL_MJ", "AG_KEROS_MJ", "AG_ELEC_KWH",
            "NH3_KG", "NO_KG", "NRUNOFF_KG", "NLEACH_KG",
            "RESIDUE_C_KG", "SAND", "IRRIG_TERM",
        ]
        for col in opt_numeric_cols:
            if col not in crop.columns:
                crop[col] = 0.0
            crop[col] = pd.to_numeric(crop[col], errors="coerce").fillna(0.0)
        crop_param_cols = ["KEY", "CITY", "YEAR", "CROP", "CULTIVAR", "EXTENT_HA", "ORG_FLAG"]
        crop_param = crop[crop_param_cols].copy()
        crop_param["NREQ_KG"] = crop["SYN_N_KG"] + crop["COMPOST_N_KG"] + crop["DIGEST_N_KG"]
        crop_param["P2O5REQ_KG"] = crop["SYN_P_KG"] + crop["COMPOST_P_KG"] + crop["DIGEST_P_KG"]
        crop_param["K2OREQ_KG"] = crop["SYN_K_KG"]
        crop_param["BASE_DIGEST_C_KG"] = crop["DIGEST_C_KG"]
        for col in opt_numeric_cols:
            if col not in {"YEAR", "EXTENT_HA"}:
                crop_param[f"BASE_{col}"] = crop[col]
    else:
        crop_param = pd.DataFrame(columns=[
            "KEY", "CITY", "YEAR", "CROP", "CULTIVAR", "EXTENT_HA", "ORG_FLAG",
            "NREQ_KG", "P2O5REQ_KG", "K2OREQ_KG", "BASE_DIGEST_C_KG",
        ])
    crop_param = crop_param[pd.to_numeric(crop_param.get("YEAR", 0), errors="coerce").fillna(0).astype(int) >= OUTPUT_START_YEAR].copy()
    crop_param.to_csv(output_dir / "05_param_crop_req.csv", index=False)
    log(f"Wrote 05_param_crop_req.csv: {len(crop_param):,} rows")
    log("Exported optimization parameter tables: 05_param_feedstock/facility/distance/crop_req.csv")




def _read_optional_optimizer_csv(path: Path | None, required_cols: list[str] | None = None) -> pd.DataFrame:
    """Read an optimizer output CSV if provided; return empty DataFrame on absence/empty file."""
    if path is None:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        log(f"Could not read optimizer output {path}: {exc}")
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            log(f"Optimizer output {path} missing columns {missing}; ignoring it.")
            return pd.DataFrame()
    return df


def _normalize_crop_key_columns(df: pd.DataFrame, *, key_col: str = "KEY") -> pd.DataFrame:
    out = df.copy()
    if key_col in out.columns:
        out[key_col] = out[key_col].map(normalize_code).astype(str)
    if "TO_KEY" in out.columns:
        out["TO_KEY"] = out["TO_KEY"].map(normalize_code).astype(str)
    if "YEAR" in out.columns:
        out["YEAR"] = pd.to_numeric(out["YEAR"], errors="coerce").fillna(0).astype(int)
    if "CROP" in out.columns:
        out["CROP"] = out["CROP"].astype(str).str.upper().str.strip()
    if "CULTIVAR" in out.columns:
        out["CULTIVAR"] = out["CULTIVAR"].astype(str).str.lower().str.strip()
    return out


def _weighted_average(values: pd.Series, weights: pd.Series, default: float = 0.0) -> float:
    v = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if w.sum() <= 0:
        return float(default)
    return float(np.average(v, weights=w))


def recalculate_crop_after_optimizer(crop_df: pd.DataFrame) -> pd.DataFrame:
    crop_df = crop_df.copy()
    if crop_df.empty:
        return crop_df
    for col in ["SYN_N_KG", "SYN_P_KG", "SYN_K_KG", "COMPOST_N_KG", "COMPOST_P_KG", "COMPOST_K_KG", "DIGEST_N_KG", "DIGEST_P_KG", "DIGEST_K_KG", "DIGEST_C_KG", "DIGEST_KG", "COMPOST_LOC_KG", "COMPOST_IMP_KG", "EXTENT_HA"]:
        if col not in crop_df.columns:
            crop_df[col] = 0.0
        crop_df[col] = pd.to_numeric(crop_df[col], errors="coerce").fillna(0.0)
    crop_df["SYN_KG"] = crop_df["SYN_N_KG"] + crop_df["SYN_P_KG"] + crop_df["SYN_K_KG"]
    crop_df["COMPOST_KG"] = crop_df["COMPOST_LOC_KG"] + crop_df["COMPOST_IMP_KG"]

    # ORG/agrochemical rule: if no synthetic nutrient is used for a row, agrochemical is zero.
    if "AGROCHEM" not in crop_df.columns:
        crop_df["AGROCHEM"] = 0.0
    crop_df["AGROCHEM"] = pd.to_numeric(crop_df["AGROCHEM"], errors="coerce").fillna(0.0)
    crop_df["AGROCHEM_KG"] = np.where(crop_df["SYN_KG"] > 1e-9, crop_df["AGROCHEM"] * 100.0 * crop_df["EXTENT_HA"], 0.0)

    for col in ["DIESEL", "GASOL", "KEROS", "ELEC"]:
        if col not in crop_df.columns:
            crop_df[col] = 0.0
        crop_df[col] = pd.to_numeric(crop_df[col], errors="coerce").fillna(0.0)
    crop_df["AG_DIESEL_L"] = crop_df["DIESEL"] * 100.0 * crop_df["EXTENT_HA"] + crop_df["DIGEST_KG"] * 0.001 * 60.3 * 15 / 60 * 3.6 * DIESEL_L_PER_MJ
    crop_df["AG_DIESEL_MJ"] = crop_df["AG_DIESEL_L"] / DIESEL_L_PER_MJ
    crop_df["AG_GASOL_L"] = crop_df["GASOL"] * 100.0 * crop_df["EXTENT_HA"]
    crop_df["AG_GASOL_MJ"] = crop_df["AG_GASOL_L"] / GASOL_L_PER_MJ
    crop_df["AG_KEROS_L"] = crop_df["KEROS"] * 100.0 * crop_df["EXTENT_HA"]
    crop_df["AG_KEROS_MJ"] = crop_df["AG_KEROS_L"] / KEROS_L_PER_MJ
    crop_df["AG_ELEC_KWH"] = crop_df["ELEC"] * 100.0 * crop_df["EXTENT_HA"]

    crop_df["DIGEST_MACHINERY_UNITS"] = 0.0
    rice_mask_for_mach = crop_df["CROP"].astype(str).str.upper().eq("RICE") if "CROP" in crop_df.columns else pd.Series(False, index=crop_df.index)
    crop_df.loc[rice_mask_for_mach, "DIGEST_MACHINERY_UNITS"] = crop_df.loc[rice_mask_for_mach, "DIGEST_KG"] * 0.001 / 2500.0
    crop_df["DIGEST_MACHINERY_NEW_UNITS"] = 0.0
    crop_df = add_incremental_digest_machinery_units(crop_df)

    required_env = ["AIRTEMP", "WIND", "PRECIP", "TN", "SOILTEMP", "pH", "CEC", "SAND"]
    if any(c not in crop_df.columns for c in required_env):
        return crop_df

    for col in required_env + NH3_ENV_MONTH_COLS:
        if col not in crop_df.columns:
            crop_df[col] = np.nan
        crop_df[col] = pd.to_numeric(crop_df[col], errors="coerce")

    pH = crop_df["pH"].fillna(6.0)
    cec_cmol = crop_df["CEC"].fillna(0.0) / 10.0
    fpH = 0.067 * (pH ** 2) - 0.69 * pH + 0.68
    fCEC = np.select([cec_cmol <= 6, (cec_cmol > 6) & (cec_cmol <= 24), (cec_cmol > 24) & (cec_cmol <= 32), cec_cmol > 32], [0.088, 0.012, 0.163, 0.0], default=0.0)
    air = crop_df["AIRTEMP"].fillna(15.0)
    wind = crop_df["WIND"].fillna(2.0)

    rice_air = crop_df["AIRTEMP_M05"].fillna(air)
    rice_wind = crop_df["WIND_M05"].fillna(wind)
    alpha_rice = np.exp(0.233 * rice_air + 0.0419 * rice_wind) / np.exp(0.233 * air + 0.0419 * wind)
    syn_ef_rice = np.clip(alpha_rice * np.exp(fpH + fCEC + 0.014 - 1.895), 0.0, 0.95)
    dig_ef_rice = np.clip(alpha_rice * np.exp(fpH + fCEC + 0.995 - 1.895), 0.0, 0.95)

    tea_air_cols = [c for c in ["AIRTEMP_M03", "AIRTEMP_M06", "AIRTEMP_M09"] if c in crop_df.columns]
    tea_wind_cols = [c for c in ["WIND_M03", "WIND_M06", "WIND_M09"] if c in crop_df.columns]
    tea_air = crop_df[tea_air_cols].fillna(air).mean(axis=1) if tea_air_cols else air
    tea_wind = crop_df[tea_wind_cols].fillna(wind).mean(axis=1) if tea_wind_cols else wind
    alpha_tea = np.exp(0.233 * tea_air + 0.0419 * tea_wind) / np.exp(0.233 * air + 0.0419 * wind)
    syn_ef_tea = np.clip(alpha_tea * np.exp(fpH + fCEC - 0.045 + 0.014 - 1.895), 0.0, 0.95)
    dig_ef_tea = np.clip(alpha_tea * np.exp(fpH + fCEC - 0.045 + 0.995 - 1.895), 0.0, 0.95)

    veg_air_vals = []
    veg_wind_vals = []
    for _, row in crop_df.iterrows():
        cultivar = str(row.get("CULTIVAR", "")).lower().strip()
        month = {"spring": "M03", "summer": "M05", "autumn": "M09", "winter": "M11"}.get(cultivar, "M05")
        a = pd.to_numeric(row.get(f"AIRTEMP_{month}", row.get("AIRTEMP", AIRTEMP_FALLBACK)), errors="coerce")
        w = pd.to_numeric(row.get(f"WIND_{month}", row.get("WIND", 2.0)), errors="coerce")
        veg_air_vals.append(float(a) if pd.notna(a) and np.isfinite(float(a)) else AIRTEMP_FALLBACK)
        veg_wind_vals.append(float(w) if pd.notna(w) and np.isfinite(float(w)) else 2.0)
    veg_air = np.asarray(veg_air_vals, dtype=float)
    veg_wind = np.asarray(veg_wind_vals, dtype=float)
    alpha_veg = np.exp(0.233 * veg_air + 0.0419 * veg_wind) / np.exp(0.233 * air + 0.0419 * wind)
    syn_ef_veg = np.clip(alpha_veg * np.exp(fpH + fCEC - 0.045 + 0.014 - 1.895), 0.0, 0.95)
    dig_ef_veg = np.clip(alpha_veg * np.exp(fpH + fCEC - 0.045 + 0.995 - 1.895), 0.0, 0.95)

    crop_df["NH3_KG"] = 0.0
    rice_mask = crop_df["CROP"].astype(str).str.upper().eq("RICE")
    tea_mask = crop_df["CROP"].astype(str).str.upper().eq("TEA")
    veg_mask = crop_df["CROP"].astype(str).str.upper().eq("VEG")
    crop_df.loc[rice_mask, "NH3_KG"] = crop_df.loc[rice_mask, "SYN_N_KG"] * syn_ef_rice[rice_mask] + crop_df.loc[rice_mask, "COMPOST_N_KG"] * 0.0155 + crop_df.loc[rice_mask, "DIGEST_N_KG"] * dig_ef_rice[rice_mask]
    crop_df.loc[tea_mask, "NH3_KG"] = crop_df.loc[tea_mask, "SYN_N_KG"] * syn_ef_tea[tea_mask] + crop_df.loc[tea_mask, "COMPOST_N_KG"] * 0.013175 + crop_df.loc[tea_mask, "DIGEST_N_KG"] * dig_ef_tea[tea_mask]
    veg_extent = crop_df.loc[veg_mask, "EXTENT_HA"].replace(0, np.nan)
    crop_df.loc[veg_mask, "NH3_KG"] = crop_df.loc[veg_mask, "SYN_N_KG"] * syn_ef_veg[veg_mask] + (0.002 * safe_div(crop_df.loc[veg_mask, "COMPOST_N_KG"], veg_extent, 0.0) + 0.64) * crop_df.loc[veg_mask, "EXTENT_HA"] + crop_df.loc[veg_mask, "DIGEST_N_KG"] * dig_ef_veg[veg_mask]

    total_n = crop_df["SYN_N_KG"] + crop_df["COMPOST_N_KG"] + crop_df["DIGEST_N_KG"]
    crop_df["NH3_KG"] = np.minimum(crop_df["NH3_KG"], total_n)
    crop_df["NO_KG"] = 0.0
    crop_df.loc[rice_mask, "NO_KG"] = total_n[rice_mask] * 0.0012
    crop_df.loc[tea_mask, "NO_KG"] = total_n[tea_mask] * 0.0154
    crop_df.loc[veg_mask, "NO_KG"] = crop_df.loc[veg_mask, "EXTENT_HA"] * 0.1228 * np.exp(0.3869 * crop_df.loc[veg_mask, "SOILTEMP"].fillna(15.0) * safe_div(total_n[veg_mask] * 0.001, crop_df.loc[veg_mask, "EXTENT_HA"], 0.0))

    crop_df["NRUNOFF_KG"] = 0.0
    crop_df["NLEACH_KG"] = 0.0
    N_deposit_rice = crop_df["EXTENT_HA"] * 100.0 * 0.062
    irrig_term = np.maximum(26.1 * 113 - crop_df["PRECIP"].fillna(0.0), 0.0)
    irrig_vol = irrig_term * 0.001 * crop_df["EXTENT_HA"] * 10000.0
    irrig_N_input = irrig_vol * crop_df["TN"].fillna(0.0) * 0.001
    unit_N_rice = safe_div(total_n + N_deposit_rice + irrig_N_input - crop_df["NH3_KG"], crop_df["EXTENT_HA"], 0.0)
    residue_kg_for_leach = pd.to_numeric(crop_df.get("RESIDUE_KG", pd.Series(0.0, index=crop_df.index)), errors="coerce").fillna(0.0)
    runleach_rice = np.where(
        residue_kg_for_leach > 0,
        np.exp(-11.73 + 0.79 * crop_df["SOILTEMP"].fillna(15.0) + 0.00793 * unit_N_rice + 0.000220 * irrig_term + 0.43),
        np.exp(-11.73 + 0.79 * crop_df["SOILTEMP"].fillna(15.0) + 0.00793 * unit_N_rice + 0.000220 * irrig_term),
    ) * crop_df["EXTENT_HA"]
    crop_df.loc[rice_mask, "NRUNOFF_KG"] = runleach_rice[rice_mask] * (1.0 / 7.0)
    crop_df.loc[rice_mask, "NLEACH_KG"] = runleach_rice[rice_mask] * (6.0 / 7.0)
    unit_N_tea = safe_div(total_n, crop_df["EXTENT_HA"], 0.0)
    runoff_tea = 1.07 + 0.00335 * unit_N_tea
    leach_tea = np.where(unit_N_tea < 178, 1.51 + 0.03292 * unit_N_tea, 7.37 + 0.25204 * (unit_N_tea - 178))
    crop_df.loc[tea_mask, "NRUNOFF_KG"] = runoff_tea[tea_mask] * crop_df.loc[tea_mask, "EXTENT_HA"]
    crop_df.loc[tea_mask, "NLEACH_KG"] = leach_tea[tea_mask] * crop_df.loc[tea_mask, "EXTENT_HA"]
    crop_df.loc[veg_mask, "NLEACH_KG"] = total_n[veg_mask] * 0.0007
    crop_df.loc[veg_mask, "NRUNOFF_KG"] = total_n[veg_mask] * 0.0007 * 11.7 / 56.1
    crop_df.loc[tea_mask | veg_mask, ["NRUNOFF_KG", "NLEACH_KG"]] = crop_df.loc[tea_mask | veg_mask, ["NRUNOFF_KG", "NLEACH_KG"]].clip(lower=0.0)
    rice_available_n = total_n + N_deposit_rice + irrig_N_input
    crop_df.loc[rice_mask, "NH3_KG"], crop_df.loc[rice_mask, "NO_KG"], crop_df.loc[rice_mask, "NRUNOFF_KG"] = cap_n_loss_pathways(
        crop_df.loc[rice_mask, "NH3_KG"], crop_df.loc[rice_mask, "NO_KG"], crop_df.loc[rice_mask, "NRUNOFF_KG"], rice_available_n[rice_mask] - crop_df.loc[rice_mask, "NLEACH_KG"])
    crop_df.loc[tea_mask | veg_mask, "NH3_KG"], crop_df.loc[tea_mask | veg_mask, "NO_KG"], crop_df.loc[tea_mask | veg_mask, "NLEACH_KG"] = cap_n_loss_pathways(
        crop_df.loc[tea_mask | veg_mask, "NH3_KG"], crop_df.loc[tea_mask | veg_mask, "NO_KG"], crop_df.loc[tea_mask | veg_mask, "NLEACH_KG"], total_n[tea_mask | veg_mask])
    crop_df["IRRIG_TERM"] = irrig_term
    return crop_df


def apply_optimized_agriculture_outputs(ctx: InventoryContext, crop_df: pd.DataFrame) -> pd.DataFrame:
    if crop_df.empty:
        return crop_df
    fert = _read_optional_optimizer_csv(ctx.scenario.fertilizer_links_path, ["YEAR", "TO_KEY", "CROP", "CULTIVAR"])
    dig = _read_optional_optimizer_csv(ctx.scenario.digestate_links_path, ["YEAR", "TO_KEY", "CROP", "CULTIVAR"])
    comp_supply = _read_optional_optimizer_csv(ctx.scenario.compost_supply_path)
    fallback = _read_optional_optimizer_csv(ctx.scenario.fallback_flows_path)
    if not comp_supply.empty:
        comp_supply.to_csv(ctx.paths.output_dir / "02_optimizer_compost_supply.csv", index=False)
    if not fallback.empty:
        fallback.to_csv(ctx.paths.output_dir / "02_optimizer_fallback_flows.csv", index=False)
    if fert.empty and dig.empty:
        return crop_df

    out = _normalize_crop_key_columns(crop_df, key_col="KEY")

    def _expand_all_cultivar_optimizer_rows(df: pd.DataFrame, additive_cols: list[str]) -> pd.DataFrame:
        if df.empty or "CULTIVAR" not in df.columns:
            return df
        tmp = _normalize_crop_key_columns(df, key_col="KEY")
        all_mask = tmp["CULTIVAR"].astype(str).str.upper().isin(["ALL", "*", "__ALL__"])
        if not all_mask.any():
            return tmp
        direct = tmp.loc[~all_mask].copy()
        all_rows = tmp.loc[all_mask].copy().reset_index(drop=True)
        all_rows["_ALL_ROW_ID"] = np.arange(len(all_rows))
        base = out[["KEY", "YEAR", "CROP", "CULTIVAR", "EXTENT_HA"]].copy()
        base["EXTENT_HA"] = pd.to_numeric(base.get("EXTENT_HA", 0.0), errors="coerce").fillna(0.0)
        expanded = all_rows.merge(base, on=["KEY", "YEAR", "CROP"], how="left", suffixes=("", "_INV"))
        expanded["CULTIVAR"] = expanded["CULTIVAR_INV"].fillna(expanded["CULTIVAR"]).astype(str).str.lower()
        denom = expanded.groupby("_ALL_ROW_ID")["EXTENT_HA"].transform("sum")
        nmatch = expanded.groupby("_ALL_ROW_ID")["CULTIVAR"].transform("count").replace(0, 1)
        share = np.where(denom > 0, expanded["EXTENT_HA"] / denom, 1.0 / nmatch)
        for col in additive_cols:
            if col in expanded.columns:
                expanded[col] = pd.to_numeric(expanded[col], errors="coerce").fillna(0.0) * share
        expanded = expanded.drop(columns=["_ALL_ROW_ID", "CULTIVAR_INV", "EXTENT_HA"], errors="ignore")
        return pd.concat([direct, expanded], ignore_index=True, sort=False)

    out["_OPT_KEY"] = list(zip(out["KEY"].astype(str), out["YEAR"].astype(int), out["CROP"].astype(str).str.upper(), out["CULTIVAR"].astype(str).str.lower()))
    opt_keys: set[tuple[str, int, str, str]] = set()

    key_cols_out = ["KEY", "YEAR", "CROP", "CULTIVAR"]
    if not fert.empty:
        fert = _normalize_crop_key_columns(fert.rename(columns={"TO_KEY": "KEY"}), key_col="KEY")
        fert = _expand_all_cultivar_optimizer_rows(fert, ["SYN_N_KG", "SYN_P_KG", "SYN_K_KG", "COMPOST_IMP_KG", "COMPOST_LOC_KG", "MATERIAL_KG"])
        for col in ["SYN_N_KG", "SYN_P_KG", "SYN_K_KG", "COMPOST_IMP_KG", "COMPOST_LOC_KG"]:
            if col not in fert.columns:
                fert[col] = 0.0
            fert[col] = pd.to_numeric(fert[col], errors="coerce").fillna(0.0)
        fert_g = fert.groupby(key_cols_out, as_index=False)[["SYN_N_KG", "SYN_P_KG", "SYN_K_KG", "COMPOST_IMP_KG", "COMPOST_LOC_KG"]].sum()
        opt_keys.update(set(zip(fert_g["KEY"].astype(str), fert_g["YEAR"].astype(int), fert_g["CROP"].astype(str), fert_g["CULTIVAR"].astype(str))))
        out = out.merge(fert_g.add_suffix("_OPT").rename(columns={"KEY_OPT": "KEY", "YEAR_OPT": "YEAR", "CROP_OPT": "CROP", "CULTIVAR_OPT": "CULTIVAR"}), on=key_cols_out, how="left")
    else:
        for col in ["SYN_N_KG_OPT", "SYN_P_KG_OPT", "SYN_K_KG_OPT", "COMPOST_IMP_KG_OPT", "COMPOST_LOC_KG_OPT"]:
            out[col] = np.nan

    if not dig.empty:
        dig = _normalize_crop_key_columns(dig.rename(columns={"TO_KEY": "KEY"}), key_col="KEY")
        dig = _expand_all_cultivar_optimizer_rows(dig, ["DIGEST_KG", "DIGEST_N_KG", "DIGEST_P_KG", "DIGEST_K_KG", "DIGEST_C_KG", "DIGEST_MACHINERY_UNITS"])
        for col in ["DIGEST_KG", "DIGEST_N_KG", "DIGEST_P_KG", "DIGEST_K_KG", "DIGEST_C_KG", "DIGEST_KM", "DIGEST_MACHINERY_UNITS"]:
            if col not in dig.columns:
                dig[col] = 0.0
            dig[col] = pd.to_numeric(dig[col], errors="coerce").fillna(0.0)
        rows = []
        for keys, g in dig.groupby(key_cols_out, dropna=False):
            kg = g["DIGEST_KG"]
            rows.append({
                "KEY": keys[0], "YEAR": int(keys[1]), "CROP": keys[2], "CULTIVAR": keys[3],
                "DIGEST_KG_OPT": float(g["DIGEST_KG"].sum()),
                "DIGEST_N_KG_OPT": float(g["DIGEST_N_KG"].sum()),
                "DIGEST_P_KG_OPT": float(g["DIGEST_P_KG"].sum()),
                "DIGEST_K_KG_OPT": float(g["DIGEST_K_KG"].sum()),
                "DIGEST_C_KG_OPT": float(g["DIGEST_C_KG"].sum()),
                "DIGEST_KM_OPT": _weighted_average(g["DIGEST_KM"], kg, 0.0),
                "DIGEST_MACHINERY_UNITS_OPT": float(g["DIGEST_MACHINERY_UNITS"].sum()),
                "LINKED_DIGEST_FACILITY_OPT": str(g.get("FROM_FACILITY", pd.Series([""])).iloc[0]) if "FROM_FACILITY" in g.columns else "",
            })
        dig_g = pd.DataFrame(rows)
        opt_keys.update(set(zip(dig_g["KEY"].astype(str), dig_g["YEAR"].astype(int), dig_g["CROP"].astype(str), dig_g["CULTIVAR"].astype(str))))
        out = out.merge(dig_g, on=key_cols_out, how="left")
    else:
        for col in ["DIGEST_KG_OPT", "DIGEST_N_KG_OPT", "DIGEST_P_KG_OPT", "DIGEST_K_KG_OPT", "DIGEST_C_KG_OPT", "DIGEST_KM_OPT", "DIGEST_MACHINERY_UNITS_OPT"]:
            out[col] = np.nan
        out["LINKED_DIGEST_FACILITY_OPT"] = ""

    if not opt_keys:
        cleaned = out.drop(columns=[c for c in out.columns if c.endswith("_OPT") or c == "_OPT_KEY"], errors="ignore")
        return cleaned
    mask = out["_OPT_KEY"].isin(opt_keys)
    if not mask.any():
        return out.drop(columns=[c for c in out.columns if c.endswith("_OPT") or c == "_OPT_KEY"], errors="ignore")

    baseline_org_y_mask = pd.Series(False, index=out.index)
    if str(ctx.scenario.org_mode).lower().strip() == "baseline" and "ORG_FLAG" in out.columns:
        baseline_org_y_mask = out["ORG_FLAG"].astype(str).str.upper().eq("Y")

    controlled = [
        "SYN_N_KG", "SYN_P_KG", "SYN_K_KG", "SYN_KG",
        "COMPOST_IMP_KG", "COMPOST_LOC_KG", "COMPOST_KG", "COMPOST_N_KG", "COMPOST_P_KG", "COMPOST_K_KG",
        "DIGEST_KG", "DIGEST_N_KG", "DIGEST_P_KG", "DIGEST_K_KG", "DIGEST_C_KG", "DIGEST_KM", "DIGEST_MACHINERY_UNITS", "DIGEST_MACHINERY_NEW_UNITS",
    ]
    for col in controlled:
        if col not in out.columns:
            out[col] = 0.0
        out.loc[mask, col] = 0.0

    for col in ["SYN_N_KG", "SYN_P_KG", "SYN_K_KG", "COMPOST_IMP_KG", "COMPOST_LOC_KG"]:
        opt_col = f"{col}_OPT"
        if opt_col in out.columns:
            out.loc[mask & out[opt_col].notna(), col] = pd.to_numeric(out.loc[mask & out[opt_col].notna(), opt_col], errors="coerce").fillna(0.0)
    for col in ["DIGEST_KG", "DIGEST_N_KG", "DIGEST_P_KG", "DIGEST_K_KG", "DIGEST_C_KG", "DIGEST_KM", "DIGEST_MACHINERY_UNITS"]:
        opt_col = f"{col}_OPT"
        if opt_col in out.columns:
            out.loc[mask & out[opt_col].notna(), col] = pd.to_numeric(out.loc[mask & out[opt_col].notna(), opt_col], errors="coerce").fillna(0.0)
    if "LINKED_DIGEST_FACILITY_OPT" in out.columns:
        out.loc[mask & out["LINKED_DIGEST_FACILITY_OPT"].astype(str).ne(""), "LINKED_DIGEST_FACILITY"] = out.loc[mask & out["LINKED_DIGEST_FACILITY_OPT"].astype(str).ne(""), "LINKED_DIGEST_FACILITY_OPT"]

    for col in ["COMPOST_N_CONC", "COMPOST_P_CONC", "COMPOST_K_CONC"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out.loc[mask, "COMPOST_KG"] = out.loc[mask, "COMPOST_LOC_KG"] + out.loc[mask, "COMPOST_IMP_KG"]
    out.loc[mask, "COMPOST_N_KG"] = out.loc[mask, "COMPOST_LOC_KG"] * out.loc[mask, "COMPOST_N_CONC"] + out.loc[mask, "COMPOST_IMP_KG"] * 0.037
    out.loc[mask, "COMPOST_P_KG"] = out.loc[mask, "COMPOST_KG"] * out.loc[mask, "COMPOST_P_CONC"]
    out.loc[mask, "COMPOST_K_KG"] = out.loc[mask, "COMPOST_KG"] * out.loc[mask, "COMPOST_K_CONC"]
    out.loc[mask, "COMPOST_LOC_KM"] = np.where(out.loc[mask, "COMPOST_LOC_KG"] > 0, out.loc[mask, "COMPOST_LOC_KM"], 0.0)
    out.loc[mask, "COMPOST_IMP_KM"] = np.where(out.loc[mask, "COMPOST_IMP_KG"] > 0, out.loc[mask, "COMPOST_IMP_KM"], 0.0)
    out.loc[mask, "DIGEST_KM"] = np.where(out.loc[mask, "DIGEST_KG"] > 0, out.loc[mask, "DIGEST_KM"], 0.0)

    protected_baseline_org = mask & baseline_org_y_mask
    if protected_baseline_org.any():
        out.loc[protected_baseline_org, ["SYN_N_KG", "SYN_P_KG", "SYN_K_KG", "SYN_KG"]] = 0.0

    out.loc[mask, "ORG_FLAG"] = np.where(out.loc[mask, ["SYN_N_KG", "SYN_P_KG", "SYN_K_KG"]].sum(axis=1) > 1e-9, "X", "Y")
    if protected_baseline_org.any():
        out.loc[protected_baseline_org, "ORG_FLAG"] = "Y"

    out = recalculate_crop_after_optimizer(out)
    log(f"Applied optimized agriculture outputs to {int(mask.sum())} crop rows")
    drop_cols = [c for c in out.columns if c.endswith("_OPT") or c == "_OPT_KEY"]
    return out.drop(columns=drop_cols, errors="ignore")


def force_zero_crop_digestate(crop_df: pd.DataFrame) -> pd.DataFrame:
    out = crop_df.copy()
    zero_cols = [
        "DIGEST_KG",
        "DIGEST_KM",
        "DIGEST_MACHINERY_UNITS",
        "DIGEST_MACHINERY_NEW_UNITS",
        "DIGEST_N_KG",
        "DIGEST_P_KG",
        "DIGEST_C_KG",
        "DIGEST_K_KG",
        "DIGEST_P2O5_KG",
        "DIGEST_K2O_KG",
        "DIGEST_P_ELEM_KG",
        "DIGEST_K_ELEM_KG",
    ]
    for col in zero_cols:
        if col in out.columns:
            out[col] = 0.0
    for col in ["LINKED_DIGEST_FACILITY"]:
        if col in out.columns:
            out[col] = ""
    if "ACTIVE_ORG_SOURCE" in out.columns:
        mask = out["ACTIVE_ORG_SOURCE"].astype(str).str.upper().eq("DIGEST")
        out.loc[mask, "ACTIVE_ORG_SOURCE"] = "COMPOST"
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build inventory datasets for manure, crop, and waste modules.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("."))
    parser.add_argument("--gis-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--cache-dir", type=Path, default=Path("."))
    parser.add_argument("--script-dir", type=Path, default=None)
    parser.add_argument("--scenario", type=str, default="BASE")
    parser.add_argument("--organic-source", type=str, default="COMPOST", choices=["COMPOST", "DIGEST"])
    parser.add_argument("--disable-osrm", action="store_true")
    parser.add_argument("--disable-crop-digestate", action="store_true", help="Force crop-side digestate application to zero. Use this for OPT_MODE=facility_only.")
    parser.add_argument("--osrm-url", type=str, default=DEFAULT_OSRM_URL)
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--facility-plan", type=Path, default=None, help="Path to optimized 06_facility.csv for OPT re-estimation.")
    parser.add_argument("--digestate-links", type=Path, default=None, help="Path to optimized 06_digestate_links.csv for OPT agriculture re-estimation.")
    parser.add_argument("--fertilizer-links", type=Path, default=None, help="Path to optimized 06_fertilizer_links.csv for OPT agriculture re-estimation.")
    parser.add_argument("--compost-supply", type=Path, default=None, help="Path to optimized 06_compost_supply.csv for diagnostics/supply consistency.")
    parser.add_argument("--fallback-flows", type=Path, default=None, help="Path to optimized 06_fallback_flows.csv for diagnostics/fallback consistency.")
    parser.add_argument("--org-mode", type=str, default="baseline", choices=["baseline", "free"], help="Optimization sub-mode metadata for downstream agriculture logic.")
    parser.add_argument("--link-mode", type=str, default="single", choices=["single", "multi"], help="Optimization sub-mode metadata for downstream routing logic.")
    parser.add_argument("--skip-opt-params", action="store_true", help="Skip exporting 05_param_*.csv optimization tables.")
    parser.add_argument("--param-distance-mode", type=str, default="haversine", choices=["haversine", "osrm"], help="Distance method for 05_param_distance.csv only. Default haversine is much faster; final inventory routing still uses OSRM unless --disable-osrm is set.")
    parser.add_argument("--max-param-facilities-per-key", type=int, default=5, help="Keep only nearest N facilities per KEY in 05_param_distance.csv. Use 0 for all. Keep this small for faster 05_optimize.py runs.")
    args = parser.parse_args()
    if args.max_param_facilities_per_key < 0:
        raise ValueError("--max-param-facilities-per-key must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    paths = InventoryPaths(
        dataset_dir=args.dataset_dir,
        gis_dir=args.gis_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        script_dir=args.script_dir if args.script_dir is not None else args.dataset_dir,
    )
    scenario_name = args.scenario.upper().strip()
    scenario = ScenarioSettings(
        scenario_name=scenario_name,
        active_organic_n_source=args.organic_source.upper(),
        use_alt_adfac=is_alt_scenario(scenario_name),
        use_osrm=not args.disable_osrm,
        facility_plan_path=args.facility_plan,
        digestate_links_path=args.digestate_links,
        fertilizer_links_path=args.fertilizer_links,
        compost_supply_path=args.compost_supply,
        fallback_flows_path=args.fallback_flows,
        org_mode=args.org_mode,
        link_mode=args.link_mode,
        disable_crop_digestate=bool(getattr(args, "disable_crop_digestate", False)),
    )
    ctx = InventoryContext(paths=paths, scenario=scenario, osrm_url=args.osrm_url, timeout_sec=args.timeout_sec)
    ensure_dir(paths.output_dir)
    ensure_dir(paths.cache_dir)
    cache_path = paths.cache_dir / "01_inventory_distance_cache.json"
    ctx.distance_cache = load_distance_cache(cache_path)
    log("Loading and validating inputs")
    _ = build_shared_inputs(ctx)
    log("Building preliminary crop inventory")
    crop_prelim = build_crop_inventory(ctx, manure_inventory=None)
    log("Building manure inventory")
    manure = build_manure_inventory(ctx, crop_prelim=crop_prelim)
    log("Rebuilding crop inventory with manure-derived compost/digest supply")
    crop_with_supply = build_crop_inventory(ctx, manure_inventory=manure)
    log("Linking crop residue supply to manure demand")
    crop_final, manure_final = allocate_residue_supply(ctx, crop_with_supply, manure)
    log("Building waste framework inventory")
    waste = build_waste_inventory(ctx, manure_inventory=manure_final)
    if scenario_name == "OPT":
        if bool(getattr(ctx.scenario, "disable_crop_digestate", False)):
            log("Skipping optimized fertilizer/digestate overlay because --disable-crop-digestate is active")
        else:
            log("Applying optimized fertilizer/digestate outputs, if provided")
            crop_final = apply_optimized_agriculture_outputs(ctx, crop_final)
    if bool(getattr(ctx.scenario, "disable_crop_digestate", False)):
        log("Forcing crop-side digestate fields to zero")
        crop_final = force_zero_crop_digestate(crop_final)
    log("Writing outputs (02_inventory_manure.csv, 02_inventory_crop.csv, 02_inventory_waste.csv)")
    write_outputs(manure_final, crop_final, waste, paths.output_dir, scenario_name=scenario_name)
    log("Writing link outputs (02_waste_link.csv, 02_fertilizer_link.csv)")
    # 02_waste_link.csv is written inside build_waste_routing before KEY-level detail
    # is aggregated away.  02_fertilizer_link.csv is written here while crop_final
    # still has LINKED_COMPOST_FACILITY and LINKED_DIGEST_FACILITY.
    try:
        write_fertilizer_link(ctx, crop_final, waste, paths.output_dir, manure_df=manure_final)
    except Exception as exc:
        log(f"Could not write 02_fertilizer_link.csv: {exc}")
    save_distance_cache(ctx.distance_cache, cache_path)
    log("Exporting optimization parameter tables")
    export_optimization_parameters(
        ctx, crop_final, manure_final, waste, paths.output_dir,
        skip=bool(args.skip_opt_params),
        distance_mode=args.param_distance_mode,
        max_facilities_per_key=int(args.max_param_facilities_per_key),
    )
    save_distance_cache(ctx.distance_cache, cache_path)
    log("Done")


if __name__ == "__main__":
    main()
