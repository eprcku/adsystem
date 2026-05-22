
from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

YEARS = list(range(2030, 2051))
FAC_TYPES_OPT = ["COMPOST", "MWAD", "TDAD"]
EXISTING_TYPES = ["INC", "RDF"]
ENERGY_USES = ["ELEC", "GRIDGAS"]
DIESEL_L_PER_MJ = 0.0262868597
BIOGAS_HV_MJ_PER_NM3 = 33.906
KJ_PER_KWH = 3600.0

# 最適化単位調整
#   GWP: kgCO2e -> ktCO2e
#   COST: JPY -> billion JPY
GWP_OBJ_SCALE_TO_KT = 1.0e-6
COST_OBJ_SCALE_TO_BILLION_JPY = 1.0e-9
FOOD_INC_OBJ_SCALE_TO_TONNES = 1.0e-3
MIN_IMPORTED_COMPOST_PENALTY_RAW = 1.0e4
MIN_NUTRIENT_SLACK_PENALTY_RAW = 1.0e6

GUROBI_METHOD_AUTO = -1
GUROBI_METHOD_BARRIER = 2
GUROBI_METHOD_DUAL_SIMPLEX = 1
DEFAULT_GUROBI_METHOD = GUROBI_METHOD_DUAL_SIMPLEX
DEFAULT_GUROBI_NODE_METHOD = GUROBI_METHOD_DUAL_SIMPLEX
GUROBI_PRESOLVE_AGGRESSIVE = 2
GUROBI_SCALEFLAG_AGGRESSIVE = 2
GUROBI_NUMERIC_FOCUS_MIN = 2
DEFAULT_FOOD_INC_ABSTOL_T = 1.0


def target_objective_scale(target: str) -> float:
    """Return scalar used to express the optimizer's target objective in final reporting units."""
    return GWP_OBJ_SCALE_TO_KT if str(target).upper() == "GWP" else COST_OBJ_SCALE_TO_BILLION_JPY


INC_ENERGY_L_PER_KG = 0.00373
INC_ELEC_KWH_PER_KG = 0.0264
DEFAULT_ASH_RATIO = 0.10
MWAD_DIGSOLID_TS_RECOVERY = 0.95
MWAD_DIGSOLID_WATER = 0.75
TDAD_DIGSOLID_TS_RECOVERY = 0.95
TDAD_DIGSOLID_WATER = 0.75


@dataclass(frozen=True)
class OptimizeSettings:
    target: str
    mode: str
    org_mode: str
    link_mode: str
    compost_mode: str
    enable_gridgas: bool
    min_scale: int
    max_scale: int
    scale_step: int
    max_digestate_facilities_per_key: int
    fert_cap_tolerance: float
    imported_compost_penalty: float
    food_inc_abstol_t: float
    gurobi_method: int
    gurobi_node_method: int
    gurobi_crossover: int | None
    numeric_focus: int | None
    mip_focus: int | None
    heuristics: float | None
    time_limit: float | None
    mip_gap: float | None
    threads: int
    log_dir: Path
    cache_dir: Path
    use_model_cache: bool
    rebuild_model_cache: bool
    write_model_mps: bool
    dataset_dir: Path
    input_dir: Path
    output_dir: Path
    script_dir: Path


def compost_candidates_enabled(settings: OptimizeSettings) -> bool:
    return str(settings.compost_mode).lower().strip() == "include"


def log(msg: str) -> None:
    print(f"[05_optimize] {msg}", flush=True)


def configure_gurobi_model(model: Any, settings: OptimizeSettings, model_name: str) -> None:
    """Apply common Gurobi controls and log-file settings."""
    if gp is None or model is None:
        return

    model.Params.Method = int(getattr(settings, "gurobi_method", DEFAULT_GUROBI_METHOD))
    model.Params.NodeMethod = int(getattr(settings, "gurobi_node_method", DEFAULT_GUROBI_NODE_METHOD))
    crossover = getattr(settings, "gurobi_crossover", None)
    if crossover is not None:
        model.Params.Crossover = int(crossover)
    model.Params.Presolve = GUROBI_PRESOLVE_AGGRESSIVE
    model.Params.ScaleFlag = GUROBI_SCALEFLAG_AGGRESSIVE

    if settings.time_limit is not None:
        model.Params.TimeLimit = float(settings.time_limit)
    if settings.mip_gap is not None:
        model.Params.MIPGap = float(settings.mip_gap)
    if settings.threads is not None and settings.threads >= 0:
        model.Params.Threads = int(settings.threads)

    if getattr(settings, "numeric_focus", None) is not None:
        model.Params.NumericFocus = max(int(settings.numeric_focus), GUROBI_NUMERIC_FOCUS_MIN)
    else:
        model.Params.NumericFocus = GUROBI_NUMERIC_FOCUS_MIN

    if getattr(settings, "mip_focus", None) is not None:
        model.Params.MIPFocus = int(settings.mip_focus)
    if getattr(settings, "heuristics", None) is not None:
        model.Params.Heuristics = float(settings.heuristics)
    if settings.log_dir is not None:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in model_name)
        model.Params.LogFile = str(settings.log_dir / f"{safe}.gurobi.log")


def write_iis_if_infeasible(model: Any, settings: OptimizeSettings, model_name: str) -> None:
    """Export a compact IIS .ilp file when Gurobi proves infeasibility."""
    if gp is None or GRB is None or model is None:
        return
    status = int(getattr(model, "Status", -1))
    if status == GRB.INF_OR_UNBD:
        try:
            log(f"{model_name}: status is INF_OR_UNBD; re-optimizing with DualReductions=0 before IIS export")
            model.Params.DualReductions = 0
            model.optimize()
            status = int(getattr(model, "Status", -1))
        except Exception as exc:
            log(f"{model_name}: could not re-optimize for IIS diagnosis: {exc}")
            return
    if status != GRB.INFEASIBLE:
        return
    try:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in model_name)
        iis_path = settings.output_dir / f"{safe}.iis.ilp"
        log(f"{model_name}: computing IIS and writing {iis_path}")
        model.computeIIS()
        model.write(str(iis_path))
    except Exception as exc:
        log(f"{model_name}: IIS export failed: {exc}")


def _file_sig(path: Path) -> dict[str, Any]:
    """Small signature for cache invalidation without reading whole files."""
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {"exists": True, "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def model_cache_key(settings: OptimizeSettings) -> str:
    names = [
        "05_param_feedstock.csv", "05_param_facility.csv", "05_param_distance.csv",
        "02_EF.csv", "02_energy.csv", "00_cost.csv", "00_manurefac.csv", "00_mswfac.csv",
    ]
    if settings.mode == "facility_agriculture":
        names.append("05_param_crop_req.csv")
    payload: dict[str, Any] = {
        "target": settings.target,
        "mode": settings.mode,
        "org_mode": settings.org_mode,
        "link_mode": settings.link_mode,
        "compost_mode": settings.compost_mode,
        "enable_gridgas": settings.enable_gridgas,
        "objective_schema": "material_independent_flows_event_food_inc_slack_compost_food_v1_dualsimplex",
        "min_scale": settings.min_scale,
        "max_scale": settings.max_scale,
        "scale_step": settings.scale_step,
        "max_digestate_facilities_per_key": settings.max_digestate_facilities_per_key,
        "fert_cap_tolerance": settings.fert_cap_tolerance,
        "imported_compost_penalty": settings.imported_compost_penalty,
        "food_inc_abstol_t": settings.food_inc_abstol_t,
        "gurobi_method": settings.gurobi_method,
        "gurobi_node_method": settings.gurobi_node_method,
        "gurobi_crossover": settings.gurobi_crossover,
        "numeric_focus": settings.numeric_focus,
        "mip_focus": settings.mip_focus,
        "heuristics": settings.heuristics,
        "script": _file_sig(Path(__file__).resolve()),
        "files": {},
    }
    for name in names:
        p = settings.input_dir / name
        if not p.exists():
            p = settings.dataset_dir / name
        payload["files"][name] = _file_sig(p)
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def model_cache_paths(settings: OptimizeSettings) -> tuple[Path, Path]:
    key = model_cache_key(settings)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings.cache_dir / f"05_flow_explicit_{key}.mps", settings.cache_dir / f"05_flow_explicit_{key}.json"


def _var_name(prefix: str, *parts: Any) -> str:
    return f"{prefix}[{','.join(map(str, parts))}]"


def save_flow_model_cache(model: Any, aux: dict[str, Any], settings: OptimizeSettings) -> None:
    """Write model MPS + small metadata for solution extraction on later runs."""
    if gp is None or model is None:
        return
    mps_path, meta_path = model_cache_paths(settings)
    meta = {
        "years": aux.get("years", []),
        "keys": aux.get("keys", []),
        "facs": aux.get("facs", []),
        "z_keys": [list(k) for k in aux.get("z", {}).keys()],
        "assign_keys": [list(k) for k in aux.get("assign", {}).keys()],
        "assign_food_keys": [list(k) for k in aux.get("assign_food", {}).keys()],
        "assign_manure_keys": [list(k) for k in aux.get("assign_manure", {}).keys()],
        "assign_pw_keys": [list(k) for k in aux.get("assign_pw", {}).keys()],
        "scale_choice_keys": [list(k) for k in aux.get("scale_choice", {}).keys()],
        "dig_apply_keys": [list(k) for k in aux.get("dig_apply", {}).keys()],
        "syn_apply_keys": [list(k) for k in aux.get("syn_apply", {}).keys()],
        "compost_imp_apply_keys": [list(k) for k in aux.get("compost_imp_apply", {}).keys()],
        "compost_loc_apply_keys": [list(k) for k in aux.get("compost_loc_apply", {}).keys()],
        "nutrient_slack_keys": [list(k) for k in aux.get("nutrient_slack", {}).keys()],
        "dig_unused_keys": [list(k) for k in aux.get("dig_unused", {}).keys()],
        "bio_use_keys": [list(k) for k in aux.get("bio_use", {}).keys()],
        "assign_energy_keys": [list(k) for k in aux.get("assign_energy", {}).keys()],
        "waste_inc_keys": [list(k) for k in aux.get("waste_inc", {}).keys()],
        "food_inc_keys": [list(k) for k in aux.get("food_inc", {}).keys()],
        "event_food_inc_slack_keys": [list(k) for k in aux.get("event_food_inc_slack", {}).keys()],
        "manure_compbase_keys": [list(k) for k in aux.get("manure_compbase", {}).keys()],
        "comp_prod_keys": [list(k) for k in aux.get("comp_prod", {}).keys()],
        "ad_input_keys": [list(k) for k in aux.get("ad_input", {}).keys()],
        "ad_digest_prod_keys": [list(k) for k in aux.get("ad_digest_prod", {}).keys()],
        "ad_coeff": {"|".join(map(str, k)): v for k, v in aux.get("ad_coeff", {}).items()},
        "cache_key": model_cache_key(settings),
    }
    log(f"Writing flow model cache: {mps_path}")
    model.write(str(mps_path))
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_flow_model_cache(settings: OptimizeSettings) -> tuple[Any | None, dict[str, Any] | None]:
    """Load cached model if available. The returned aux uses live Var objects."""
    if gp is None or not settings.use_model_cache or settings.rebuild_model_cache:
        return None, None
    mps_path, meta_path = model_cache_paths(settings)
    if not (mps_path.exists() and meta_path.exists()):
        return None, None
    log(f"Loading cached flow model: {mps_path}")
    model = gp.read(str(mps_path))
    configure_gurobi_model(model, settings, "CE_AD_flow_explicit")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    z = {}
    for f, t in meta.get("z_keys", []):
        v = model.getVarByName(_var_name("Z", f, t))
        if v is not None:
            z[(f, t)] = v
    s = {}
    for f in meta.get("facs", []):
        v = model.getVarByName(_var_name("S", f))
        if v is not None:
            s[f] = v
    assign = {}
    for rec in meta.get("assign_keys", []):
        if len(rec) == 4:
            k, f, t, y = rec
            v = model.getVarByName(_var_name("A", k, f, t, int(y)))
            if v is not None:
                assign[(k, f, t, int(y))] = v
        elif len(rec) == 3:
            k, f, t = rec
            v = model.getVarByName(_var_name("A", k, f, t))
            if v is not None:
                assign[(k, f, t)] = v
    def _load_route_var(prefix: str, rec):
        if len(rec) == 4:
            k, f, t, y = rec
            return (k, f, t, int(y)), model.getVarByName(_var_name(prefix, k, f, t, int(y)))
        if len(rec) == 3:
            k, f, t = rec
            return (k, f, t), model.getVarByName(_var_name(prefix, k, f, t))
        return None, None

    assign_food = {}
    for rec in meta.get("assign_food_keys", []):
        key, v = _load_route_var("Bfood", rec)
        if key is not None and v is not None:
            assign_food[key] = v
    if not assign_food:
        assign_food = dict(assign)
    assign_manure = {}
    for rec in meta.get("assign_manure_keys", []):
        key, v = _load_route_var("Bmanure", rec)
        if key is not None and v is not None:
            assign_manure[key] = v
    assign_pw = {}
    for rec in meta.get("assign_pw_keys", []):
        key, v = _load_route_var("Bpw", rec)
        if key is not None and v is not None:
            assign_pw[key] = v

    scale_choice = {}
    for f, t, sc in meta.get("scale_choice_keys", []):
        v = model.getVarByName(_var_name("Yscale", f, t, sc))
        if v is not None:
            scale_choice[(f, t, int(sc))] = v
    dig_apply = {}
    for rec in meta.get("dig_apply_keys", []):
        if len(rec) == 5:
            f, k, crop, y = rec[0], rec[1], rec[2], rec[3]
            try:
                int(y)
            except Exception:
                f, k, crop, _old_cultivar, y = rec
        else:
            f, k, crop, y = rec
        v = model.getVarByName(_var_name("Xdig", f, k, crop, int(y)))
        if v is None and len(rec) == 5:
            try:
                f0, k0, c0, cultivar0, y0 = rec
                v = model.getVarByName(_var_name("Xdig", f0, k0, c0, cultivar0, y0))
            except Exception:
                v = None
        if v is not None:
            dig_apply[(f, k, crop, int(y))] = v
    syn_apply = {}
    for rec in meta.get("syn_apply_keys", []):
        if len(rec) == 5:
            key, crop, y, nut = rec[0], rec[1], rec[2], rec[3]
            try:
                int(y)
            except Exception:
                key, crop, _old_cultivar, y, nut = rec
        else:
            key, crop, y, nut = rec
        v = model.getVarByName(_var_name("Xsyn", key, crop, int(y), nut))
        if v is None and len(rec) == 5:
            try:
                key0, crop0, cultivar0, y0, nut0 = rec
                v = model.getVarByName(_var_name("Xsyn", key0, crop0, cultivar0, y0, nut0))
            except Exception:
                v = None
        if v is not None:
            syn_apply[(key, crop, int(y), nut)] = v
    compost_imp_apply = {}
    for rec in meta.get("compost_imp_apply_keys", []):
        if len(rec) == 4:
            key, crop, y = rec[0], rec[1], rec[2]
            try:
                int(y)
            except Exception:
                key, crop, _old_cultivar, y = rec
        else:
            key, crop, y = rec
        v = model.getVarByName(_var_name("XcompImp", key, crop, int(y)))
        if v is None and len(rec) == 4:
            try:
                key0, crop0, cultivar0, y0 = rec
                v = model.getVarByName(_var_name("XcompImp", key0, crop0, cultivar0, y0))
            except Exception:
                v = None
        if v is not None:
            compost_imp_apply[(key, crop, int(y))] = v
    compost_loc_apply = {}
    for rec in meta.get("compost_loc_apply_keys", []):
        if len(rec) == 4:
            key, crop, y = rec[0], rec[1], rec[2]
            try:
                int(y)
            except Exception:
                key, crop, _old_cultivar, y = rec
        else:
            key, crop, y = rec
        v = model.getVarByName(_var_name("XcompLoc", key, crop, int(y)))
        if v is None and len(rec) == 4:
            try:
                key0, crop0, cultivar0, y0 = rec
                v = model.getVarByName(_var_name("XcompLoc", key0, crop0, cultivar0, y0))
            except Exception:
                v = None
        if v is not None:
            compost_loc_apply[(key, crop, int(y))] = v
    nutrient_slack = {}
    for rec in meta.get("nutrient_slack_keys", []):
        try:
            key, crop, y, nut = rec
            v = model.getVarByName(_var_name(f"Sfert{nut}", key, crop, int(y)))
            if v is not None:
                nutrient_slack[(key, crop, int(y), nut)] = v
        except Exception:
            pass
    dig_unused = {}
    for f, y in meta.get("dig_unused_keys", []):
        v = model.getVarByName(_var_name("Udig", f, y))
        if v is not None:
            dig_unused[(f, int(y))] = v
    bio_use = {}
    for f, g in meta.get("bio_use_keys", []):
        v = model.getVarByName(_var_name("Ubio", f, g))
        if v is not None:
            bio_use[(f, g)] = v
    assign_energy = {}
    for rec in meta.get("assign_energy_keys", []):
        if len(rec) == 6:
            stream, k, f, t, g, y = rec
            v = model.getVarByName(_var_name("AE", stream, k, f, t, g, int(y)))
            if v is not None:
                assign_energy[(stream, k, f, t, g, int(y))] = v
        elif len(rec) == 5:
            k, f, t, g, y = rec
            v = model.getVarByName(_var_name("AE", k, f, t, g, int(y)))
            if v is not None:
                assign_energy[(k, f, t, g, int(y))] = v
        elif len(rec) == 4:
            k, f, t, g = rec
            v = model.getVarByName(_var_name("AE", k, f, t, g))
            if v is not None:
                assign_energy[(k, f, t, g)] = v
    waste_inc = {}
    for k, y in meta.get("waste_inc_keys", []):
        v = model.getVarByName(_var_name("Winc", k, int(y)))
        if v is not None:
            waste_inc[(k, int(y))] = v
    food_inc = {}
    for k, y in meta.get("food_inc_keys", []):
        v = model.getVarByName(_var_name("FWinc", k, int(y)))
        if v is not None:
            food_inc[(k, int(y))] = v
    event_food_inc_slack = {}
    for k, y in meta.get("event_food_inc_slack_keys", []):
        v = model.getVarByName(_var_name("SeventFWinc", k, int(y)))
        if v is not None:
            event_food_inc_slack[(k, int(y))] = v
    manure_compbase = {}
    for k, y in meta.get("manure_compbase_keys", []):
        v = model.getVarByName(_var_name("Mcompbase", k, int(y)))
        if v is not None:
            manure_compbase[(k, int(y))] = v
    comp_prod = {}
    for f, y in meta.get("comp_prod_keys", []):
        v = model.getVarByName(_var_name("COMPprod", f, int(y)))
        if v is not None:
            comp_prod[(f, int(y))] = v
    ad_input = {}
    for f, t, y in meta.get("ad_input_keys", []):
        v = model.getVarByName(_var_name("ADinput", f, t, int(y)))
        if v is not None:
            ad_input[(f, t, int(y))] = v
    ad_digest_prod = {}
    for f, y in meta.get("ad_digest_prod_keys", []):
        v = model.getVarByName(_var_name("DIGprod", f, int(y)))
        if v is not None:
            ad_digest_prod[(f, int(y))] = v
    ad_coeff = {}
    for raw_k, raw_v in meta.get("ad_coeff", {}).items():
        try:
            typ, sc = str(raw_k).split("|", 1)
            ad_coeff[(typ, int(sc))] = raw_v
        except Exception:
            pass
    aux = {"z": z, "s": s, "assign": assign, "assign_food": assign_food, "assign_manure": assign_manure, "assign_pw": assign_pw, "scale_choice": scale_choice,
           "dig_apply": dig_apply, "syn_apply": syn_apply,
           "compost_imp_apply": compost_imp_apply, "compost_loc_apply": compost_loc_apply,
           "nutrient_slack": nutrient_slack, "dig_unused": dig_unused, "bio_use": bio_use, "assign_energy": assign_energy,
           "waste_inc": waste_inc, "food_inc": food_inc, "event_food_inc_slack": event_food_inc_slack, "manure_compbase": manure_compbase, "comp_prod": comp_prod,
           "ad_input": ad_input, "ad_digest_prod": ad_digest_prod,
           "years": meta.get("years", []), "keys": meta.get("keys", []), "facs": meta.get("facs", []), "ad_coeff": ad_coeff}
    log(f"Loaded cached model variables: Z={len(z)}, S={len(s)}, A={len(assign)}, Xdig={len(dig_apply)}, Xsyn={len(syn_apply)}, Ubio={len(bio_use)}")
    return model, aux


def read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def to_num(x: Any, default: float = 0.0):
    out = pd.to_numeric(x, errors="coerce")
    if isinstance(out, pd.Series):
        return out.fillna(default)
    return default if pd.isna(out) else float(out)


def norm_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def ceil_scale_t_day(value: Any, default: int = 0) -> int:
    val = to_num(value, default)
    if not np.isfinite(val) or val <= 0:
        return int(default)
    return int(math.ceil(float(val)))


def event_from_years(rebuild_year: Any, renov_year: Any = np.nan) -> str:
    ry = to_num(rebuild_year, np.nan)
    ny = to_num(renov_year, np.nan)
    if np.isfinite(ry) and 2030 <= int(ry) <= 2050:
        return "REBUILD"
    if np.isfinite(ny) and 2030 <= int(ny) <= 2050:
        return "RENEW"
    return ""


def existing_tdad_fixed_scale(row: pd.Series | dict[str, Any]) -> int | None:
    base_type = norm_text(row.get("BASE_TYPE", ""))
    event = norm_text(row.get("EVENT", "")) or event_from_years(row.get("REBUILD_YEAR", np.nan), row.get("RENOV_YEAR", np.nan))
    if base_type == "TDAD" and event != "REBUILD":
        base_scale = to_num(row.get("BASE_CAPACITY_KG", 0.0), 0.0) / 365000.0
        return max(1, ceil_scale_t_day(base_scale, 1))
    return None


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def import_module_from_file(path: Path, module_name: str):
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_optional_modules(settings: OptimizeSettings):
    mwad_path = first_existing([
        settings.script_dir / "MWAD_module.py",
        settings.input_dir / "MWAD_module.py",
        settings.dataset_dir / "MWAD_module.py",
        Path.cwd() / "MWAD_module.py",
    ])
    tdad_path = first_existing([
        settings.script_dir / "TDAD_module.py",
        settings.input_dir / "TDAD_module.py",
        settings.dataset_dir / "TDAD_module.py",
        Path.cwd() / "TDAD_module.py",
    ])
    return (
        import_module_from_file(mwad_path, "mwad_module_opt") if mwad_path else None,
        import_module_from_file(tdad_path, "tdad_module_opt") if tdad_path else None,
    )


def build_price_table(cost_path: Path, years: list[int] = YEARS) -> pd.DataFrame:
    if not cost_path.exists():
        return pd.DataFrame({"YEAR": years})
    cost = pd.read_csv(cost_path)
    if "YEAR" not in cost.columns:
        return pd.DataFrame({"YEAR": years})
    out = pd.DataFrame({"YEAR": years})
    for c in cost.columns:
        if c == "YEAR":
            continue
        s = cost[["YEAR", c]].dropna()
        if s.empty:
            out[c] = 0.0
            continue
        x = to_num(s["YEAR"]).to_numpy(float)
        y = to_num(s[c]).to_numpy(float)
        if len(x) < 2:
            out[c] = float(y[-1])
        else:
            a, b = np.polyfit(x, y, 1)
            out[c] = a * np.asarray(years, dtype=float) + b
    # Inflation helpers consistent with 03_estimate_cost.py style.
    for col, base_year, name in [
        ("CHEM", 2024, "CHEM_INFL_2024"),
        ("PUBLIC", 2024, "PUBLIC_INFL_2024"),
        ("PUBLIC", 2020, "PUBLIC_INFL_2020"),
        ("PUBLIC", 2015, "PUBLIC_INFL_2015"),
    ]:
        if col in out.columns:
            base = float(out.loc[out["YEAR"].eq(base_year), col].iloc[0]) if (out["YEAR"] == base_year).any() else np.nan
            if not np.isfinite(base) or base == 0:
                base = 1.0
            out[name] = out[col] / base
        else:
            out[name] = 1.0
    return out


def ef_value(ef: pd.DataFrame, factor: str, col: str, default: float = 0.0) -> float:
    if ef.empty or "FACTOR" not in ef.columns or col not in ef.columns:
        return default
    sub = ef[ef["FACTOR"].astype(str).str.upper().eq(str(factor).upper())]
    if sub.empty:
        return default
    return float(to_num(sub.iloc[0][col], default))


def build_elec_ef(energy: pd.DataFrame, ef: pd.DataFrame, years: list[int] = YEARS) -> pd.DataFrame:
    if energy.empty or "YEAR" not in energy.columns:
        return pd.DataFrame({"YEAR": years, "ELEC_GWP": 0.0})
    e = energy.copy()
    e["YEAR"] = to_num(e["YEAR"]).astype(int)
    source_cols = ["COAL", "PETRO", "LNG", "NUC", "HYDRO", "SOLAR", "WIND", "BIO"]
    for c in source_cols:
        if c not in e.columns:
            e[c] = 0.0
        e[c] = to_num(e[c], 0.0)
    rows = []
    for year in years:
        if year in set(e["YEAR"]):
            row = e[e["YEAR"].eq(year)].iloc[-1]
        else:
            # nearest year fallback
            row = e.iloc[(e["YEAR"] - year).abs().argsort().iloc[0]]
        val = sum(float(row[c]) * ef_value(ef, "GWP", f"ELEC_{c}") for c in source_cols)
        rows.append({"YEAR": year, "ELEC_GWP": val})
    return pd.DataFrame(rows)


def capex_mwad(scale_t_day: float) -> float:
    if scale_t_day <= 0:
        return 0.0
    return 91.129 * (scale_t_day ** 0.7999) * 1e6


def capex_tdad(scale_t_day: float) -> float:
    if scale_t_day <= 0:
        return 0.0
    return 112.86 * (scale_t_day ** 0.7999) * 1e6


def process_cost_mwad(scale_t_day: float, public_infl_2015: float = 1.0) -> float:
    return 0.0 if scale_t_day <= 0 else 5.5676 * (scale_t_day ** 0.4674) * 1e6 * public_infl_2015


def process_cost_tdad(scale_t_day: float, public_infl_2015: float = 1.0) -> float:
    return 0.0 if scale_t_day <= 0 else 5.4353 * (scale_t_day ** 0.7232) * 1e6 * public_infl_2015


def process_cost_compost(input_kg_year: float, public_infl_2015: float = 1.0) -> float:
    t_day = max(input_kg_year * 0.001 / 365.0, 0.0)
    return 0.0 if t_day <= 0 else 43.748 * (t_day ** 0.0138) * 1e6 * public_infl_2015


def canonical_facility_name(name: Any) -> str:
    text = str(name).strip()
    return re.sub(r"_[0-9]+$", "", text)


def representative_msw_candidates(mswfac: pd.DataFrame) -> pd.DataFrame:
    if mswfac.empty:
        return mswfac
    wf = mswfac.copy()
    for col in ["FACILITY", "TYPE", "LAT", "LON", "CAPACITY"]:
        if col not in wf.columns:
            wf[col] = np.nan
    wf["FACILITY"] = wf["FACILITY"].astype(str)
    wf["TYPE"] = wf["TYPE"].astype(str).str.upper().str.strip()
    wf["LAT"] = to_num(wf["LAT"], np.nan)
    wf["LON"] = to_num(wf["LON"], np.nan)
    wf["_LAT_KEY"] = wf["LAT"].round(7)
    wf["_LON_KEY"] = wf["LON"].round(7)
    wf["_CANON"] = wf["FACILITY"].map(canonical_facility_name)
    rows = []
    for _, g in wf.groupby(["_LAT_KEY", "_LON_KEY", "TYPE"], dropna=False):
        first = g.iloc[0].copy()
        canon = canonical_facility_name(first.get("FACILITY", ""))
        first["FACILITY"] = canon
        first["CAPACITY"] = to_num(g.get("CAPACITY", pd.Series([0.0])), 0.0).sum()
        for year_col in ["CONST_YEAR", "RENOV_YEAR", "REBUILD_YEAR", "CAP_YEAR", "MWAD_YEAR", "YEAR"]:
            if year_col in g.columns:
                vals = pd.to_numeric(g[year_col], errors="coerce").dropna()
                if not vals.empty:
                    first[year_col] = vals.min()
        rows.append(first.drop(labels=["_LAT_KEY", "_LON_KEY", "_CANON"], errors="ignore"))
    return pd.DataFrame(rows).reset_index(drop=True)


def aggregate_param_same_coordinate_msw(out: pd.DataFrame) -> pd.DataFrame:
    if out.empty or "SOURCE" not in out.columns:
        return out
    df = out.copy()
    msw_mask = df["SOURCE"].astype(str).str.contains("MSWFAC", case=False, na=False)
    if not msw_mask.any():
        return df
    non_msw = df.loc[~msw_mask].copy()
    msw = df.loc[msw_mask].copy()
    msw["_LAT_KEY"] = to_num(msw["LAT"], np.nan).round(7)
    msw["_LON_KEY"] = to_num(msw["LON"], np.nan).round(7)
    rows = []
    for _, g in msw.groupby(["_LAT_KEY", "_LON_KEY", "BASE_TYPE"], dropna=False):
        r = g.iloc[0].copy()
        r["FACILITY"] = canonical_facility_name(r["FACILITY"])
        r["SOURCE"] = "+".join(sorted(set(map(str, g["SOURCE"]))))
        r["BASE_CAPACITY_KG"] = to_num(g.get("BASE_CAPACITY_KG", pd.Series([0.0])), 0.0).sum()
        r["BUILT_YEAR_BASE"] = to_num(g.get("BUILT_YEAR_BASE", pd.Series([2030])), 2030).min()
        for col in ["IS_INC_SITE", "ALLOW_COMPOST", "ALLOW_MWAD", "ALLOW_TDAD", "ALLOW_INC", "ALLOW_RDF"]:
            if col in g.columns:
                r[col] = to_num(g[col], 0.0).max()
        rows.append(r.drop(labels=["_LAT_KEY", "_LON_KEY"], errors="ignore"))
    return pd.concat([non_msw, pd.DataFrame(rows)], ignore_index=True).reset_index(drop=True)


def load_param_facilities(settings: OptimizeSettings) -> pd.DataFrame:
    path = settings.input_dir / "05_param_facility.csv"
    if not path.exists():
        path = settings.dataset_dir / "05_param_facility.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or "FACILITY" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    for col in ["SOURCE", "LAT", "LON", "CITY", "BASE_TYPE", "BASE_CAPACITY_KG", "IS_INC_SITE",
                "ALLOW_COMPOST", "ALLOW_MWAD", "ALLOW_TDAD", "ALLOW_INC", "ALLOW_RDF",
                "REBUILD_YEAR", "RENOV_YEAR", "EVENT"]:
        if col not in out.columns:
            out[col] = 0 if col.startswith("ALLOW_") or col == "IS_INC_SITE" else ("" if col == "EVENT" else np.nan)
    if "BUILT_YEAR_BASE" not in out.columns:
        out["BUILT_YEAR_BASE"] = to_num(out.get("OPEN_YEAR", 2030), 2030).astype(int) if isinstance(out.get("OPEN_YEAR", 2030), pd.Series) else 2030
    out["FACILITY"] = out["FACILITY"].astype(str)
    out["BASE_TYPE"] = out["BASE_TYPE"].astype(str).str.upper().str.strip()
    out = out[~out["BASE_TYPE"].eq("LANDFILL")].copy()
    out["EVENT"] = out.apply(lambda r: norm_text(r.get("EVENT", "")) or event_from_years(r.get("REBUILD_YEAR", np.nan), r.get("RENOV_YEAR", np.nan)), axis=1)
    out = aggregate_param_same_coordinate_msw(out)
    return out[["FACILITY", "SOURCE", "LAT", "LON", "CITY", "BASE_TYPE", "IS_INC_SITE",
                "ALLOW_COMPOST", "ALLOW_MWAD", "ALLOW_TDAD", "ALLOW_INC", "ALLOW_RDF",
                "BASE_CAPACITY_KG", "BUILT_YEAR_BASE", "REBUILD_YEAR", "RENOV_YEAR", "EVENT"]]



MWAD_YEAR_BOUNDARY = {
    "KYOTO_NORTH": {101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111},
    "KYOTO_NE": {101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111},
    "KYOTO_SOUTH": {101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111},
    "FUKUCHIYAMA": {201}, "MAIZURU": {202}, "AYABE": {203},
    "SAKURAZUKA": {206, 213, 407}, "KYOTANABE": {211}, "MINEYAMA": {212},
    "ORII": {204, 207, 210, 322, 343, 344, 364, 365, 367},
    "HASEYAMA": {204, 207, 210, 322, 343, 344, 364, 365, 367},
    "KIZUGAWA": {214, 366}, "OTOKUNI": {208, 209, 303},
    "MIYAZUYOSA": {205, 463, 465},
}



def mwad_base_name_for_year_lookup(name: Any) -> str:
    text = str(name).strip()
    parts = text.split("_")
    while parts and (parts[-1].isdigit() or parts[-1] in {"R", "ALT", "OPT", "INC", "RDF", "TDAD", "MWAD", "COMPOST", "AD"}):
        parts = parts[:-1]
    return "_".join(parts)

def build_mwad_year_lookup_from_mswfac(msw: pd.DataFrame) -> tuple[dict[str, int], dict[int, int], dict[int, int]]:
    if msw.empty or "MWAD_YEAR" not in msw.columns:
        return {}, {}, {}
    df = msw.copy()
    df["BASE"] = df.get("FACILITY", "").astype(str).map(mwad_base_name_for_year_lookup)
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
    for base, cities in MWAD_YEAR_BOUNDARY.items():
        year = base_map.get(base)
        if year is None:
            continue
        for city in cities:
            city_i = int(city)
            boundary_map[city_i] = min(boundary_map.get(city_i, int(year)), int(year))
    return {str(k): int(v) for k, v in base_map.items()}, {int(k): int(v) for k, v in city_map.items()}, boundary_map


def mwad_year_for_facility_row(row: pd.Series | dict[str, Any], base_map: dict[str, int], city_map: dict[int, int], boundary_map: dict[int, int]) -> int | None:
    base = mwad_base_name_for_year_lookup(row.get("FACILITY", ""))
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


def event_year_from_candidate_row(row: pd.Series | dict[str, Any]) -> int | None:
    years: list[int] = []
    for col in ["REBUILD_YEAR", "RENOV_YEAR"]:
        val = to_num(row.get(col, np.nan), np.nan)
        if np.isfinite(val) and 2030 <= int(val) <= 2050:
            years.append(int(val))
    return min(years) if years else None


def build_event_city_year_lookup(candidates: pd.DataFrame) -> dict[int, int]:
    out: dict[int, int] = {}
    if candidates.empty:
        return out
    for _, row in candidates.iterrows():
        ey = event_year_from_candidate_row(row)
        if ey is None:
            continue
        base = mwad_base_name_for_year_lookup(row.get("FACILITY", ""))
        cities = MWAD_YEAR_BOUNDARY.get(str(base), None)
        if not cities:
            city_val = to_num(row.get("CITY", np.nan), np.nan)
            cities = {int(city_val)} if np.isfinite(city_val) else set()
        for city in cities:
            city_i = int(city)
            out[city_i] = min(out.get(city_i, int(ey)), int(ey))
    return out

def ad_gwp_per_input_kg(settings: OptimizeSettings, typ: str, ef: pd.DataFrame,
                        elec_gwp: float, ad_lookup: pd.DataFrame, scale: int | None = None,
                        include_heat: bool = True) -> float:
    if ad_lookup.empty:
        return 0.0
    sub = ad_lookup[ad_lookup["TYPE"].astype(str).str.upper().eq(str(typ).upper())].copy()
    if sub.empty:
        return 0.0
    if scale is not None and "SCALE_T_DAY" in sub.columns and sub["SCALE_T_DAY"].eq(scale).any():
        r = sub[sub["SCALE_T_DAY"].eq(scale)].iloc[0]
    else:
        r = sub.sort_values("SCALE_T_DAY").iloc[-1]
    cap = float(to_num(r.get("CAPACITY_KG_YEAR", 0.0), 0.0))
    if cap <= 0:
        return 0.0
    typ_u = str(typ).upper()
    combust_ef = ef_value(ef, "GWP", "TDAD_COMBUST" if typ_u == "TDAD" else "MWAD_COMBUST", 0.0)
    digww_ef = ef_value(ef, "GWP", "TDAD_DIGWW" if typ_u == "TDAD" else "MWAD_DIGWW", 0.0)
    water_ef = 0.0 if typ_u == "TDAD" else ef_value(ef, "GWP", "MWAD_WATER", 0.0)
    heat_ef = ef_value(ef, "GWP", "HEAT", 0.0)
    gridgas_ef = ef_value(ef, "GWP", "GRID_GAS", 0.0)
    val = 0.0
    val += float(to_num(r.get("BIOGAS_NM3", 0.0), 0.0)) * BIOGAS_HV_MJ_PER_NM3 * combust_ef
    val += float(to_num(r.get("ELEC_KWH", 0.0), 0.0)) * elec_gwp
    if include_heat:
        val += float(to_num(r.get("HEAT_MJ", 0.0), 0.0)) * heat_ef
    val += float(to_num(r.get("WATER_M3", 0.0), 0.0)) * water_ef
    if typ_u == "TDAD":
        val += float(to_num(r.get("DIGEST_KG", 0.0), 0.0)) * 0.001 * digww_ef
    if settings.enable_gridgas:
        val -= float(to_num(r.get("BIOGAS_NM3", 0.0), 0.0)) * gridgas_ef
    return val / cap


def build_candidate_facilities(settings: OptimizeSettings) -> pd.DataFrame:
    param_fac = load_param_facilities(settings)
    if not param_fac.empty:
        return param_fac

    manurefac = read_csv_if_exists(settings.dataset_dir / "00_manurefac.csv")
    mswfac = representative_msw_candidates(read_csv_if_exists(settings.dataset_dir / "00_mswfac.csv"))

    rows: list[dict[str, Any]] = []

    if not manurefac.empty:
        mf = manurefac.copy()
        for col in ["FACILITY", "TYPE", "LAT", "LON", "CITY", "CAPACITY"]:
            if col not in mf.columns:
                mf[col] = np.nan
        mf["FACILITY"] = mf["FACILITY"].astype(str)
        for fac, g in mf.groupby("FACILITY", dropna=False):
            base_type = "COMPOST" if (g["TYPE"].astype(str).str.upper() == "COMPOST").any() else norm_text(g["TYPE"].iloc[0])
            rebuild_year = np.nan
            renov_year = np.nan
            rows.append({
                "FACILITY": str(fac),
                "SOURCE": "MANUREFAC",
                "LAT": float(to_num(g["LAT"].dropna().iloc[0], np.nan)) if g["LAT"].notna().any() else np.nan,
                "LON": float(to_num(g["LON"].dropna().iloc[0], np.nan)) if g["LON"].notna().any() else np.nan,
                "CITY": int(to_num(g["CITY"].dropna().iloc[0], -1)) if g["CITY"].notna().any() else -1,
                "BASE_TYPE": base_type or "COMPOST",
                "IS_INC_SITE": 0,
                "ALLOW_COMPOST": 1,
                "ALLOW_MWAD": 1,
                "ALLOW_TDAD": 1,
                "ALLOW_INC": 0,
                "ALLOW_RDF": 0,
                "BASE_CAPACITY_KG": float(to_num(g.get("CAPACITY", pd.Series([0.0])).sum(), 0.0)),
                "BUILT_YEAR_BASE": int(to_num(g.get("CONST_YEAR", g.get("YEAR", pd.Series([2030]))).min(), 2030)) if ("CONST_YEAR" in g.columns or "YEAR" in g.columns) else 2030,
                "REBUILD_YEAR": rebuild_year,
                "RENOV_YEAR": renov_year,
                "EVENT": event_from_years(rebuild_year, renov_year),
            })

    if not mswfac.empty:
        wf = mswfac.copy()
        for col in ["FACILITY", "TYPE", "LAT", "LON", "CITY", "CAPACITY"]:
            if col not in wf.columns:
                wf[col] = np.nan
        wf["FACILITY"] = wf["FACILITY"].astype(str)
        wf["TYPE"] = wf["TYPE"].astype(str).str.upper().str.strip()
        for fac, g in wf.groupby("FACILITY", dropna=False):
            base_type = norm_text(g["TYPE"].iloc[0]) or "INC"
            is_landfill = int((g["TYPE"] == "LANDFILL").any())
            is_inc = int((g["TYPE"] == "INC").any())
            rebuild_vals = pd.to_numeric(g.get("REBUILD_YEAR", pd.Series(dtype=float)), errors="coerce").dropna() if "REBUILD_YEAR" in g.columns else pd.Series(dtype=float)
            renov_vals = pd.to_numeric(g.get("RENOV_YEAR", pd.Series(dtype=float)), errors="coerce").dropna() if "RENOV_YEAR" in g.columns else pd.Series(dtype=float)
            rebuild_year = float(rebuild_vals.min()) if not rebuild_vals.empty else np.nan
            renov_year = float(renov_vals.min()) if not renov_vals.empty else np.nan
            rows.append({
                "FACILITY": str(fac),
                "SOURCE": "MSWFAC",
                "LAT": float(to_num(g["LAT"].dropna().iloc[0], np.nan)) if g["LAT"].notna().any() else np.nan,
                "LON": float(to_num(g["LON"].dropna().iloc[0], np.nan)) if g["LON"].notna().any() else np.nan,
                "CITY": int(to_num(g["CITY"].dropna().iloc[0], -1)) if g["CITY"].notna().any() else -1,
                "BASE_TYPE": base_type,
                "IS_INC_SITE": is_inc,
                "ALLOW_COMPOST": 0,
                "ALLOW_MWAD": 0 if is_landfill else 1,
                "ALLOW_TDAD": 0 if is_landfill else 1,
                "ALLOW_INC": 1 if base_type == "INC" else 0,
                "ALLOW_RDF": 1 if base_type == "RDF" else 0,
                "BASE_CAPACITY_KG": float(to_num(g.get("CAPACITY", pd.Series([0.0])).sum(), 0.0)),
                "BUILT_YEAR_BASE": int(to_num(g.get("CONST_YEAR", g.get("YEAR", pd.Series([2030]))).min(), 2030)) if ("CONST_YEAR" in g.columns or "YEAR" in g.columns) else 2030,
                "REBUILD_YEAR": rebuild_year,
                "RENOV_YEAR": renov_year,
                "EVENT": event_from_years(rebuild_year, renov_year),
            })

    cand = pd.DataFrame(rows)
    if cand.empty:
        raise FileNotFoundError("No candidate facilities found. Expected 00_manurefac.csv and/or 00_mswfac.csv in dataset-dir.")

    agg = {
        "SOURCE": lambda s: "+".join(sorted(set(map(str, s)))),
        "LAT": "first", "LON": "first", "CITY": "first",
        "BASE_TYPE": "first", "IS_INC_SITE": "max",
        "ALLOW_COMPOST": "max", "ALLOW_MWAD": "max", "ALLOW_TDAD": "max",
        "ALLOW_INC": "max", "ALLOW_RDF": "max", "BASE_CAPACITY_KG": "sum",
        "BUILT_YEAR_BASE": "min",
        "REBUILD_YEAR": "min", "RENOV_YEAR": "min", "EVENT": "first",
    }
    for c in ["REBUILD_YEAR", "RENOV_YEAR", "EVENT"]:
        if c not in cand.columns:
            cand[c] = "" if c == "EVENT" else np.nan
    cand = cand.groupby("FACILITY", as_index=False).agg(agg)
    cand["BASE_TYPE"] = np.where(cand["ALLOW_COMPOST"].eq(1), "COMPOST", cand["BASE_TYPE"])
    cand["EVENT"] = cand.apply(lambda r: norm_text(r.get("EVENT", "")) or event_from_years(r.get("REBUILD_YEAR", np.nan), r.get("RENOV_YEAR", np.nan)), axis=1)
    return cand


def load_waste_activity(settings: OptimizeSettings) -> pd.DataFrame:
    waste = read_csv_if_exists(settings.input_dir / "02_inventory_waste.csv")
    if waste.empty:
        waste = read_csv_if_exists(settings.dataset_dir / "02_inventory_waste.csv")
    if waste.empty:
        return pd.DataFrame()
    for c in ["FACILITY", "YEAR", "TYPE"]:
        if c not in waste.columns:
            waste[c] = ""
    waste["FACILITY"] = waste["FACILITY"].astype(str)
    waste["YEAR"] = to_num(waste["YEAR"], 0).astype(int)
    waste["TYPE"] = waste["TYPE"].astype(str).str.upper().str.strip()
    mass_cols = [
        "WASTE_KG", "MANURE_KG", "FOOD_KG", "PAPER_KG", "WOOD_KG",
        "PLASTIC_KG", "SYNTEX_KG", "NATTEX_KG", "INC_INPUT_KG",
        "ASH_KG", "TDAD_REFUSE_KG", "MWAD_REFUSE_KG", "TDAD_DIGEST_KG", "MWAD_DIGEST_KG",
        "DIGEST_UNUSED_KG", "TDAD_DIGSOLID_KG", "MWAD_DIGSOLID_KG",
        "INC_CAPEX", "TDAD_CAPEX", "RDF_CAPEX", "MWAD_CAPEX",
    ]
    for c in mass_cols:
        if c not in waste.columns:
            waste[c] = 0.0
        waste[c] = to_num(waste[c], 0.0)
    return waste


def facility_baseline_loads(candidates: pd.DataFrame, waste: pd.DataFrame) -> pd.DataFrame:
    years = pd.DataFrame({"YEAR": YEARS})
    base = candidates[["FACILITY"]].assign(_k=1).merge(years.assign(_k=1), on="_k").drop(columns="_k")
    if waste.empty:
        for c in ["INPUT_KG", "FOOD_KG", "PAPER_KG", "WOOD_KG", "MANURE_KG", "INC_INPUT_KG"]:
            base[c] = 0.0
        return base
    w = waste.copy()
    w["INPUT_KG"] = w[["WASTE_KG", "MANURE_KG"]].sum(axis=1)
    grouped = w.groupby(["FACILITY", "YEAR"], as_index=False)[["INPUT_KG", "FOOD_KG", "PAPER_KG", "WOOD_KG", "MANURE_KG", "INC_INPUT_KG"]].sum()
    out = base.merge(grouped, on=["FACILITY", "YEAR"], how="left").fillna(0.0)
    return out


def representative_tdad_feedstock_mix(settings: OptimizeSettings, capacity_kg_y: float) -> tuple[dict[str, float], dict[str, float]]:
    path = settings.input_dir / "05_param_feedstock.csv"
    if not path.exists():
        path = settings.dataset_dir / "05_param_feedstock.csv"
    if not path.exists():
        return {"FOOD_WASTE": capacity_kg_y}, {"FOOD_SHARE": 1.0, "PAPER_SHARE": 0.0, "WOOD_SHARE": 0.0, "MANURE_SHARE": 0.0}
    try:
        feed = pd.read_csv(path, usecols=lambda c: c in {
            "FOOD_KG", "PAPER_KG", "WOOD_KG", "MANURE_DAIRY_KG", "MANURE_CATTLE_KG",
            "MANURE_SWINE_KG", "MANURE_CHICKEN_KG", "MANURE_BROILER_KG"
        })
    except Exception:
        return {"FOOD_WASTE": capacity_kg_y}, {"FOOD_SHARE": 1.0, "PAPER_SHARE": 0.0, "WOOD_SHARE": 0.0, "MANURE_SHARE": 0.0}
    food = float(to_num(feed.get("FOOD_KG", pd.Series([0.0])), 0.0).sum())
    paper = float(to_num(feed.get("PAPER_KG", pd.Series([0.0])), 0.0).sum())
    wood = float(to_num(feed.get("WOOD_KG", pd.Series([0.0])), 0.0).sum())
    manure = 0.0
    for c in ["MANURE_DAIRY_KG", "MANURE_CATTLE_KG", "MANURE_SWINE_KG", "MANURE_CHICKEN_KG", "MANURE_BROILER_KG"]:
        if c in feed.columns:
            manure += float(to_num(feed[c], 0.0).sum())
    total = food + paper + wood + manure
    if total <= 0:
        return {"FOOD_WASTE": capacity_kg_y}, {"FOOD_SHARE": 1.0, "PAPER_SHARE": 0.0, "WOOD_SHARE": 0.0, "MANURE_SHARE": 0.0}
    shares = {
        "FOOD_SHARE": food / total,
        "PAPER_SHARE": paper / total,
        "WOOD_SHARE": wood / total,
        "MANURE_SHARE": manure / total,
    }
    mix = {
        "FOOD_WASTE": capacity_kg_y * shares["FOOD_SHARE"],
        "PAPER_WASTE": capacity_kg_y * shares["PAPER_SHARE"],
        "WOOD_WASTE": capacity_kg_y * shares["WOOD_SHARE"],
        "CATTLE_F": capacity_kg_y * shares["MANURE_SHARE"],
    }
    mix = {k: v for k, v in mix.items() if v > 0}
    return mix, shares


def enumerate_ad_feedstock_mixes(settings: OptimizeSettings, capacity_kg_y: float) -> list[tuple[str, dict[str, float], dict[str, float]]]:
    components = ["FOOD", "PAPER", "WOOD", "MANURE"]
    rows: list[tuple[str, dict[str, float], dict[str, float]]] = []
    for mask in range(1, 1 << len(components)):
        active = [c for i, c in enumerate(components) if mask & (1 << i)]
        share_each = 1.0 / len(active)
        shares = {
            "FOOD_SHARE": share_each if "FOOD" in active else 0.0,
            "PAPER_SHARE": share_each if "PAPER" in active else 0.0,
            "WOOD_SHARE": share_each if "WOOD" in active else 0.0,
            "MANURE_SHARE": share_each if "MANURE" in active else 0.0,
        }
        mix: dict[str, float] = {}
        if shares["FOOD_SHARE"] > 0:
            mix["FOOD_WASTE"] = capacity_kg_y * shares["FOOD_SHARE"]
        if shares["PAPER_SHARE"] > 0:
            mix["PAPER_WASTE"] = capacity_kg_y * shares["PAPER_SHARE"]
        if shares["WOOD_SHARE"] > 0:
            mix["WOOD_WASTE"] = capacity_kg_y * shares["WOOD_SHARE"]
        if shares["MANURE_SHARE"] > 0:
            mix["CATTLE_F"] = capacity_kg_y * shares["MANURE_SHARE"]
        rows.append(("_".join(active) + "_MIX", mix, shares))
    avg_mix, avg_shares = representative_tdad_feedstock_mix(settings, capacity_kg_y)
    if any(avg_shares.get(k, 0.0) > 1e-9 for k in ["PAPER_SHARE", "WOOD_SHARE", "MANURE_SHARE"]):
        rows.append(("AVG_FLOW_MIX", avg_mix, avg_shares))
    return rows

def _ad_lookup_row(typ: str, mix_id: str, scale: int, capacity_kg_y: float, r: dict[str, Any], shares: dict[str, float]) -> dict[str, Any]:
    prefix = typ.upper()
    capex_default = capex_mwad(scale) if prefix == "MWAD" else capex_tdad(scale)
    return {
        "TYPE": prefix, "MIX_ID": mix_id, "SCALE_T_DAY": scale,
        "FOOD_SHARE": float(shares.get("FOOD_SHARE", 0.0)),
        "PAPER_SHARE": float(shares.get("PAPER_SHARE", 0.0)),
        "WOOD_SHARE": float(shares.get("WOOD_SHARE", 0.0)),
        "MANURE_SHARE": float(shares.get("MANURE_SHARE", 0.0)),
        "CAPACITY_KG_YEAR": capacity_kg_y,
        "CAPEX": float(r.get(f"{prefix}_CAPEX", capex_default)),
        "ELEC_KWH": float(r.get(f"{prefix}_ELEC_REQ", 0.0)),
        "HEAT_MJ": float(r.get(f"{prefix}_HEAT_REQ", 0.0)),
        "WATER_M3": float(r.get(f"{prefix}_WATER_REQ", 0.0)),
        "BIOGAS_NM3": float(r.get(f"{prefix}_BIOGAS", 0.0)),
        "DIGEST_KG": float(r.get(f"{prefix}_DIGESTATE", 0.0)),
        "DIGEST_C_KG": float(r.get(f"{prefix}_DIG_ORGC", 0.0)),
        "DIGEST_N_KG": float(r.get(f"{prefix}_DIG_N", 0.0)),
        "DIGEST_P2O5_KG": float(r.get(f"{prefix}_DIG_P2O5", 0.0)),
        "DIGEST_K2O_KG": float(r.get(f"{prefix}_DIG_K2O", 0.0)),
        "REFUSE_KG": float(r.get(f"{prefix}_REFUSED", 0.0)),
    }


def estimate_ad_lookup(settings: OptimizeSettings, candidates: pd.DataFrame) -> pd.DataFrame:
    mwad_module, tdad_module = load_optional_modules(settings)
    rows = []
    for scale in range(settings.min_scale, settings.max_scale + 1, settings.scale_step):
        capacity_kg_y = scale * 1000.0 * 365.0
        for mix_id, mix, shares in enumerate_ad_feedstock_mixes(settings, capacity_kg_y):
            if mwad_module is not None and hasattr(mwad_module, "mwad_run"):
                try:
                    r = mwad_module.mwad_run(capacity_kg_y, capacity_kg_y, Biogas_griduse=settings.enable_gridgas)
                except Exception:
                    r = {}
            else:
                r = {"MWAD_CAPEX": capex_mwad(scale)}
            rows.append(_ad_lookup_row("MWAD", mix_id, scale, capacity_kg_y, r, shares))

            if tdad_module is not None and hasattr(tdad_module, "tdad_run"):
                try:
                    r = tdad_module.tdad_run(mix, capacity_amount=capacity_kg_y, Biogas_griduse=settings.enable_gridgas)
                except Exception:
                    r = {}
            else:
                r = {"TDAD_CAPEX": capex_tdad(scale)}
            rows.append(_ad_lookup_row("TDAD", mix_id, scale, capacity_kg_y, r, shares))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(["TYPE", "MIX_ID", "SCALE_T_DAY"], keep="first")
    return out

def option_score(
    settings: OptimizeSettings,
    fac_row: pd.Series,
    opt_type: str,
    scale: int,
    annual_input_kg: float,
    price_y: pd.DataFrame,
    ef: pd.DataFrame,
    elec_ef: pd.DataFrame,
    ad_lookup: pd.DataFrame,
) -> float:

    target = settings.target.upper()
    public_2015 = float(price_y.get("PUBLIC_INFL_2015", pd.Series([1.0])).mean()) if not price_y.empty else 1.0
    is_inc_site = int(fac_row.get("IS_INC_SITE", 0)) == 1

    if target == "COST":
        if opt_type == "COMPOST":
            return process_cost_compost(annual_input_kg, public_2015)
        if opt_type == "MWAD":
            return process_cost_mwad(scale, public_2015) + capex_mwad(scale) / 21.0
        if opt_type == "TDAD":
            return process_cost_tdad(scale, public_2015) + capex_tdad(scale) / 21.0
        return 0.0

    build_ef = ef_value(ef, "GWP", "WASTE_REBUILD", 0.0)
    heat_ef = ef_value(ef, "GWP", "HEAT", 0.0)
    water_ef = ef_value(ef, "GWP", "MWAD_WATER", 0.0)
    gridgas_ef = ef_value(ef, "GWP", "GRID_GAS", 0.0)
    elec_gwp = float(to_num(elec_ef.get("ELEC_GWP", pd.Series([0.0])), 0.0).mean()) if "ELEC_GWP" in elec_ef.columns else 0.0

    if opt_type == "COMPOST":
        return 0.0

    lookup = ad_lookup[(ad_lookup["TYPE"].eq(opt_type)) & (ad_lookup["SCALE_T_DAY"].eq(scale))]
    if lookup.empty:
        cap = capex_mwad(scale) if opt_type == "MWAD" else capex_tdad(scale)
        return cap * build_ef / 21.0
    r = lookup.iloc[0]
    capex = float(r["CAPEX"])
    val = capex * build_ef / 21.0
    val += float(r.get("ELEC_KWH", 0.0)) * elec_gwp
    if not is_inc_site:
        val += float(r.get("HEAT_MJ", 0.0)) * heat_ef
    val += float(r.get("WATER_M3", 0.0)) * water_ef
    if settings.enable_gridgas:
        val -= float(r.get("BIOGAS_NM3", 0.0)) * gridgas_ef
    return val


def build_facility_choice_model(
    settings: OptimizeSettings,
    candidates: pd.DataFrame,
    loads: pd.DataFrame,
    price_y: pd.DataFrame,
    ef: pd.DataFrame,
    elec_ef: pd.DataFrame,
    ad_lookup: pd.DataFrame,
):

    if gp is None:
        return None, {}, {}

    m = gp.Model("CE_AD_facility_choice")
    configure_gurobi_model(m, settings, "CE_AD_facility_choice")

    cand = candidates.set_index("FACILITY")
    max_input = loads.groupby("FACILITY")["INPUT_KG"].max().to_dict() if not loads.empty else {}

    option_vars: dict[tuple[str, str, int], Any] = {}
    obj_terms = []
    for fac, row in cand.iterrows():
        locked_tdad_scale = existing_tdad_fixed_scale(row)
        allowed_options: list[tuple[str, int]] = []
        if locked_tdad_scale is not None:
            allowed_options = [("TDAD", locked_tdad_scale)]
        else:
            if compost_candidates_enabled(settings) and int(row.get("ALLOW_COMPOST", 0)) == 1:
                allowed_options.append(("COMPOST", 0))
            if int(row.get("ALLOW_MWAD", 0)) == 1:
                for s in range(settings.min_scale, settings.max_scale + 1, settings.scale_step):
                    allowed_options.append(("MWAD", s))
            if int(row.get("ALLOW_TDAD", 0)) == 1:
                for s in range(settings.min_scale, settings.max_scale + 1, settings.scale_step):
                    allowed_options.append(("TDAD", s))
        if locked_tdad_scale is None and int(row.get("ALLOW_INC", 0)) == 1:
            allowed_options.append(("INC", 0))
        if locked_tdad_scale is None and int(row.get("ALLOW_RDF", 0)) == 1:
            allowed_options.append(("RDF", 0))
        if not allowed_options:
            continue

        vars_for_fac = []
        for typ, scale in allowed_options:
            v = m.addVar(vtype=GRB.BINARY, name=f"select[{fac},{typ},{scale}]")
            option_vars[(fac, typ, scale)] = v
            vars_for_fac.append(v)
            annual_input = float(max_input.get(fac, 0.0))
            score = option_score(settings, row, typ, scale, annual_input, price_y, ef, elec_ef, ad_lookup)
            obj_terms.append(score * v)

        m.addConstr(gp.quicksum(vars_for_fac) == 1, name=f"one_option[{fac}]")


        load_kg = float(max_input.get(fac, 0.0))

        if load_kg > 0 and locked_tdad_scale is None:
            ad_capacity_expr = gp.quicksum(
                (scale * 1000.0 * 365.0) * option_vars[(fac, typ, scale)]
                for (f0, typ, scale) in option_vars
                if f0 == fac and typ in {"MWAD", "TDAD"}
            )
            ad_selected_expr = gp.quicksum(
                option_vars[(fac, typ, scale)]
                for (f0, typ, scale) in option_vars
                if f0 == fac and typ in {"MWAD", "TDAD"}
            )
            # Only bind capacity when AD is selected.
            m.addConstr(load_kg * ad_selected_expr <= ad_capacity_expr, name=f"ad_capacity_cover[{fac}]")

    m.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)
    return m, option_vars, {"max_input": max_input}



def load_flow_parameter_tables(settings: OptimizeSettings) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    names = ["feedstock", "facility", "distance"]
    if settings.mode in {"facility_agriculture", "facility_only_dig"}:
        names.append("crop_req")
    for name in names:
        path = settings.input_dir / f"05_param_{name}.csv"
        if not path.exists():
            path = settings.dataset_dir / f"05_param_{name}.csv"
        tables[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not tables.get("distance", pd.DataFrame()).empty and "FACILITY" in tables["distance"].columns:
        d = tables["distance"].copy()
        d["FACILITY"] = d["FACILITY"].map(canonical_facility_name)
        if "DIST_KM" in d.columns:
            d["DIST_KM"] = to_num(d["DIST_KM"], 0.0)
            d = d.groupby(["KEY", "FACILITY"], as_index=False)["DIST_KM"].min()
        tables["distance"] = d
    if not tables.get("facility", pd.DataFrame()).empty and "FACILITY" in tables["facility"].columns:
        tables["facility"]["FACILITY"] = tables["facility"]["FACILITY"].map(canonical_facility_name)
    if settings.mode not in {"facility_agriculture", "facility_only_dig"}:
        tables["crop_req"] = pd.DataFrame()
    tables = prune_optimization_distance_candidates(tables, settings)
    return tables


def prune_optimization_distance_candidates(tables: dict[str, pd.DataFrame], settings: OptimizeSettings) -> dict[str, pd.DataFrame]:

    limit = int(getattr(settings, "max_digestate_facilities_per_key", 0) or 0)
    dist = tables.get("distance", pd.DataFrame())
    if limit <= 0 or dist.empty or not {"KEY", "FACILITY", "DIST_KM"}.issubset(dist.columns):
        return tables
    d = dist.copy()
    d["DIST_KM"] = to_num(d["DIST_KM"], 0.0)
    before = len(d)
    d = (
        d.sort_values(["KEY", "DIST_KM", "FACILITY"])
         .groupby("KEY", as_index=False, group_keys=False)
         .head(limit)
         .reset_index(drop=True)
    )
    tables["distance"] = d
    if before != len(d):
        log(f"Pruned optimization distance candidates: {before:,} -> {len(d):,} rows; nearest={limit} per KEY")
    return tables


def has_flow_parameters(tables: dict[str, pd.DataFrame]) -> bool:
    return not tables.get("feedstock", pd.DataFrame()).empty and not tables.get("distance", pd.DataFrame()).empty


def _series_sum_by_year(feed: pd.DataFrame, col: str) -> dict[tuple[str, int], float]:
    if col not in feed.columns:
        feed[col] = 0.0
    return {(str(r["KEY"]), int(r["YEAR"])): float(r[col]) for _, r in feed[["KEY", "YEAR", col]].iterrows()}



def ad_lookup_coefficients(ad_lookup: pd.DataFrame) -> dict[tuple[str, int], dict[str, float]]:

    coeff: dict[tuple[str, int], dict[str, float]] = {}
    if ad_lookup.empty:
        return coeff
    df = ad_lookup.copy()
    df["TYPE"] = df.get("TYPE", "").astype(str).str.upper().str.strip()
    df["SCALE_T_DAY"] = to_num(df.get("SCALE_T_DAY", 0), 0).astype(int)
    coeff_cols = [
        "CAPACITY_KG_YEAR", "CAPEX", "ELEC_KWH", "HEAT_MJ", "WATER_M3",
        "BIOGAS_NM3", "DIGEST_KG", "DIGEST_N_KG", "DIGEST_P2O5_KG",
        "DIGEST_K2O_KG", "DIGEST_C_KG", "REFUSE_KG",
    ]
    for col in coeff_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = to_num(df[col], 0.0)
    for (typ, scale), g in df.groupby(["TYPE", "SCALE_T_DAY"], dropna=False):
        coeff[(str(typ), int(scale))] = {col: float(g[col].mean()) for col in coeff_cols}
    return coeff


def crop_req_records_for_agriculture(settings: OptimizeSettings, crop_req: pd.DataFrame, years: list[int]) -> pd.DataFrame:

    if settings.mode != "facility_agriculture" or crop_req.empty:
        return pd.DataFrame()
    cr = crop_req.copy()
    for col in ["KEY", "CROP", "CULTIVAR", "ORG_FLAG"]:
        if col not in cr.columns:
            cr[col] = ""
    cr["KEY"] = cr["KEY"].astype(str)
    cr["CROP"] = cr["CROP"].astype(str).str.upper().str.strip()
    cr["CULTIVAR"] = cr["CULTIVAR"].astype(str).str.lower().str.strip()
    cr["ORG_FLAG"] = cr["ORG_FLAG"].astype(str).str.upper().str.strip()
    opt_cols = [
        "CITY", "YEAR", "EXTENT_HA", "NREQ_KG", "P2O5REQ_KG", "K2OREQ_KG", "BASE_DIGEST_C_KG",
        "BASE_SYN_KG", "BASE_SYN_KM", "BASE_SYN_N_KG", "BASE_SYN_P_KG", "BASE_SYN_K_KG",
        "BASE_COMPOST_IMP_KG", "BASE_COMPOST_IMP_KM", "BASE_COMPOST_LOC_KG", "BASE_COMPOST_LOC_KM",
        "BASE_COMPOST_N_KG", "BASE_COMPOST_P_KG", "BASE_COMPOST_K_KG",
        "BASE_DIGEST_KG", "BASE_DIGEST_KM", "BASE_DIGEST_N_KG", "BASE_DIGEST_P_KG",
        "BASE_AGROCHEM_KG", "BASE_AGROCHEM_KM",
        "BASE_AG_DIESEL_L", "BASE_AG_DIESEL_MJ", "BASE_AG_GASOL_MJ", "BASE_AG_KEROS_MJ", "BASE_AG_ELEC_KWH",
        "BASE_NH3_KG", "BASE_NO_KG", "BASE_NRUNOFF_KG", "BASE_NLEACH_KG",
        "BASE_RESIDUE_C_KG", "BASE_SAND", "BASE_IRRIG_TERM",
    ]
    for col in opt_cols:
        if col not in cr.columns:
            cr[col] = 0.0
        cr[col] = to_num(cr[col], 0.0)
    cr["YEAR"] = cr["YEAR"].astype(int)
    cr = cr[cr["YEAR"].isin(years) & cr["CROP"].isin(["RICE", "TEA", "VEG"])].copy()
    cr = cr[cr["NREQ_KG"] > 1e-9].copy()

    if cr.empty:
        return cr.reset_index(drop=True)

    w = cr["EXTENT_HA"].clip(lower=0.0).replace(0.0, np.nan)
    for col in ["BASE_SYN_KM", "BASE_COMPOST_IMP_KM", "BASE_COMPOST_LOC_KM", "BASE_DIGEST_KM", "BASE_AGROCHEM_KM", "BASE_SAND", "BASE_IRRIG_TERM"]:
        cr[f"_{col}_WX"] = cr[col] * w.fillna(0.0)

    sum_cols = [
        "EXTENT_HA", "NREQ_KG", "P2O5REQ_KG", "K2OREQ_KG", "BASE_DIGEST_C_KG",
        "BASE_SYN_KG", "BASE_SYN_N_KG", "BASE_SYN_P_KG", "BASE_SYN_K_KG",
        "BASE_COMPOST_IMP_KG", "BASE_COMPOST_LOC_KG", "BASE_COMPOST_N_KG", "BASE_COMPOST_P_KG", "BASE_COMPOST_K_KG",
        "BASE_DIGEST_KG", "BASE_DIGEST_N_KG", "BASE_DIGEST_P_KG",
        "BASE_AGROCHEM_KG", "BASE_AG_DIESEL_L", "BASE_AG_DIESEL_MJ", "BASE_AG_GASOL_MJ", "BASE_AG_KEROS_MJ", "BASE_AG_ELEC_KWH",
        "BASE_NH3_KG", "BASE_NO_KG", "BASE_NRUNOFF_KG", "BASE_NLEACH_KG", "BASE_RESIDUE_C_KG",
    ]
    wx_cols = [c for c in cr.columns if c.startswith("_") and c.endswith("_WX")]
    agg = {c: "sum" for c in sum_cols + wx_cols if c in cr.columns}
    agg["CITY"] = "first"
    agg["ORG_FLAG"] = lambda x: "Y" if (x.astype(str).str.upper() == "Y").all() else "X"
    out = cr.groupby(["KEY", "YEAR", "CROP"], as_index=False).agg(agg)
    extent = out["EXTENT_HA"].replace(0.0, np.nan)
    for col in ["BASE_SYN_KM", "BASE_COMPOST_IMP_KM", "BASE_COMPOST_LOC_KM", "BASE_DIGEST_KM", "BASE_AGROCHEM_KM", "BASE_SAND", "BASE_IRRIG_TERM"]:
        wx_col = f"_{col}_WX"
        if wx_col in out.columns:
            out[col] = (out[wx_col] / extent).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out = out.drop(columns=[wx_col])
        elif col not in out.columns:
            out[col] = 0.0
    out["CULTIVAR"] = "ALL"
    out = out.sort_values(["KEY", "YEAR", "CROP"]).reset_index(drop=True)
    return out

def marginal_rice_field_gwp_per_kg(n_conc: float, c_conc: float) -> float:
    ch4 = c_conc * 0.09509 * (16.0 / 12.0) * 27.2
    n2o = n_conc * 0.0038 * (44.0 / 28.0) * 273.0
    return float(ch4 + n2o)


def marginal_nonrice_field_gwp_per_kg(crop: str, n_conc: float) -> float:
    crop = str(crop).upper()
    if crop == "TEA":
        return float(n_conc * 0.0017 * (44.0 / 28.0) * 273.0)
    if crop == "VEG":
        return float(n_conc * 0.0017 * (44.0 / 28.0) * 273.0)
    return 0.0


RESIDUE_REQ_COEF_OPT = {
    "DAIRY": 0.38,
    "CATTLE": 0.38,
    "SWINE": 0.132,
    "CHICKEN": 0.096,
    "BROILER": 0.096,
}
COMPOST_COEF_OPT = {
    "DAIRY": 0.665,
    "CATTLE": 0.665,
    "SWINE": 0.415,
    "CHICKEN": 0.327,
    "BROILER": 0.327,
}


def elec_gwp_for_year(elec_ef: pd.DataFrame, year: int) -> float:
    if elec_ef.empty:
        return 0.0
    if "YEAR" in elec_ef.columns:
        sub = elec_ef[pd.to_numeric(elec_ef["YEAR"], errors="coerce").eq(int(year))]
        if not sub.empty and "ELEC_GWP" in sub.columns:
            return float(to_num(sub.iloc[-1]["ELEC_GWP"], 0.0))
    return float(to_num(elec_ef.get("ELEC_GWP", pd.Series([0.0])), 0.0).mean()) if "ELEC_GWP" in elec_ef.columns else 0.0


def ad_coeff_per_input_kg(ad_coeff: dict[tuple[str, int], dict[str, float]], typ: str, scale: int, field: str) -> float:
    typ = str(typ).upper()
    scale = int(scale)
    rec = ad_coeff.get((typ, scale), {})
    cap = float(rec.get("CAPACITY_KG_YEAR", 0.0))
    if cap <= 0:
        cap = float(scale) * 365000.0
    if cap <= 0:
        return 0.0
    return float(rec.get(field, 0.0)) / cap


def ad_process_gwp_coeff_per_kg(
    settings: OptimizeSettings,
    typ: str,
    scale: int,
    ef: pd.DataFrame,
    elec_gwp_y: float,
    ad_coeff: dict[tuple[str, int], dict[str, float]],
    *,
    include_heat: bool,
) -> float:
    typ = str(typ).upper()
    combust_ef = ef_value(ef, "GWP", "TDAD_COMBUST" if typ == "TDAD" else "MWAD_COMBUST", 0.0)
    water_ef = 0.0 if typ == "TDAD" else ef_value(ef, "GWP", "MWAD_WATER", 0.0)
    heat_ef = ef_value(ef, "GWP", "HEAT", 0.0)
    ref_trans_ef = ef_value(ef, "GWP", "TDAD_REF_TRANS" if typ == "TDAD" else "MWAD_REF_TRANS", 0.0)
    sludge_ef = ef_value(ef, "GWP", "TDAD_SLUDGE" if typ == "TDAD" else "MWAD_SLUDGE", 0.0)
    gridgas_ef = ef_value(ef, "GWP", "GRID_GAS", 0.0)
    val = 0.0
    val += ad_coeff_per_input_kg(ad_coeff, typ, scale, "BIOGAS_NM3") * BIOGAS_HV_MJ_PER_NM3 * combust_ef
    val += ad_coeff_per_input_kg(ad_coeff, typ, scale, "ELEC_KWH") * elec_gwp_y
    if include_heat:
        val += ad_coeff_per_input_kg(ad_coeff, typ, scale, "HEAT_MJ") * heat_ef
    val += ad_coeff_per_input_kg(ad_coeff, typ, scale, "WATER_M3") * water_ef

    val += ad_coeff_per_input_kg(ad_coeff, typ, scale, "REFUSE_KG") * 0.001 * 10.0 * ref_trans_ef
    digest_kg_per_kg = ad_coeff_per_input_kg(ad_coeff, typ, scale, "DIGEST_KG")
    digsolid_kg_per_kg = digest_kg_per_kg * 0.10
    val += digsolid_kg_per_kg * 0.001 * sludge_ef

    if typ == "TDAD":
        val += digest_kg_per_kg * 0.001 * ef_value(ef, "GWP", "TDAD_DIGWW", 0.0)
    if settings.enable_gridgas:
        val -= ad_coeff_per_input_kg(ad_coeff, typ, scale, "BIOGAS_NM3") * gridgas_ef
    return float(val)


def field_direct_digest_gwp_per_kg(crop_name: str, n_conc: float, c_conc: float, extent_ha: float) -> float:
    crop_name = str(crop_name).upper()
    if crop_name == "RICE":
        ch4 = c_conc * 0.09509 * (16.0 / 12.0) * 27.2
        n2o = n_conc * 0.0038 * (44.0 / 28.0) * 273.0
        return float(ch4 + n2o)
    if crop_name == "TEA":
        return float(n_conc * 0.0017 * (44.0 / 28.0) * 273.0)
    if crop_name == "VEG":
        return float(n_conc * 0.0017 * (44.0 / 28.0) * 273.0)
    return 0.0


def field_indirect_digest_gwp_per_kg(crop_name: str, n_conc: float, base_nreq: float, base_nh3: float, base_no: float, base_nleach: float, base_nrunoff: float) -> float:
    if base_nreq <= 0 or n_conc <= 0:
        return 0.0
    volatil = (base_nh3 + base_no) / base_nreq * n_conc
    leach_runoff = (base_nleach + base_nrunoff) / base_nreq * n_conc
    return float((volatil * 0.014 + leach_runoff * 0.011) * (44.0 / 28.0) * 273.0)


def crop_gwp_delta_coeff_per_kg_digest(
    r: Any,
    d_km: float,
    n_conc: float,
    p_conc: float,
    k_conc: float,
    c_conc: float,
    ef: pd.DataFrame,
) -> float:

    crop_name = str(getattr(r, "CROP", "")).upper()
    nreq = float(to_num(getattr(r, "NREQ_KG", 0.0), 0.0))
    preq = float(to_num(getattr(r, "P2O5REQ_KG", 0.0), 0.0))
    kreq = float(to_num(getattr(r, "K2OREQ_KG", 0.0), 0.0))
    extent = float(to_num(getattr(r, "EXTENT_HA", 0.0), 0.0))
    org_flag = str(getattr(r, "ORG_FLAG", "")).upper()

    ef_syn_n = ef_value(ef, "GWP", "SYN_N", 0.0)
    ef_syn_p = ef_value(ef, "GWP", "SYN_P", 0.0)
    ef_syn_k = ef_value(ef, "GWP", "SYN_K", 0.0)
    ef_syn_trans = ef_value(ef, "GWP", "SYN_TRANS", 0.0)
    ef_org = ef_value(ef, "GWP", "ORG", 0.0)
    ef_org_trans = ef_value(ef, "GWP", "ORG_TRANS", 0.0)
    ef_dig_trans = ef_value(ef, "GWP", "DIG_TRANS", ef_org_trans)
    ef_agrochem = ef_value(ef, "GWP", "AGROCHEM", 0.0)
    ef_agrochem_trans = ef_value(ef, "GWP", "AGROCHEM_TRANS", 0.0)
    ef_diesel = ef_value(ef, "GWP", "DIESEL_E", 0.0)

    coeff = 0.0
    coeff += 0.001 * float(d_km) * ef_dig_trans
    coeff += field_direct_digest_gwp_per_kg(crop_name, n_conc, c_conc, extent)
    coeff += field_indirect_digest_gwp_per_kg(
        crop_name, n_conc, nreq,
        float(to_num(getattr(r, "BASE_NH3_KG", 0.0), 0.0)),
        float(to_num(getattr(r, "BASE_NO_KG", 0.0), 0.0)),
        float(to_num(getattr(r, "BASE_NLEACH_KG", 0.0), 0.0)),
        float(to_num(getattr(r, "BASE_NRUNOFF_KG", 0.0), 0.0)),
    )

    coeff += 0.001 * 60.3 * 15.0 / 60.0 * 3.6 * ef_diesel

    syn_n_share = 0.0 if org_flag == "Y" else min(n_conc, nreq) if nreq > 0 else 0.0
    syn_p_share = 0.0 if org_flag == "Y" else min(p_conc, preq) if preq > 0 else 0.0
    syn_k_share = 0.0 if org_flag == "Y" else min(k_conc, kreq) if kreq > 0 else 0.0
    coeff -= syn_n_share * ef_syn_n + syn_p_share * ef_syn_p + syn_k_share * ef_syn_k
    syn_km = float(to_num(getattr(r, "BASE_SYN_KM", 0.0), 0.0))
    syn_total_per_kg = syn_n_share + syn_p_share + syn_k_share
    coeff -= syn_total_per_kg * 0.001 * syn_km * ef_syn_trans

    base_comp_imp = float(to_num(getattr(r, "BASE_COMPOST_IMP_KG", 0.0), 0.0))
    base_comp_loc = float(to_num(getattr(r, "BASE_COMPOST_LOC_KG", 0.0), 0.0))
    base_comp_n = float(to_num(getattr(r, "BASE_COMPOST_N_KG", 0.0), 0.0))
    comp_replace_kg = (n_conc / base_comp_n) * (base_comp_imp + base_comp_loc) if base_comp_n > 0 else 0.0
    if comp_replace_kg > 0:
        imp_share = base_comp_imp / max(base_comp_imp + base_comp_loc, 1e-12)
        loc_share = base_comp_loc / max(base_comp_imp + base_comp_loc, 1e-12)
        comp_imp_km = float(to_num(getattr(r, "BASE_COMPOST_IMP_KM", 0.0), 0.0))
        comp_loc_km = float(to_num(getattr(r, "BASE_COMPOST_LOC_KM", 0.0), 0.0))
        coeff -= comp_replace_kg * imp_share * ef_org
        coeff -= comp_replace_kg * imp_share * 0.001 * comp_imp_km * ef_org_trans
        coeff -= comp_replace_kg * loc_share * 0.001 * comp_loc_km * ef_org_trans

    base_syn_n = float(to_num(getattr(r, "BASE_SYN_N_KG", 0.0), 0.0))
    base_agrochem = float(to_num(getattr(r, "BASE_AGROCHEM_KG", 0.0), 0.0))
    if base_syn_n > 0 and base_agrochem > 0 and syn_n_share > 0:
        agro_per_kg_digest = base_agrochem * min(n_conc / base_syn_n, 1.0)
        agro_km = float(to_num(getattr(r, "BASE_AGROCHEM_KM", 0.0), 0.0))
        coeff -= agro_per_kg_digest * ef_agrochem
        coeff -= agro_per_kg_digest * 0.001 * (300.0 + agro_km) * ef_agrochem_trans

    return float(coeff)


def source_field_direct_gwp_coeff_per_kg_n(crop_name: str, source: str) -> float:
    crop_name = str(crop_name).upper()
    source = str(source).upper()
    if crop_name == "RICE":
        ef_n = {"SYN": 0.0031, "COMPOST": 0.0116, "DIGEST": 0.0038}.get(source, 0.0)
    elif crop_name == "TEA":
        ef_n = {"SYN": 0.021, "COMPOST": 0.0234375, "DIGEST": 0.0017}.get(source, 0.0)
    elif crop_name == "VEG":
        ef_n = {"SYN": 0.0046, "COMPOST": 0.015, "DIGEST": 0.0017}.get(source, 0.0)
    else:
        ef_n = 0.0
    return float(ef_n * (44.0 / 28.0) * 273.0)


def source_field_indirect_gwp_coeff_per_kg_n(r: Any) -> float:
    nreq = float(to_num(getattr(r, "NREQ_KG", 0.0), 0.0))
    if nreq <= 0:
        return 0.0
    nh3 = float(to_num(getattr(r, "BASE_NH3_KG", 0.0), 0.0))
    no = float(to_num(getattr(r, "BASE_NO_KG", 0.0), 0.0))
    nleach = float(to_num(getattr(r, "BASE_NLEACH_KG", 0.0), 0.0))
    nrunoff = float(to_num(getattr(r, "BASE_NRUNOFF_KG", 0.0), 0.0))
    return float((((nh3 + no) / nreq) * 0.014 + ((nleach + nrunoff) / nreq) * 0.011) * (44.0 / 28.0) * 273.0)


def synthetic_gwp_coeffs_per_kg_nutrient(r: Any, ef: pd.DataFrame) -> dict[str, float]:
    """Estimator-aligned GWP coefficients for synthetic N/P2O5/K2O nutrient variables."""
    syn_km = float(to_num(getattr(r, "BASE_SYN_KM", 0.0), 0.0))
    trans = 0.001 * syn_km * ef_value(ef, "GWP", "SYN_TRANS", 0.0)
    crop_name = str(getattr(r, "CROP", "")).upper()
    indirect_n = source_field_indirect_gwp_coeff_per_kg_n(r)
    field_n = source_field_direct_gwp_coeff_per_kg_n(crop_name, "SYN") + indirect_n
    base_syn_n = float(to_num(getattr(r, "BASE_SYN_N_KG", 0.0), 0.0))
    base_agro = float(to_num(getattr(r, "BASE_AGROCHEM_KG", 0.0), 0.0))
    agro_km = float(to_num(getattr(r, "BASE_AGROCHEM_KM", 0.0), 0.0))
    if base_syn_n > 0 and base_agro > 0:
        agro_per_kg_n = base_agro / base_syn_n
        field_n += agro_per_kg_n * ef_value(ef, "GWP", "AGROCHEM", 0.0)
        field_n += agro_per_kg_n * 0.001 * (300.0 + agro_km) * ef_value(ef, "GWP", "AGROCHEM_TRANS", 0.0)
    return {
        "N": ef_value(ef, "GWP", "SYN_N", 0.0) + trans + field_n,
        "P": ef_value(ef, "GWP", "SYN_P", 0.0) + trans,
        "K": ef_value(ef, "GWP", "SYN_K", 0.0) + trans,
    }


def compost_concentrations_from_baseline(r: Any) -> tuple[float, float, float]:

    loc_kg = max(float(to_num(getattr(r, "BASE_COMPOST_LOC_KG", 0.0), 0.0)), 0.0)
    imp_kg = max(float(to_num(getattr(r, "BASE_COMPOST_IMP_KG", 0.0), 0.0)), 0.0)
    total_kg = loc_kg + imp_kg

    total_n = max(float(to_num(getattr(r, "BASE_COMPOST_N_KG", 0.0), 0.0)), 0.0)
    total_p = max(float(to_num(getattr(r, "BASE_COMPOST_P_KG", 0.0), 0.0)), 0.0)
    total_k = max(float(to_num(getattr(r, "BASE_COMPOST_K_KG", 0.0), 0.0)), 0.0)

    loc_n_raw = to_num(getattr(r, "BASE_COMPOST_LOC_N_KG", np.nan), np.nan)
    loc_p_raw = to_num(getattr(r, "BASE_COMPOST_LOC_P_KG", np.nan), np.nan)
    loc_k_raw = to_num(getattr(r, "BASE_COMPOST_LOC_K_KG", np.nan), np.nan)

    if loc_kg > 0:
        loc_n = float(loc_n_raw) if np.isfinite(loc_n_raw) else total_n * loc_kg / max(total_kg, 1e-12)
        loc_p = float(loc_p_raw) if np.isfinite(loc_p_raw) else total_p * loc_kg / max(total_kg, 1e-12)
        loc_k = float(loc_k_raw) if np.isfinite(loc_k_raw) else total_k * loc_kg / max(total_kg, 1e-12)
        return max(loc_n / loc_kg, 0.0), max(loc_p / loc_kg, 0.0), max(loc_k / loc_kg, 0.0)

    if total_kg > 0:
        return max(total_n / total_kg, 0.0), max(total_p / total_kg, 0.0), max(total_k / total_kg, 0.0)


    return 0.037, 0.0, 0.0


def compost_material_gwp_coeff(r: Any, ef: pd.DataFrame, source: str) -> float:

    source = str(source).upper()
    crop_name = str(getattr(r, "CROP", "")).upper()
    nconc, _, _ = compost_concentrations_from_baseline(r)
    coeff = 0.0
    if source == "IMP":
        coeff += ef_value(ef, "GWP", "ORG", 0.0)
        coeff += 0.001 * float(to_num(getattr(r, "BASE_COMPOST_IMP_KM", 0.0), 0.0)) * ef_value(ef, "GWP", "ORG_TRANS", 0.0)
    else:
        coeff += 0.001 * float(to_num(getattr(r, "BASE_COMPOST_LOC_KM", 0.0), 0.0)) * ef_value(ef, "GWP", "ORG_TRANS", 0.0)
    coeff += nconc * source_field_direct_gwp_coeff_per_kg_n(crop_name, "COMPOST")
    coeff += nconc * source_field_indirect_gwp_coeff_per_kg_n(r)
    return float(coeff)


def digest_material_gwp_coeff_absolute(r: Any, d_km: float, n_conc: float, p_conc: float, k_conc: float, c_conc: float, ef: pd.DataFrame) -> float:

    crop_name = str(getattr(r, "CROP", "")).upper()
    coeff = 0.0
    coeff += 0.001 * float(d_km) * ef_value(ef, "GWP", "DIG_TRANS", ef_value(ef, "GWP", "ORG_TRANS", 0.0))
    coeff += field_direct_digest_gwp_per_kg(crop_name, n_conc, c_conc, float(to_num(getattr(r, "EXTENT_HA", 0.0), 0.0)))
    coeff += field_indirect_digest_gwp_per_kg(
        crop_name, n_conc, float(to_num(getattr(r, "NREQ_KG", 0.0), 0.0)),
        float(to_num(getattr(r, "BASE_NH3_KG", 0.0), 0.0)),
        float(to_num(getattr(r, "BASE_NO_KG", 0.0), 0.0)),
        float(to_num(getattr(r, "BASE_NLEACH_KG", 0.0), 0.0)),
        float(to_num(getattr(r, "BASE_NRUNOFF_KG", 0.0), 0.0)),
    )

    coeff += 0.001 * 60.3 * 15.0 / 60.0 * 3.6 * ef_value(ef, "GWP", "DIESEL_E", 0.0)
    return float(coeff)

def build_flow_explicit_model(
    settings: OptimizeSettings,
    candidates: pd.DataFrame,
    price_y: pd.DataFrame,
    ef: pd.DataFrame,
    elec_ef: pd.DataFrame,
    ad_lookup: pd.DataFrame,
    param_tables: dict[str, pd.DataFrame],
):

    if gp is None:
        return None, {}, {}
    feed = param_tables["feedstock"].copy()
    dist = param_tables["distance"].copy()
    crop_req = param_tables.get("crop_req", pd.DataFrame()).copy()
    if feed.empty or dist.empty:
        return None, {}, {}

    feed["KEY"] = feed["KEY"].astype(str)
    feed["YEAR"] = to_num(feed["YEAR"], 0).astype(int)
    dist["KEY"] = dist["KEY"].astype(str)
    dist["FACILITY"] = dist["FACILITY"].astype(str)
    dist["DIST_KM"] = to_num(dist["DIST_KM"], 0.0)

    years = sorted([y for y in feed["YEAR"].unique().tolist() if y in YEARS]) or YEARS
    keys = sorted(feed["KEY"].unique().tolist())
    facs = sorted(candidates["FACILITY"].astype(str).unique().tolist())
    cand = candidates.copy()
    cand["FACILITY"] = cand["FACILITY"].astype(str)
    cand = cand.set_index("FACILITY")
    dmap = {(str(r.KEY), str(r.FACILITY)): float(r.DIST_KM) for r in dist.itertuples(index=False)}

    for col in [
        "FOOD_KG", "PAPER_KG", "WOOD_KG", "PLA_KG", "TEX_KG", "OTH_KG",
        "MANURE_DAIRY_KG", "MANURE_CATTLE_KG", "MANURE_SWINE_KG", "MANURE_CHICKEN_KG", "MANURE_BROILER_KG",
    ]:
        if col not in feed.columns:
            feed[col] = 0.0
        feed[col] = to_num(feed[col], 0.0)
    feed["MANURE_TOTAL_KG"] = feed[["MANURE_DAIRY_KG", "MANURE_CATTLE_KG", "MANURE_SWINE_KG", "MANURE_CHICKEN_KG", "MANURE_BROILER_KG"]].sum(axis=1)
    # CITY/event-year lookup for the post-EVENT food-waste INC policy rule.
    # Once an EVENT occurs in a boundary/city, food waste in that city is forced
    # away from fallback INC except for diagnostic slack.
    event_city_year = build_event_city_year_lookup(candidates)
    key_city = {}
    if "CITY" in feed.columns:
        tmp_city = feed[["KEY", "CITY"]].dropna().copy()
        tmp_city["KEY"] = tmp_city["KEY"].astype(str)
        tmp_city["CITY"] = to_num(tmp_city["CITY"], -1).astype(int)
        key_city = tmp_city.groupby("KEY")["CITY"].first().to_dict()

    fy = feed.set_index(["KEY", "YEAR"])
    def avail(key: str, year: int, col: str) -> float:
        try:
            return float(fy.at[(key, year), col])
        except Exception:
            return 0.0

    # Tight global upper bound by year for capacity relaxation.
    x_year = feed.groupby("YEAR")[["FOOD_KG", "PAPER_KG", "WOOD_KG", "MANURE_TOTAL_KG"]].sum().sum(axis=1).to_dict()
    crop_opt = crop_req_records_for_agriculture(settings, crop_req, years)
    crop_by_key: dict[str, list[Any]] = {}
    if not crop_opt.empty:
        for r in crop_opt.itertuples(index=False):
            crop_by_key.setdefault(str(r.KEY), []).append(r)

    m = gp.Model("CE_AD_flow_explicit")
    configure_gurobi_model(m, settings, "CE_AD_flow_explicit")

    types = (["COMPOST"] if compost_candidates_enabled(settings) else []) + ["MWAD", "TDAD"]
    z = {}
    s_var = {}
    scale_choice = {}
    assign = {}  # backward-compatible alias for food-route variables
    assign_food = {}
    assign_manure = {}
    assign_pw = {}
    dig_apply = {}
    syn_apply = {}
    compost_imp_apply = {}
    compost_loc_apply = {}
    nutrient_slack = {}
    dig_unused = {}
    bio_use = {}
    assign_energy = {}
    waste_inc = {}
    food_inc = {}
    event_food_inc_slack = {}
    manure_compbase = {}
    comp_prod = {}
    ad_input = {}
    ad_digest_prod = {}
    rgrp = {}
    hgrp = {}


    for f in facs:
        row = cand.loc[f]
        locked_tdad_scale = existing_tdad_fixed_scale(row)
        zsum = []
        if locked_tdad_scale is not None:

            z[f, "TDAD"] = m.addVar(vtype=GRB.BINARY, name=f"Z[{f},TDAD]")
            zsum.append(z[f, "TDAD"])
        else:
            for t in types + ["INC", "RDF"]:
                if t in types and int(row.get(f"ALLOW_{t}", 0)) == 1:
                    z[f, t] = m.addVar(vtype=GRB.BINARY, name=f"Z[{f},{t}]")
                    zsum.append(z[f, t])
                elif t in {"INC", "RDF"} and int(row.get(f"ALLOW_{t}", 0)) == 1:
                    z[f, t] = m.addVar(vtype=GRB.BINARY, name=f"Z[{f},{t}]")
                    zsum.append(z[f, t])
        if zsum:
            m.addConstr(gp.quicksum(zsum) == 1, name=f"one_type[{f}]")
        s_ub = max(int(settings.max_scale), int(locked_tdad_scale or 0))
        s_var[f] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=s_ub, name=f"S[{f}]")
        scale_terms = []
        for t in ["MWAD", "TDAD"]:
            if (f, t) not in z:
                continue
            y_terms = []
            if locked_tdad_scale is not None and t == "TDAD":
                scale_iter = [int(locked_tdad_scale)]
            else:
                scale_iter = range(settings.min_scale, settings.max_scale + 1, settings.scale_step)
            for sc in scale_iter:
                yv = m.addVar(vtype=GRB.BINARY, name=f"Yscale[{f},{t},{sc}]")
                scale_choice[f, t, int(sc)] = yv
                y_terms.append(yv)
                scale_terms.append(int(sc) * yv)
            m.addConstr(gp.quicksum(y_terms) == z[f, t], name=f"scale_choice[{f},{t}]")
        m.addConstr(s_var[f] == gp.quicksum(scale_terms), name=f"scale_def[{f}]")


        ad_selected = gp.quicksum(z.get((f, tt), 0) for tt in ["MWAD", "TDAD"])
        if (f, "MWAD") in z or (f, "TDAD") in z:
            bio_use[f, "ELEC"] = m.addVar(vtype=GRB.BINARY, name=f"Ubio[{f},ELEC]")
            bio_use[f, "GRIDGAS"] = m.addVar(vtype=GRB.BINARY, name=f"Ubio[{f},GRIDGAS]")
            m.addConstr(bio_use[f, "ELEC"] + bio_use[f, "GRIDGAS"] == ad_selected, name=f"bio_one[{f}]")
            if not settings.enable_gridgas:
                m.addConstr(bio_use[f, "GRIDGAS"] == 0, name=f"bio_gridgas_disabled[{f}]")


    vtype = GRB.BINARY if settings.link_mode == "single" else GRB.CONTINUOUS

    def _add_route_var(store: dict, prefix: str, k: str, f: str, t: str, y: int):
        v = m.addVar(vtype=vtype, lb=0.0, ub=1.0, name=f"{prefix}[{k},{f},{t},{y}]")
        store[k, f, t, y] = v
        m.addConstr(v <= z[f, t], name=f"{prefix}_type[{k},{f},{t},{y}]")
        return v

    for k in keys:
        for y in years:
            food_vars_ky = []
            manure_vars_ky = []
            pw_vars_ky = []
            for f in facs:
                if (k, f) not in dmap:
                    continue
                for t in types:
                    if (f, t) not in z:
                        continue

                    bf = _add_route_var(assign_food, "Bfood", k, f, t, y)

                    assign[k, f, t, y] = bf
                    food_vars_ky.append(bf)

                    bm = _add_route_var(assign_manure, "Bmanure", k, f, t, y)
                    manure_vars_ky.append(bm)

                    if t == "TDAD":
                        bpw = _add_route_var(assign_pw, "Bpw", k, f, t, y)
                        pw_vars_ky.append(bpw)
                    if t in {"MWAD", "TDAD"}:
                        for stream_name, route_var in [("food", bf), ("manure", bm)]:
                            for g in ["ELEC", "GRIDGAS"]:
                                u = bio_use.get((f, g))
                                if u is not None:
                                    ae = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"AE[{stream_name},{k},{f},{t},{g},{y}]")
                                    assign_energy[stream_name, k, f, t, g, y] = ae
                                    m.addConstr(ae <= route_var, name=f"ae_{stream_name}_le_b[{k},{f},{t},{g},{y}]")
                                    m.addConstr(ae <= u, name=f"ae_{stream_name}_le_u[{k},{f},{t},{g},{y}]")
                                    m.addConstr(ae >= route_var + u - 1.0, name=f"ae_{stream_name}_ge_b_plus_u[{k},{f},{t},{g},{y}]")
                        if t == "TDAD":
                            bpw0 = assign_pw.get((k, f, t, y))
                            if bpw0 is not None:
                                for g in ["ELEC", "GRIDGAS"]:
                                    u = bio_use.get((f, g))
                                    if u is not None:
                                        ae = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"AE[pw,{k},{f},{t},{g},{y}]")
                                        assign_energy["pw", k, f, t, g, y] = ae
                                        m.addConstr(ae <= bpw0, name=f"ae_pw_le_b[{k},{f},{t},{g},{y}]")
                                        m.addConstr(ae <= u, name=f"ae_pw_le_u[{k},{f},{t},{g},{y}]")
                                        m.addConstr(ae >= bpw0 + u - 1.0, name=f"ae_pw_ge_b_plus_u[{k},{f},{t},{g},{y}]")
            if food_vars_ky:
                m.addConstr(gp.quicksum(food_vars_ky) <= 1.0, name=f"single_food_origin[{k},{y}]")
            if manure_vars_ky:
                m.addConstr(gp.quicksum(manure_vars_ky) <= 1.0, name=f"single_manure_origin[{k},{y}]")
            if pw_vars_ky:
                m.addConstr(gp.quicksum(pw_vars_ky) <= 1.0, name=f"single_paperwood_origin[{k},{y}]")


    for k in keys:
        for y in years:
            food_sum = gp.quicksum(v for (kk, _, _, yy), v in assign_food.items() if kk == k and yy == y)
            manure_sum = gp.quicksum(v for (kk, _, _, yy), v in assign_manure.items() if kk == k and yy == y)
            pw_sum = gp.quicksum(v for (kk, _, _, yy), v in assign_pw.items() if kk == k and yy == y)
            food_total = avail(k, y, "FOOD_KG")
            pw_total = avail(k, y, "PAPER_KG") + avail(k, y, "WOOD_KG")
            manure_total = avail(k, y, "MANURE_TOTAL_KG")
            wi = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"Winc[{k},{y}]")
            fwi = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"FWinc[{k},{y}]")
            mc = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"Mcompbase[{k},{y}]")
            waste_inc[k, y] = wi
            food_inc[k, y] = fwi
            manure_compbase[k, y] = mc
            m.addConstr(fwi == food_total * (1.0 - food_sum), name=f"fallback_food_inc[{k},{y}]")
            m.addConstr(wi == food_total * (1.0 - food_sum) + pw_total * (1.0 - pw_sum), name=f"fallback_waste_inc[{k},{y}]")
            m.addConstr(mc == manure_total * (1.0 - manure_sum), name=f"fallback_manure_compbase[{k},{y}]")
            city = int(key_city.get(str(k), -1)) if key_city else -1
            event_year = event_city_year.get(city)
            if event_year is not None and int(y) >= int(event_year):
                es = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"SeventFWinc[{k},{y}]")
                event_food_inc_slack[k, y] = es
                m.addConstr(fwi <= es, name=f"event_food_inc_limit[{k},{y}]")


    for f in facs:
        row = cand.loc[f]
        open_year = int(to_num(row.get("BUILT_YEAR_BASE", row.get("OPEN_YEAR", 2030)), 2030))
        for y in years:
            input_expr = gp.LinExpr()
            ad_input_expr = gp.LinExpr()
            comp_input_expr = gp.LinExpr()
            comp_reachable_ub = 0.0
            for k in keys:
                food = avail(k, y, "FOOD_KG")
                paper = avail(k, y, "PAPER_KG")
                wood = avail(k, y, "WOOD_KG")
                pw = paper + wood
                manure = avail(k, y, "MANURE_TOTAL_KG")
                bf_comp = assign_food.get((k, f, "COMPOST", y))
                bm_comp = assign_manure.get((k, f, "COMPOST", y))
                if bf_comp is not None or bm_comp is not None:
                    comp_reachable_ub += food + manure
                for t in types:
                    bf = assign_food.get((k, f, t, y))
                    bm = assign_manure.get((k, f, t, y))
                    bpw = assign_pw.get((k, f, t, y)) if t == "TDAD" else None
                    kg_expr = gp.LinExpr()
                    if bf is not None and food:
                        kg_expr += food * bf
                    if bm is not None and manure:
                        kg_expr += manure * bm
                    if bpw is not None and pw:
                        kg_expr += pw * bpw
                    if kg_expr.size() > 0:
                        input_expr += kg_expr
                        if t in {"MWAD", "TDAD"}:
                            ad_input_expr += kg_expr
                        elif t == "COMPOST":
                            comp_input_expr += kg_expr
            if y < open_year:
                m.addConstr(input_expr <= 0.0, name=f"opening[{f},{y}]")
            m.addConstr(ad_input_expr <= 365000.0 * s_var[f], name=f"ad_capacity[{f},{y}]")
            if (f, "COMPOST") in z:
                base_comp_cap = float(to_num(row.get("BASE_CAPACITY_KG", 0.0), 0.0))
                comp_cap = base_comp_cap if base_comp_cap > 0 else comp_reachable_ub
                comp_cap = max(float(comp_cap), 0.0)
                m.addConstr(comp_input_expr <= comp_cap * z[f, "COMPOST"], name=f"compost_capacity[{f},{y}]")

            for tt in ["MWAD", "TDAD"]:
                if (f, tt) in z:
                    ad_in = gp.LinExpr()
                    for k in keys:
                        food = avail(k, y, "FOOD_KG")
                        pw = avail(k, y, "PAPER_KG") + avail(k, y, "WOOD_KG")
                        manure = avail(k, y, "MANURE_TOTAL_KG")
                        bf = assign_food.get((k, f, tt, y))
                        bm = assign_manure.get((k, f, tt, y))
                        bpw = assign_pw.get((k, f, tt, y)) if tt == "TDAD" else None
                        if bf is not None and food:
                            ad_in += food * bf
                        if bm is not None and manure:
                            ad_in += manure * bm
                        if bpw is not None and pw:
                            ad_in += pw * bpw
                    adv = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"ADinput[{f},{tt},{y}]")
                    ad_input[f, tt, y] = adv
                    m.addConstr(adv == ad_in, name=f"ad_input_def[{f},{tt},{y}]")

            if (f, "COMPOST") in z:
                comp_in = gp.LinExpr()
                for k in keys:
                    food = avail(k, y, "FOOD_KG")
                    manure = avail(k, y, "MANURE_TOTAL_KG")
                    bf = assign_food.get((k, f, "COMPOST", y))
                    bm = assign_manure.get((k, f, "COMPOST", y))
                    if bf is not None and food:
                        comp_in += food * bf
                    if bm is not None and manure:
                        comp_in += manure * bm
                cp = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"COMPprod[{f},{y}]")
                comp_prod[f, y] = cp
                m.addConstr(cp == 0.50 * comp_in, name=f"comp_prod_def[{f},{y}]")

                rgrp[f, y] = m.addVar(vtype=GRB.BINARY, name=f"Rcomp[{f},{y}]")
                hgrp[f, y] = m.addVar(vtype=GRB.BINARY, name=f"Hcomp[{f},{y}]")
                m.addConstr(rgrp[f, y] + hgrp[f, y] <= z[f, "COMPOST"], name=f"comp_group_one[{f},{y}]")
                l1_expr = gp.LinExpr(); l2_expr = gp.LinExpr()
                for k in keys:
                    bm = assign_manure.get((k, f, "COMPOST", y))
                    if bm is None:
                        continue
                    l1 = avail(k, y, "MANURE_DAIRY_KG") + avail(k, y, "MANURE_CATTLE_KG") + avail(k, y, "MANURE_SWINE_KG")
                    l2 = avail(k, y, "MANURE_CHICKEN_KG") + avail(k, y, "MANURE_BROILER_KG")
                    if l1: l1_expr += l1 * bm
                    if l2: l2_expr += l2 * bm
                x = max(float(x_year.get(y, 0.0)), 1.0)
                m.addConstr(l1_expr <= x * (1 - z[f, "COMPOST"] + rgrp[f, y]), name=f"comp_l1[{f},{y}]")
                m.addConstr(l2_expr <= x * (1 - z[f, "COMPOST"] + hgrp[f, y]), name=f"comp_l2[{f},{y}]")


    target = settings.target.upper()
    ad_coeff = ad_lookup_coefficients(ad_lookup)

    ef_waste_trans = ef_value(ef, "GWP", "WASTE_TRANS", 0.0)
    ef_inc_keros = ef_value(ef, "GWP", "INC_KEROS_E", 0.0)
    ef_ash_trans = ef_value(ef, "GWP", "ASH_TRANS", 0.0)
    ef_ash_landfill = ef_value(ef, "GWP", "ASH_LANDFILL", 0.0)
    ef_build = ef_value(ef, "GWP", "WASTE_REBUILD", 0.0)
    ef_residue_trans = ef_value(ef, "GWP", "TDAD_REF_TRANS", 0.0)
    ef_digww = max(ef_value(ef, "GWP", "MWAD_DIGWW", 0.0), ef_value(ef, "GWP", "TDAD_DIGWW", 0.0))
    ef_mach = ef_value(ef, "GWP", "MACHINERY", 0.0)
    public_2015 = float(price_y.get("PUBLIC_INFL_2015", pd.Series([1.0])).mean()) if not price_y.empty else 1.0

    obj = gp.LinExpr()


    for (f, t, sc), yv in scale_choice.items():
        if t == "MWAD":
            if target == "COST":
                obj += (process_cost_mwad(sc, public_2015) + capex_mwad(sc) / 21.0) * yv
            else:
                obj += capex_mwad(sc) * ef_build / 21.0 * yv  # CONSTRUCT_MWAD_GWP proxy
        elif t == "TDAD":
            if target == "COST":
                obj += (process_cost_tdad(sc, public_2015) + capex_tdad(sc) / 21.0) * yv
            else:
                obj += capex_tdad(sc) * ef_build / 21.0 * yv  # CONSTRUCT_TDAD_GWP proxy


    for (f, t), zv in z.items():
        if t not in {"INC", "RDF"}:
            continue
        row = cand.loc[f]
        event = norm_text(row.get("EVENT", ""))
        cap_kg_y = float(to_num(row.get("BASE_CAPACITY_KG", 0.0), 0.0))
        cap_scale = cap_kg_y * 0.001 / 300.0 if cap_kg_y > 0 else 0.0
        capex_event = 0.0
        if cap_scale > 0 and event in {"RENEW", "RENOV", "RENOVATE"}:
            unit_capex = math.exp(5.242 - 0.500 * math.log(max(cap_scale, 1e-9)))
            capex_event = cap_scale * unit_capex * 1e6
            obj += capex_event * ef_value(ef, "GWP", "WASTE_RENOV", 0.0) * zv
        elif cap_scale > 0 and event == "REBUILD" and t == "INC":
            unit_capex = math.exp(5.578 - 0.323 * math.log(max(cap_scale, 1e-9)) + 0.19)
            capex_event = cap_scale * unit_capex * 1e6
            obj += capex_event * ef_build * zv
        elif cap_scale > 0 and event == "REBUILD" and t == "RDF":
            capex_event = 336.61 * (cap_scale ** 0.7999) * 1e6
            obj += capex_event * ef_build * zv


    cost_tkm_proxy = 100.0
    ef_compost_food = ef_value(ef, "GWP", "COMPOST_FOOD", 0.0)

    def _add_transport_and_avoided_inc(route_var, kg: float, dist_km: float, y: int, is_waste: bool, manure_sp: str | None = None):
        nonlocal obj
        if route_var is None or kg <= 0:
            return
        if target == "COST":
            obj += kg * 0.001 * dist_km * cost_tkm_proxy * route_var
            return
        if is_waste:
            obj += kg * 0.001 * dist_km * ef_waste_trans * route_var
            obj += -kg * INC_ENERGY_L_PER_KG / 0.0274013757 * ef_inc_keros * route_var
            obj += -kg * INC_ELEC_KWH_PER_KG * elec_gwp_for_year(elec_ef, y) * route_var
            obj += -kg * DEFAULT_ASH_RATIO * 0.001 * 10.0 * ef_ash_trans * route_var
            obj += -kg * DEFAULT_ASH_RATIO * ef_ash_landfill * route_var
        elif manure_sp:
            obj += kg * 0.001 * dist_km * ef_value(ef, "GWP", f"MANURE_{manure_sp}_TRANS", 0.0) * route_var

    for k in keys:
        for f in facs:
            d = float(dmap.get((k, f), 0.0))
            for y in years:
                food = avail(k, y, "FOOD_KG")
                paper = avail(k, y, "PAPER_KG")
                wood = avail(k, y, "WOOD_KG")
                pw = paper + wood
                manure_species = {
                    "DAIRY": avail(k, y, "MANURE_DAIRY_KG"),
                    "CATTLE": avail(k, y, "MANURE_CATTLE_KG"),
                    "SWINE": avail(k, y, "MANURE_SWINE_KG"),
                    "CHICKEN": avail(k, y, "MANURE_CHICKEN_KG"),
                    "BROILER": avail(k, y, "MANURE_BROILER_KG"),
                }
                manure = sum(manure_species.values())
                for t in types:
                    bf = assign_food.get((k, f, t, y))
                    bm = assign_manure.get((k, f, t, y))
                    bpw = assign_pw.get((k, f, t, y)) if t == "TDAD" else None

                    _add_transport_and_avoided_inc(bf, food, d, y, True)
                    if t == "TDAD":
                        _add_transport_and_avoided_inc(bpw, pw, d, y, True)
                    for sp, kg in manure_species.items():
                        _add_transport_and_avoided_inc(bm, kg, d, y, False, sp)

                    if target == "GWP" and t == "COMPOST":
                        if bf is not None and food > 0:
                            obj += food * ef_compost_food * bf
                        if bm is not None:
                            for sp, kg in manure_species.items():
                                if kg <= 0:
                                    continue
                                residue_kg = kg * RESIDUE_REQ_COEF_OPT.get(sp, 0.0)
                                compost_kg = kg * COMPOST_COEF_OPT.get(sp, 0.0)
                                obj += residue_kg * 0.001 * d * ef_residue_trans * bm
                                obj += compost_kg * ef_value(ef, "GWP", f"COMPOST_{sp}", 0.0) * bm

                    if target == "GWP" and t in {"MWAD", "TDAD"}:
                        include_heat = int(cand.loc[f].get("IS_INC_SITE", 0)) != 1
                        for stream_name, kg in [("food", food), ("manure", manure), ("pw", pw if t == "TDAD" else 0.0)]:
                            if kg <= 0:
                                continue
                            for g in ["ELEC", "GRIDGAS"]:
                                ae = assign_energy.get((stream_name, k, f, t, g, y))
                                if ae is None:
                                    continue
                                class _Tmp:
                                    pass
                                tmp = _Tmp()
                                tmp.enable_gridgas = (g == "GRIDGAS")
                                coeffs = []
                                for (typ0, sc0), rec in ad_coeff.items():
                                    if typ0 == t:
                                        coeffs.append(ad_process_gwp_coeff_per_kg(tmp, t, sc0, ef, elec_gwp_for_year(elec_ef, y), ad_coeff, include_heat=include_heat))
                                ad_proc_coeff = float(np.mean(coeffs)) if coeffs else ad_gwp_per_input_kg(tmp, t, ef, elec_gwp_for_year(elec_ef, y), ad_lookup, include_heat=include_heat)
                                obj += kg * ad_proc_coeff * ae


    if settings.mode == "facility_agriculture" and not crop_opt.empty:
        crop_expr: dict[tuple[str, str, int], dict[str, Any]] = {}
        crop_row_map: dict[tuple[str, str, int], Any] = {}
        for r in crop_opt.itertuples(index=False):
            key = str(r.KEY); crop_name = str(r.CROP).upper(); year = int(r.YEAR)
            kcy = (key, crop_name, year)
            crop_row_map[kcy] = r
            crop_expr[kcy] = {
                "N": gp.LinExpr(), "P": gp.LinExpr(), "K": gp.LinExpr(),
                "N_NO_IMPORT": gp.LinExpr(), "P_NO_IMPORT": gp.LinExpr(), "K_NO_IMPORT": gp.LinExpr(),
                "NREQ": float(r.NREQ_KG), "PREQ": float(r.P2O5REQ_KG), "KREQ": float(r.K2OREQ_KG),
                "CITY": int(to_num(getattr(r, "CITY", -1), -1)),
                "COMP_N": 0.0, "COMP_P": 0.0, "COMP_K": 0.0,
            }


        vals = [v for (typ, sc), v in ad_coeff.items() if typ == "MWAD"]
        if vals:
            n_conc = float(np.mean([v["DIGEST_N_KG"] / v["DIGEST_KG"] for v in vals if v["DIGEST_KG"] > 0] or [0.0]))
            p_conc = float(np.mean([v["DIGEST_P2O5_KG"] / v["DIGEST_KG"] for v in vals if v["DIGEST_KG"] > 0] or [0.0]))
            k_conc = float(np.mean([v["DIGEST_K2O_KG"] / v["DIGEST_KG"] for v in vals if v["DIGEST_KG"] > 0] or [0.0]))
            c_conc = float(np.mean([v["DIGEST_C_KG"] / v["DIGEST_KG"] for v in vals if v["DIGEST_KG"] > 0] or [0.0]))
        else:
            n_conc = p_conc = k_conc = c_conc = 0.0

        mach_digest_expr: dict[tuple[int, str, int], Any] = {}
        mach_req: dict[tuple[int, str, int], Any] = {}
        mach_new: dict[tuple[int, str, int], Any] = {}


        for kcy, exprs in crop_expr.items():
            key, crop_name, year = kcy
            r = crop_row_map[kcy]
            org_flag = str(getattr(r, "ORG_FLAG", "")).upper()
            syn_allowed = not (settings.org_mode == "baseline" and org_flag == "Y")
            syn_coeff = synthetic_gwp_coeffs_per_kg_nutrient(r, ef)
            for nut, req_name in [("N", "NREQ"), ("P", "PREQ"), ("K", "KREQ")]:
                ub = float(exprs[req_name]) if syn_allowed else 0.0
                if ub <= 1e-12:
                    continue
                xs = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=ub, name=f"Xsyn[{key},{crop_name},{year},{nut}]")
                syn_apply[(key, crop_name, year, nut)] = xs
                exprs[nut] += xs
                exprs[f"{nut}_NO_IMPORT"] += xs
                if target == "GWP":
                    obj += syn_coeff.get(nut, 0.0) * xs

            comp_n, comp_p, comp_k = compost_concentrations_from_baseline(r)
            exprs["COMP_N"] = comp_n
            exprs["COMP_P"] = comp_p
            exprs["COMP_K"] = comp_k
            if comp_n > 0:

                max_comp_by_n = exprs["NREQ"] / max(comp_n, 1e-12)
                max_comp_by_p = exprs["PREQ"] / max(comp_p, 1e-12) if comp_p > 0 and exprs["PREQ"] > 0 else max_comp_by_n
                max_comp_by_k = exprs["KREQ"] / max(comp_k, 1e-12) if comp_k > 0 and exprs["KREQ"] > 0 else max_comp_by_n
                comp_ub = max(0.0, min(max_comp_by_n, max_comp_by_p, max_comp_by_k) if max_comp_by_k > 0 else min(max_comp_by_n, max_comp_by_p))
                if comp_ub > 1e-12:
                    xcl = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=comp_ub, name=f"XcompLoc[{key},{crop_name},{year}]")
                    compost_loc_apply[(key, crop_name, year)] = xcl
                    exprs["N"] += comp_n * xcl; exprs["P"] += comp_p * xcl; exprs["K"] += comp_k * xcl
                    exprs["N_NO_IMPORT"] += comp_n * xcl; exprs["P_NO_IMPORT"] += comp_p * xcl; exprs["K_NO_IMPORT"] += comp_k * xcl
                    if target == "GWP":
                        obj += compost_material_gwp_coeff(r, ef, "LOC") * xcl

                if comp_ub > 1e-12:
                    xci = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=comp_ub, name=f"XcompImp[{key},{crop_name},{year}]")
                    compost_imp_apply[(key, crop_name, year)] = xci
                    exprs["N"] += comp_n * xci; exprs["P"] += comp_p * xci; exprs["K"] += comp_k * xci

                    obj += max(float(settings.imported_compost_penalty), MIN_IMPORTED_COMPOST_PENALTY_RAW) * xci
                    if target == "GWP":
                        obj += compost_material_gwp_coeff(r, ef, "IMP") * xci


        for y in years:
            used_loc_y = gp.quicksum(v for (kk, cc, yy), v in compost_loc_apply.items() if yy == y)
            prod_loc_y = gp.quicksum(v for (ff, yy), v in comp_prod.items() if yy == y)
            if compost_loc_apply:
                m.addConstr(used_loc_y <= prod_loc_y, name=f"compost_local_supply[{y}]")

        for f in facs:
            if (f, "MWAD") not in z:
                continue
            ad_z = z[f, "MWAD"]
            for key, rows_for_key in crop_by_key.items():
                if (key, f) not in dmap:
                    continue
                for r in rows_for_key:
                    crop_name = str(r.CROP).upper(); year = int(r.YEAR)
                    exprs = crop_expr[(key, crop_name, year)]
                    max_candidates = []
                    if n_conc > 0 and exprs["NREQ"] > 0:
                        max_candidates.append(exprs["NREQ"] / n_conc)
                    if p_conc > 0 and exprs["PREQ"] > 0:
                        max_candidates.append(exprs["PREQ"] / p_conc)
                    if k_conc > 0 and exprs["KREQ"] > 0:
                        max_candidates.append(exprs["KREQ"] / k_conc)
                    max_kg = max(max_candidates) if max_candidates else 0.0
                    if max_kg <= 0:
                        continue
                    x = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=max_kg, name=f"Xdig[{f},{key},{crop_name},{year}]")
                    dig_apply[(f, key, crop_name, year)] = x
                    m.addConstr(x <= max_kg * ad_z, name=f"xdig_selected_ad[{f},{key},{crop_name},{year}]")
                    exprs["N"] += n_conc * x
                    exprs["P"] += p_conc * x
                    exprs["K"] += k_conc * x
                    exprs["N_NO_IMPORT"] += n_conc * x
                    exprs["P_NO_IMPORT"] += p_conc * x
                    exprs["K_NO_IMPORT"] += k_conc * x

                    if target == "GWP":
                        d = float(dmap.get((key, f), 0.0))
                        obj += digest_material_gwp_coeff_absolute(r, d, n_conc, p_conc, k_conc, c_conc, ef) * x
                        city = int(exprs["CITY"])
                        mk = (city, crop_name, year)
                        mach_digest_expr[mk] = mach_digest_expr.get(mk, gp.LinExpr()) + x
                    elif target == "COST":
                        d = float(dmap.get((key, f), 0.0))
                        obj += 0.001 * d * cost_tkm_proxy * x


        fert_cap_multiplier = 1.0 + max(float(settings.fert_cap_tolerance), 0.0)

        nutrient_slack_penalty = max(float(settings.imported_compost_penalty) * 100.0, MIN_NUTRIENT_SLACK_PENALTY_RAW)
        for key_crop_year, exprs in crop_expr.items():
            key_s, crop_s, year_s = key_crop_year
            sn = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"SfertN[{key_s},{crop_s},{year_s}]")
            nutrient_slack[(key_s, crop_s, year_s, "N")] = sn
            obj += nutrient_slack_penalty * sn
            m.addConstr(exprs["N"] >= exprs["NREQ"], name=f"fert_n_req[{','.join(map(str,key_crop_year))}]")
            m.addConstr(exprs["N"] <= exprs["NREQ"] * fert_cap_multiplier + sn, name=f"fert_n_cap[{','.join(map(str,key_crop_year))}]")
            if exprs["PREQ"] > 0:
                sp = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"SfertP[{key_s},{crop_s},{year_s}]")
                nutrient_slack[(key_s, crop_s, year_s, "P")] = sp
                obj += nutrient_slack_penalty * sp
                m.addConstr(exprs["P"] >= exprs["PREQ"], name=f"fert_p_req[{','.join(map(str,key_crop_year))}]")
                m.addConstr(exprs["P"] <= exprs["PREQ"] * fert_cap_multiplier + sp, name=f"fert_p_cap[{','.join(map(str,key_crop_year))}]")
            if exprs["KREQ"] > 0:
                sk = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"SfertK[{key_s},{crop_s},{year_s}]")
                nutrient_slack[(key_s, crop_s, year_s, "K")] = sk
                obj += nutrient_slack_penalty * sk
                m.addConstr(exprs["K"] >= exprs["KREQ"], name=f"fert_k_req[{','.join(map(str,key_crop_year))}]")
                m.addConstr(exprs["K"] <= exprs["KREQ"] * fert_cap_multiplier + sk, name=f"fert_k_cap[{','.join(map(str,key_crop_year))}]")

            xci = compost_imp_apply.get(key_crop_year)
            if xci is not None:
                comp_n = float(exprs.get("COMP_N", 0.0))
                comp_p = float(exprs.get("COMP_P", 0.0))
                comp_k = float(exprs.get("COMP_K", 0.0))

                if comp_n > 0:
                    m.addConstr(exprs["N_NO_IMPORT"] + comp_n * xci <= exprs["NREQ"] * fert_cap_multiplier + nutrient_slack.get((key_crop_year[0], key_crop_year[1], key_crop_year[2], "N"), 0.0), name=f"comp_imp_residual_n[{','.join(map(str,key_crop_year))}]")
                if comp_p > 0 and exprs["PREQ"] > 0:
                    m.addConstr(exprs["P_NO_IMPORT"] + comp_p * xci <= exprs["PREQ"] * fert_cap_multiplier + nutrient_slack.get((key_crop_year[0], key_crop_year[1], key_crop_year[2], "P"), 0.0), name=f"comp_imp_residual_p[{','.join(map(str,key_crop_year))}]")
                if comp_k > 0 and exprs["KREQ"] > 0:
                    m.addConstr(exprs["K_NO_IMPORT"] + comp_k * xci <= exprs["KREQ"] * fert_cap_multiplier + nutrient_slack.get((key_crop_year[0], key_crop_year[1], key_crop_year[2], "K"), 0.0), name=f"comp_imp_residual_k[{','.join(map(str,key_crop_year))}]")

        if target == "GWP" and mach_digest_expr:
            for city_crop in sorted({(city, crop) for (city, crop, _) in mach_digest_expr.keys()}):
                city, crop_name = city_crop
                prev_req = None
                for y in sorted(years):
                    expr = mach_digest_expr.get((city, crop_name, y), gp.LinExpr())
                    req = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"Mreq[{city},{crop_name},{y}]")
                    newu = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"Mnew[{city},{crop_name},{y}]")
                    mach_req[(city, crop_name, y)] = req
                    mach_new[(city, crop_name, y)] = newu
                    m.addConstr(req >= expr * 0.001 / 2500.0, name=f"mach_req[{city},{crop_name},{y}]")
                    if prev_req is None:
                        m.addConstr(newu >= req, name=f"mach_new_first[{city},{crop_name},{y}]")
                    else:
                        m.addConstr(newu >= req - prev_req, name=f"mach_new_inc[{city},{crop_name},{y}]")
                    prev_req = req
                    obj += newu * ef_mach

        for f in facs:
            for y in years:
                supply_expr = gp.LinExpr()

                mwad_inputs = [v for (ff, typ, yy), v in ad_input.items() if ff == f and typ == "MWAD" and yy == y]
                mwad_vals = [rec for (typ, sc), rec in ad_coeff.items() if typ == "MWAD" and rec.get("CAPACITY_KG_YEAR", 0.0) > 0]
                dig_yield = float(np.mean([rec.get("DIGEST_KG", 0.0) / rec.get("CAPACITY_KG_YEAR", 1.0) for rec in mwad_vals] or [0.0]))
                if mwad_inputs and dig_yield > 0:
                    supply_expr += dig_yield * gp.quicksum(mwad_inputs)
                else:
                    for (ff, typ, sc), yv in scale_choice.items():
                        if ff == f and typ == "MWAD":
                            supply_expr += ad_coeff.get((typ, sc), {}).get("DIGEST_KG", 0.0) * yv
                applied_expr = gp.quicksum(v for (ff, _, _, yy), v in dig_apply.items() if ff == f and yy == y)
                u = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"Udig[{f},{y}]")
                dig_unused[(f, y)] = u
                m.addConstr(applied_expr + u == supply_expr, name=f"dig_supply_balance[{f},{y}]")
                if target == "GWP":
                    obj += u * 0.001 * ef_digww

    event_food_inc_slack_kg = gp.quicksum(event_food_inc_slack.values()) if event_food_inc_slack else gp.LinExpr(0.0)
    event_food_inc_slack_t = FOOD_INC_OBJ_SCALE_TO_TONNES * event_food_inc_slack_kg
    target_obj_scaled = target_objective_scale(target) * obj
    m.ModelSense = GRB.MINIMIZE
    food_inc_abstol_t = max(float(getattr(settings, "food_inc_abstol_t", DEFAULT_FOOD_INC_ABSTOL_T)), 0.0)
    m.setObjectiveN(
        event_food_inc_slack_t,
        index=0,
        priority=2,
        weight=1.0,
        abstol=food_inc_abstol_t,
        reltol=0.0,
        name="Minimize_post_event_food_INC_slack_t",
    )
    m.setObjectiveN(target_obj_scaled, index=1, priority=1, weight=1.0, name=f"Minimize_{target}_scaled")

    aux = {"z": z, "s": s_var, "scale_choice": scale_choice, "assign": assign,
           "assign_food": assign_food, "assign_manure": assign_manure, "assign_pw": assign_pw,
           "dig_apply": dig_apply, "syn_apply": syn_apply,
           "compost_imp_apply": compost_imp_apply, "compost_loc_apply": compost_loc_apply,
           "nutrient_slack": nutrient_slack, "dig_unused": dig_unused, "bio_use": bio_use, "assign_energy": assign_energy,
           "waste_inc": waste_inc, "food_inc": food_inc, "event_food_inc_slack": event_food_inc_slack,
           "manure_compbase": manure_compbase, "comp_prod": comp_prod,
           "ad_input": ad_input, "ad_digest_prod": ad_digest_prod,
           "years": years, "keys": keys, "facs": facs, "dist": dmap, "ad_coeff": ad_coeff}
    return m, aux, {}


def solve_flow_explicit_if_available(
    settings: OptimizeSettings,
    candidates: pd.DataFrame,
    price_y: pd.DataFrame,
    ef: pd.DataFrame,
    elec_ef: pd.DataFrame,
    ad_lookup: pd.DataFrame,
):
    model, aux = load_flow_model_cache(settings)
    if model is None or aux is None:
        tables = load_flow_parameter_tables(settings)
        if not has_flow_parameters(tables):
            return None, None
        log("Building k-f-y explicit allocation model")
        model, aux, _ = build_flow_explicit_model(settings, candidates, price_y, ef, elec_ef, ad_lookup, tables)
        if model is None:
            return None, None
        if settings.use_model_cache or settings.write_model_mps:
            save_flow_model_cache(model, aux, settings)
    log("Calling model.optimize() for k-f-y explicit allocation model")
    model.optimize()
    if model.SolCount <= 0:
        write_iis_if_infeasible(model, settings, "CE_AD_flow_explicit")
        log("Flow-explicit model returned no solution; falling back to facility-choice model.")
        return None, None

    cand = candidates.copy().set_index("FACILITY")
    selected_rows = []
    z = aux["z"]; s_var = aux["s"]; assign = aux["assign"]; dig_apply = aux.get("dig_apply", {})
    for f in aux["facs"]:
        selected_type = None
        for t in ["COMPOST", "MWAD", "TDAD", "INC", "RDF"]:
            v = z.get((f, t))
            if v is not None and v.X > 0.5:
                selected_type = t
                break
        if selected_type is None:
            continue
        scale = int(round(s_var[f].X)) if selected_type in {"MWAD", "TDAD"} else 0
        row = cand.loc[f].copy()
        row["FACILITY"] = f
        selected_rows.append(make_facility_output_row(row, selected_type, scale))
    facility_out = pd.DataFrame(selected_rows)

    bio_use = aux.get("bio_use", {})
    if not facility_out.empty and bio_use:
        for idx, rr in facility_out.iterrows():
            fac = str(rr.get("FACILITY", ""))
            typ = str(rr.get("TYPE", "")).upper()
            if typ in {"MWAD", "TDAD"}:
                if bio_use.get((fac, "GRIDGAS")) is not None and bio_use[(fac, "GRIDGAS")].X > 0.5:
                    facility_out.at[idx, "ENERGY_USE"] = "GRIDGAS"
                elif bio_use.get((fac, "ELEC")) is not None and bio_use[(fac, "ELEC")].X > 0.5:
                    facility_out.at[idx, "ENERGY_USE"] = "ELEC"

    link_rows = []
    for flow_name, flow_dict in [("FOOD", aux.get("assign_food", assign)), ("MANURE", aux.get("assign_manure", {})), ("PAPER_WOOD", aux.get("assign_pw", {}))]:
        for rec, v in flow_dict.items():
            if len(rec) == 4:
                k, f, t, y = rec
            else:
                k, f, t = rec
                y = "ALL"
            val = float(v.X)
            if val > 1e-6:
                link_rows.append({
                    "LINK_MODE": settings.link_mode,
                    "YEAR": y,
                    "ORIGIN_KEY": k,
                    "FACILITY": f,
                    "TYPE": t,
                    "FLOW": flow_name,
                    "SHARE": val,
                    "NOTE": "Material-explicit year-specific allocation from 05_optimize.py",
                })
    links = pd.DataFrame(link_rows)
    facility_out.attrs["links"] = links

    fallback_rows = []
    for (k, y), v in aux.get("waste_inc", {}).items():
        val = float(v.X)
        if val > 1e-6:
            fallback_rows.append({"YEAR": int(y), "KEY": k, "FLOW": "W_INC", "KG": val})
    for (k, y), v in aux.get("food_inc", {}).items():
        val = float(v.X)
        if val > 1e-6:
            fallback_rows.append({"YEAR": int(y), "KEY": k, "FLOW": "FW_INC", "KG": val})
    for (k, y), v in aux.get("manure_compbase", {}).items():
        val = float(v.X)
        if val > 1e-6:
            fallback_rows.append({"YEAR": int(y), "KEY": k, "FLOW": "M_COMPBASE", "KG": val})
    for (k, y), v in aux.get("event_food_inc_slack", {}).items():
        val = float(v.X)
        if val > 1e-6:
            fallback_rows.append({"YEAR": int(y), "KEY": k, "FLOW": "EVENT_FW_INC_SLACK", "KG": val})
    facility_out.attrs["fallback_flows"] = pd.DataFrame(fallback_rows)
    comp_rows = []
    for (f, y), v in aux.get("comp_prod", {}).items():
        val = float(v.X)
        if val > 1e-6:
            comp_rows.append({"YEAR": int(y), "FACILITY": f, "COMP_PROD_KG": val})
    facility_out.attrs["compost_supply"] = pd.DataFrame(comp_rows)

    digest_rows = []
    ad_coeff = aux.get("ad_coeff", {})

    vals = [v for (typ, sc), v in ad_coeff.items() if typ in {"MWAD", "TDAD"}]
    n_conc = float(np.mean([v["DIGEST_N_KG"] / v["DIGEST_KG"] for v in vals if v["DIGEST_KG"] > 0] or [0.0])) if vals else 0.0
    p_conc = float(np.mean([v["DIGEST_P2O5_KG"] / v["DIGEST_KG"] for v in vals if v["DIGEST_KG"] > 0] or [0.0])) if vals else 0.0
    k_conc = float(np.mean([v["DIGEST_K2O_KG"] / v["DIGEST_KG"] for v in vals if v["DIGEST_KG"] > 0] or [0.0])) if vals else 0.0
    c_conc = float(np.mean([v["DIGEST_C_KG"] / v["DIGEST_KG"] for v in vals if v["DIGEST_KG"] > 0] or [0.0])) if vals else 0.0
    dmap = aux.get("dist", {})
    crop_req = load_flow_parameter_tables(settings).get("crop_req", pd.DataFrame()) if settings.mode == "facility_agriculture" else pd.DataFrame()
    city_map = { }
    if not crop_req.empty:
        tmp = crop_req.copy(); tmp["KEY"] = tmp["KEY"].astype(str); tmp["YEAR"] = to_num(tmp["YEAR"], 0).astype(int); tmp["CROP"] = tmp["CROP"].astype(str).str.upper()
        for r in tmp.itertuples(index=False):
            city_map[(str(r.KEY), str(r.CROP).upper(), int(r.YEAR))] = int(to_num(getattr(r, "CITY", -1), -1))
    for (f, k, crop_name, y), v in dig_apply.items():
        val = float(v.X)
        if val > 1e-6:
            digest_rows.append({
                "YEAR": int(y), "FROM_FACILITY": f, "TO_KEY": k,
                "TO_CITY": city_map.get((str(k), str(crop_name).upper(), int(y)), -1),
                "CROP": str(crop_name).upper(), "CULTIVAR": "ALL",
                "DIGEST_KG": val, "DIGEST_N_KG": val * n_conc, "DIGEST_P_KG": val * p_conc,
                "DIGEST_K_KG": val * k_conc, "DIGEST_C_KG": val * c_conc,
                "DIGEST_KM": float(dmap.get((str(k), str(f)), 0.0)),
                "DIGEST_MACHINERY_UNITS": val * 0.001 / 2500.0,
            })
    digest_df = pd.DataFrame(digest_rows)
    facility_only_dig_fert = pd.DataFrame()
    if settings.mode == "facility_only_dig":
        digest_df, facility_only_dig_fert = build_facility_only_dig_links(settings, facility_out, aux)
    facility_out.attrs["digestate_links"] = digest_df

    slack_rows = []
    for (k, crop_name, y, nut), v in aux.get("nutrient_slack", {}).items():
        val = float(v.X)
        if val > 1e-7:
            slack_rows.append({
                "YEAR": int(y), "TO_KEY": k,
                "TO_CITY": city_map.get((str(k), str(crop_name).upper(), int(y)), -1),
                "CROP": str(crop_name).upper(), "CULTIVAR": "ALL",
                "NUTRIENT": str(nut).upper(), "SURPLUS_KG": val,
            })
    facility_out.attrs["nutrient_slacks"] = pd.DataFrame(slack_rows)

    fert_rows = []

    syn_apply = aux.get("syn_apply", {})
    comp_imp = aux.get("compost_imp_apply", {})
    comp_loc = aux.get("compost_loc_apply", {})
    for (k, crop_name, y, nut), v in syn_apply.items():
        val = float(v.X)
        if val > 1e-6:
            fert_rows.append({
                "YEAR": int(y), "TO_KEY": k,
                "TO_CITY": city_map.get((str(k), str(crop_name).upper(), int(y)), -1),
                "CROP": str(crop_name).upper(), "CULTIVAR": "ALL",
                "SOURCE": "SYN", "MATERIAL_KG": 0.0,
                "SYN_N_KG": val if nut == "N" else 0.0,
                "SYN_P_KG": val if nut == "P" else 0.0,
                "SYN_K_KG": val if nut == "K" else 0.0,
                "COMPOST_IMP_KG": 0.0, "COMPOST_LOC_KG": 0.0,
            })
    for (k, crop_name, y), v in comp_imp.items():
        val = float(v.X)
        if val > 1e-6:
            fert_rows.append({
                "YEAR": int(y), "TO_KEY": k,
                "TO_CITY": city_map.get((str(k), str(crop_name).upper(), int(y)), -1),
                "CROP": str(crop_name).upper(), "CULTIVAR": "ALL",
                "SOURCE": "COMPOST_IMP", "MATERIAL_KG": val,
                "SYN_N_KG": 0.0, "SYN_P_KG": 0.0, "SYN_K_KG": 0.0,
                "COMPOST_IMP_KG": val, "COMPOST_LOC_KG": 0.0,
            })
    for (k, crop_name, y), v in comp_loc.items():
        val = float(v.X)
        if val > 1e-6:
            fert_rows.append({
                "YEAR": int(y), "TO_KEY": k,
                "TO_CITY": city_map.get((str(k), str(crop_name).upper(), int(y)), -1),
                "CROP": str(crop_name).upper(), "CULTIVAR": "ALL",
                "SOURCE": "COMPOST_LOC", "MATERIAL_KG": val,
                "SYN_N_KG": 0.0, "SYN_P_KG": 0.0, "SYN_K_KG": 0.0,
                "COMPOST_IMP_KG": 0.0, "COMPOST_LOC_KG": val,
            })
    fert_df = pd.DataFrame(fert_rows)
    if settings.mode == "facility_only_dig" and isinstance(facility_only_dig_fert, pd.DataFrame) and not facility_only_dig_fert.empty:
        fert_df = facility_only_dig_fert

    facility_out.attrs["fertilizer_links"] = fert_df
    summary = {
        "TARGET": settings.target, "MODE": settings.mode, "ORG_MODE": settings.org_mode, "LINK_MODE": settings.link_mode,
        "COMPOST_MODE": settings.compost_mode,
        "ENABLE_GRIDGAS": settings.enable_gridgas, "SOLVER": "gurobi_flow_explicit", "STATUS": str(model.Status),
        "OBJECTIVE_ROLE": "lexicographic: priority 2 minimizes post-EVENT FW_INC slack with absolute tolerance; priority 1 minimizes scaled target (GWP ktCO2e or COST billion JPY)",
        "OBJECTIVE": float(model.ObjVal),
        "OBJECTIVE_TARGET_SCALE": float(target_objective_scale(settings.target)),
        "FOOD_INC_ABSTOL_T": float(getattr(settings, "food_inc_abstol_t", DEFAULT_FOOD_INC_ABSTOL_T)),
        "GUROBI_METHOD": int(getattr(settings, "gurobi_method", GUROBI_METHOD_AUTO)),
        "GUROBI_NODE_METHOD": int(getattr(settings, "gurobi_node_method", GUROBI_METHOD_AUTO)),
        "GUROBI_CROSSOVER": getattr(settings, "gurobi_crossover", None),
        "FW_INC_TOTAL_KG": float(sum(float(v.X) for v in aux.get("food_inc", {}).values())),
        "FW_INC_TOTAL_T": float(sum(float(v.X) for v in aux.get("food_inc", {}).values()) * FOOD_INC_OBJ_SCALE_TO_TONNES),
        "EVENT_FW_INC_SLACK_KG": float(sum(float(v.X) for v in aux.get("event_food_inc_slack", {}).values())),
        "EVENT_FW_INC_SLACK_T": float(sum(float(v.X) for v in aux.get("event_food_inc_slack", {}).values()) * FOOD_INC_OBJ_SCALE_TO_TONNES),
        "OBJ_BOUND": float(getattr(model, "ObjBound", np.nan)),
        "MIP_GAP_ACTUAL": float(getattr(model, "MIPGap", np.nan)),
        "RUNTIME_SEC": float(getattr(model, "Runtime", np.nan)),
        "NODE_COUNT": float(getattr(model, "NodeCount", np.nan)),
    }
    return facility_out, summary



def build_facility_only_dig_links(
    settings: OptimizeSettings,
    facility_out: pd.DataFrame,
    aux: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:

    cols_d = [
        "YEAR", "FROM_FACILITY", "TO_KEY", "TO_CITY", "CROP", "CULTIVAR",
        "DIGEST_KG", "DIGEST_N_KG", "DIGEST_P_KG", "DIGEST_K_KG", "DIGEST_C_KG",
        "DIGEST_KM", "DIGEST_MACHINERY_UNITS",
    ]
    cols_f = [
        "YEAR", "TO_KEY", "TO_CITY", "CROP", "CULTIVAR", "SOURCE", "MATERIAL_KG",
        "SYN_N_KG", "SYN_P_KG", "SYN_K_KG", "COMPOST_IMP_KG", "COMPOST_LOC_KG",
    ]
    if settings.mode != "facility_only_dig":
        return pd.DataFrame(columns=cols_d), pd.DataFrame(columns=cols_f)

    tables = load_flow_parameter_tables(settings)
    crop_req = tables.get("crop_req", pd.DataFrame()).copy()
    if crop_req.empty:
        log("facility_only_dig digestate allocation skipped: 05_param_crop_req.csv not found or empty")
        return pd.DataFrame(columns=cols_d), pd.DataFrame(columns=cols_f)

    ad_coeff = aux.get("ad_coeff", {})
    vals = [v for (typ, _sc), v in ad_coeff.items() if typ == "MWAD" and v.get("DIGEST_KG", 0.0) > 0]
    if not vals:
        return pd.DataFrame(columns=cols_d), pd.DataFrame(columns=cols_f)
    n_conc = float(np.mean([v.get("DIGEST_N_KG", 0.0) / v.get("DIGEST_KG", 1.0) for v in vals]))
    p_conc = float(np.mean([v.get("DIGEST_P2O5_KG", 0.0) / v.get("DIGEST_KG", 1.0) for v in vals]))
    k_conc = float(np.mean([v.get("DIGEST_K2O_KG", 0.0) / v.get("DIGEST_KG", 1.0) for v in vals]))
    c_conc = float(np.mean([v.get("DIGEST_C_KG", 0.0) / v.get("DIGEST_KG", 1.0) for v in vals]))
    if n_conc <= 0 and p_conc <= 0 and k_conc <= 0:
        return pd.DataFrame(columns=cols_d), pd.DataFrame(columns=cols_f)

    cr = crop_req.copy()
    for col in ["KEY", "CITY", "YEAR", "CROP", "NREQ_KG", "P2O5REQ_KG", "K2OREQ_KG"]:
        if col not in cr.columns:
            cr[col] = 0.0 if col not in {"KEY", "CROP"} else ""
    cr["KEY"] = cr["KEY"].astype(str)
    cr["CROP"] = cr["CROP"].astype(str).str.upper().str.strip()
    cr["YEAR"] = to_num(cr["YEAR"], 0).astype(int)
    cr["CITY"] = to_num(cr["CITY"], -1).astype(int)
    for col in ["NREQ_KG", "P2O5REQ_KG", "K2OREQ_KG"]:
        cr[col] = to_num(cr[col], 0.0).clip(lower=0.0)
    cr = cr.groupby(["KEY", "CITY", "YEAR", "CROP"], as_index=False)[["NREQ_KG", "P2O5REQ_KG", "K2OREQ_KG"]].sum()
    cr = cr[cr[["NREQ_KG", "P2O5REQ_KG", "K2OREQ_KG"]].sum(axis=1) > 1e-9].copy()
    if cr.empty:
        return pd.DataFrame(columns=cols_d), pd.DataFrame(columns=cols_f)


    mwad_yield = float(np.mean([v.get("DIGEST_KG", 0.0) / max(v.get("CAPACITY_KG_YEAR", 0.0), 1.0) for v in vals]))
    selected_mwad = set(facility_out.loc[facility_out["TYPE"].astype(str).str.upper().eq("MWAD"), "FACILITY"].astype(str))
    supply: dict[tuple[str, int], float] = {}
    for (f, typ, y), v in aux.get("ad_input", {}).items():
        if typ == "MWAD" and str(f) in selected_mwad:
            val = float(v.X) * mwad_yield
            if val > 1e-6:
                supply[(str(f), int(y))] = supply.get((str(f), int(y)), 0.0) + val
    if not supply:
        for (f, typ, sc), yv in aux.get("scale_choice", {}).items():
            if typ != "MWAD" or str(f) not in selected_mwad or float(yv.X) <= 0.5:
                continue
            prod = ad_coeff.get((typ, int(sc)), {}).get("DIGEST_KG", 0.0)
            for y in aux.get("years", YEARS):
                supply[(str(f), int(y))] = supply.get((str(f), int(y)), 0.0) + float(prod)
    if not supply:
        return pd.DataFrame(columns=cols_d), pd.DataFrame(columns=cols_f)

    dmap = aux.get("dist", {})
    digest_rows: list[dict[str, Any]] = []
    fert_rows: list[dict[str, Any]] = []
    for (fac, year), rem_supply0 in sorted(supply.items(), key=lambda x: (x[0][1], x[0][0])):
        rem_supply = float(rem_supply0)
        if rem_supply <= 1e-6:
            continue
        cand_rows = cr[cr["YEAR"].eq(int(year))].copy()
        cand_rows["_DIST"] = cand_rows["KEY"].map(lambda k: float(dmap.get((str(k), str(fac)), np.inf)))
        cand_rows = cand_rows[np.isfinite(cand_rows["_DIST"])].sort_values(["_DIST", "KEY", "CROP"])
        for idx, r in cand_rows.iterrows():
            if rem_supply <= 1e-6:
                break
            reqs = []
            if n_conc > 0 and float(r["NREQ_KG"]) > 0:
                reqs.append(float(r["NREQ_KG"]) / n_conc)
            if p_conc > 0 and float(r["P2O5REQ_KG"]) > 0:
                reqs.append(float(r["P2O5REQ_KG"]) / p_conc)
            if k_conc > 0 and float(r["K2OREQ_KG"]) > 0:
                reqs.append(float(r["K2OREQ_KG"]) / k_conc)
            need_kg = max(reqs) if reqs else 0.0
            take = min(rem_supply, need_kg)
            if take <= 1e-6:
                continue
            rem_supply -= take
            dkg_n = take * n_conc; dkg_p = take * p_conc; dkg_k = take * k_conc
            key = str(r["KEY"]); crop_name = str(r["CROP"]).upper(); city = int(r["CITY"])
            digest_rows.append({
                "YEAR": int(year), "FROM_FACILITY": fac, "TO_KEY": key, "TO_CITY": city,
                "CROP": crop_name, "CULTIVAR": "ALL", "DIGEST_KG": take,
                "DIGEST_N_KG": dkg_n, "DIGEST_P_KG": dkg_p, "DIGEST_K_KG": dkg_k,
                "DIGEST_C_KG": take * c_conc, "DIGEST_KM": float(r["_DIST"]),
                "DIGEST_MACHINERY_UNITS": take * 0.001 / 2500.0,
            })

            fert_rows.append({
                "YEAR": int(year), "TO_KEY": key, "TO_CITY": city, "CROP": crop_name, "CULTIVAR": "ALL",
                "SOURCE": "SYN_AFTER_DIGEST", "MATERIAL_KG": 0.0,
                "SYN_N_KG": max(float(r["NREQ_KG"]) - dkg_n, 0.0),
                "SYN_P_KG": max(float(r["P2O5REQ_KG"]) - dkg_p, 0.0),
                "SYN_K_KG": max(float(r["K2OREQ_KG"]) - dkg_k, 0.0),
                "COMPOST_IMP_KG": 0.0, "COMPOST_LOC_KG": 0.0,
            })
    return pd.DataFrame(digest_rows, columns=cols_d), pd.DataFrame(fert_rows, columns=cols_f)


def solve_or_heuristic(
    settings: OptimizeSettings,
    candidates: pd.DataFrame,
    loads: pd.DataFrame,
    price_y: pd.DataFrame,
    ef: pd.DataFrame,
    elec_ef: pd.DataFrame,
    ad_lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    flow_facility, flow_summary = solve_flow_explicit_if_available(settings, candidates, price_y, ef, elec_ef, ad_lookup)
    if flow_facility is not None and flow_summary is not None:
        return flow_facility, flow_summary

    model, option_vars, aux = build_facility_choice_model(settings, candidates, loads, price_y, ef, elec_ef, ad_lookup)
    summary: dict[str, Any] = {
        "TARGET": settings.target,
        "MODE": settings.mode,
        "ORG_MODE": settings.org_mode,
        "LINK_MODE": settings.link_mode,
        "COMPOST_MODE": settings.compost_mode,
        "ENABLE_GRIDGAS": settings.enable_gridgas,
        "SOLVER": "gurobi" if gp is not None else "heuristic",
        "STATUS": "",
        "OBJECTIVE": np.nan,
    }
    selected_rows = []
    cand = candidates.set_index("FACILITY")

    if model is not None:
        model.optimize()
        summary["STATUS"] = str(model.Status)
        if model.SolCount <= 0:
            write_iis_if_infeasible(model, settings, "CE_AD_facility_choice")
        if model.SolCount > 0:
            summary["OBJECTIVE"] = float(model.ObjVal)
            summary["OBJ_BOUND"] = float(getattr(model, "ObjBound", np.nan))
            summary["MIP_GAP_ACTUAL"] = float(getattr(model, "MIPGap", np.nan))
            summary["RUNTIME_SEC"] = float(getattr(model, "Runtime", np.nan))
            summary["NODE_COUNT"] = float(getattr(model, "NodeCount", np.nan))
            for (fac, typ, scale), v in option_vars.items():
                if v.X > 0.5:
                    row = cand.loc[fac].copy()
                    row["FACILITY"] = fac
                    selected_rows.append(make_facility_output_row(row, typ, scale))
        else:
            log("Gurobi returned no solution; falling back to baseline heuristic.")

    if not selected_rows:
        summary["SOLVER"] = "heuristic"
        summary["STATUS"] = "baseline_default"
        max_input = loads.groupby("FACILITY")["INPUT_KG"].max().to_dict() if not loads.empty else {}
        for _, row in candidates.iterrows():
            base_type = norm_text(row.get("BASE_TYPE", ""))
            locked_tdad_scale = existing_tdad_fixed_scale(row)
            if locked_tdad_scale is not None:
                typ, scale = "TDAD", locked_tdad_scale
            elif compost_candidates_enabled(settings) and row.get("ALLOW_COMPOST", 0) == 1:
                typ, scale = "COMPOST", 0
            elif base_type in {"INC", "RDF"}:
                typ, scale = base_type, 0
            elif row.get("ALLOW_MWAD", 0) == 1:
                load = float(max_input.get(row["FACILITY"], 0.0))
                scale = max(settings.min_scale, int(math.ceil(load / (1000.0 * 365.0)))) if load > 0 else settings.min_scale
                scale = min(scale, settings.max_scale)
                typ = "MWAD"
            elif row.get("ALLOW_TDAD", 0) == 1:
                load = float(max_input.get(row["FACILITY"], 0.0))
                scale = max(settings.min_scale, int(math.ceil(load / (1000.0 * 365.0)))) if load > 0 else settings.min_scale
                scale = min(scale, settings.max_scale)
                typ = "TDAD"
            else:
                typ, scale = (base_type if base_type not in {"COMPOST"} else "INC") or "INC", 0
            selected_rows.append(make_facility_output_row(row, typ, scale))

    out = pd.DataFrame(selected_rows)
    return out, summary




def preserve_protected_dual_type_facilities(settings: OptimizeSettings, facility_out: pd.DataFrame) -> pd.DataFrame:

    if not compost_candidates_enabled(settings):
        protected = {"NANTAN_YAGI": {"MWAD"}}
    else:
        protected = {"NANTAN_YAGI": {"COMPOST", "MWAD"}}
    manure_path = settings.dataset_dir / "00_manurefac.csv"
    if not manure_path.exists():
        return facility_out
    try:
        manure = pd.read_csv(manure_path)
    except Exception:
        return facility_out
    if manure.empty or "FACILITY" not in manure.columns or "TYPE" not in manure.columns:
        return facility_out

    out = facility_out.copy()
    out["FACILITY"] = out["FACILITY"].astype(str)
    out["TYPE"] = out["TYPE"].astype(str).str.upper().str.strip()
    manure = manure.copy()
    manure["FACILITY"] = manure["FACILITY"].astype(str)
    manure["TYPE"] = manure["TYPE"].astype(str).str.upper().str.strip()

    add_rows: list[dict[str, Any]] = []
    for fac, required_types in protected.items():
        src_fac = manure[manure["FACILITY"].eq(fac)].copy()
        if src_fac.empty:
            continue
        existing_types = set(out.loc[out["FACILITY"].eq(fac), "TYPE"].astype(str).str.upper())
        for typ in sorted(required_types):
            if typ in existing_types:
                continue
            g = src_fac[src_fac["TYPE"].eq(typ)].copy()
            if g.empty:
                continue
            scale_out: Any = "n/a"
            if typ in {"MWAD", "TDAD"}:
                cap = pd.to_numeric(g.get("CAPACITY", pd.Series([0.0])), errors="coerce").fillna(0.0).sum()
                scale_out = max(1, int(math.ceil(float(cap) / 365000.0))) if cap > 0 else settings.min_scale
                scale_out = min(int(scale_out), settings.max_scale)
            built_year = 2030
            if "CONST_YEAR" in g.columns:
                vals = pd.to_numeric(g["CONST_YEAR"], errors="coerce").dropna()
                if not vals.empty:
                    built_year = int(max(vals.min(), 2030))
            add_rows.append({
                "FACILITY": fac,
                "LAT": pd.to_numeric(g.get("LAT", pd.Series([np.nan])).dropna().iloc[0] if g.get("LAT", pd.Series(dtype=float)).notna().any() else np.nan, errors="coerce"),
                "LON": pd.to_numeric(g.get("LON", pd.Series([np.nan])).dropna().iloc[0] if g.get("LON", pd.Series(dtype=float)).notna().any() else np.nan, errors="coerce"),
                "CITY": pd.to_numeric(g.get("CITY", pd.Series([-1])).dropna().iloc[0] if g.get("CITY", pd.Series(dtype=float)).notna().any() else -1, errors="coerce"),
                "SOURCE": "MANUREFAC_PROTECTED_DUPLICATE",
                "BASE_TYPE": typ,
                "TYPE": typ,
                "SCALE_T_DAY": scale_out,
                "BUILT_YEAR": built_year,
                "ENERGY_USE": "GRIDGAS" if typ in {"MWAD", "TDAD"} and settings.enable_gridgas else ("ELEC" if typ in {"MWAD", "TDAD"} else "n/a"),
                "IS_INC_SITE": 0,
            })
    if add_rows:
        out = pd.concat([out, pd.DataFrame(add_rows)], ignore_index=True)
        log("Preserved protected dual-type facility rows: " + ", ".join(f"{r['FACILITY']}:{r['TYPE']}" for r in add_rows))
    return out

def make_facility_output_row(row: pd.Series, typ: str, scale: int) -> dict[str, Any]:
    if typ == "COMPOST":
        scale_out: Any = "n/a"
    elif typ in {"MWAD", "TDAD"}:
        scale_out = int(scale)
    else:
        scale_out = "n/a"
    built_year = int(to_num(row.get("BUILT_YEAR_BASE", 2030), 2030))
    if typ in {"MWAD", "TDAD"} and built_year < 2030:
        built_year = 2030
    return {
        "FACILITY": row.get("FACILITY", ""),
        "LAT": row.get("LAT", np.nan),
        "LON": row.get("LON", np.nan),
        "CITY": row.get("CITY", -1),
        "SOURCE": row.get("SOURCE", ""),
        "BASE_TYPE": row.get("BASE_TYPE", ""),
        "TYPE": typ,
        "SCALE_T_DAY": scale_out,
        "BUILT_YEAR": built_year,
        "IS_INC_SITE": int(to_num(row.get("IS_INC_SITE", 0), 0)),
        "ENERGY_USE": "GRIDGAS" if typ in {"MWAD", "TDAD"} and False else ("ELEC" if typ in {"MWAD", "TDAD", "INC", "RDF"} else "n/a"),
    }



def write_digestate_links(settings: OptimizeSettings, facility_out: pd.DataFrame) -> pd.DataFrame:

    explicit = facility_out.attrs.get("digestate_links") if hasattr(facility_out, "attrs") else None
    cols = [
        "YEAR", "FROM_FACILITY", "TO_KEY", "TO_CITY", "CROP", "CULTIVAR",
        "DIGEST_KG", "DIGEST_N_KG", "DIGEST_P_KG", "DIGEST_K_KG", "DIGEST_C_KG",
        "DIGEST_KM", "DIGEST_MACHINERY_UNITS",
    ]
    if isinstance(explicit, pd.DataFrame) and not explicit.empty:
        out = explicit.copy()
    else:
        out = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0 if c.startswith("DIGEST") or c in {"YEAR", "TO_CITY"} else ""
    return out[cols]


def write_fertilizer_links(settings: OptimizeSettings, facility_out: pd.DataFrame) -> pd.DataFrame:

    explicit = facility_out.attrs.get("fertilizer_links") if hasattr(facility_out, "attrs") else None
    cols = [
        "YEAR", "TO_KEY", "TO_CITY", "CROP", "CULTIVAR", "SOURCE", "MATERIAL_KG",
        "SYN_N_KG", "SYN_P_KG", "SYN_K_KG", "COMPOST_IMP_KG", "COMPOST_LOC_KG",
    ]
    if isinstance(explicit, pd.DataFrame) and not explicit.empty:
        out = explicit.copy()
    else:
        out = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0 if c not in {"TO_KEY", "CROP", "CULTIVAR", "SOURCE"} else ""
    if settings.mode == "facility_only_dig":
        out["COMPOST_IMP_KG"] = 0.0
    return out[cols]


def write_fallback_flows(settings: OptimizeSettings, facility_out: pd.DataFrame) -> pd.DataFrame:

    explicit = facility_out.attrs.get("fallback_flows") if hasattr(facility_out, "attrs") else None
    cols = ["YEAR", "KEY", "FLOW", "KG"]
    if isinstance(explicit, pd.DataFrame) and not explicit.empty:
        out = explicit.copy()
    else:
        out = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0 if c in {"YEAR", "KG"} else ""
    return out[cols]


def write_compost_supply(settings: OptimizeSettings, facility_out: pd.DataFrame) -> pd.DataFrame:

    explicit = facility_out.attrs.get("compost_supply") if hasattr(facility_out, "attrs") else None
    cols = ["YEAR", "FACILITY", "COMP_PROD_KG"]
    if isinstance(explicit, pd.DataFrame) and not explicit.empty:
        out = explicit.copy()
    else:
        out = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0 if c in {"YEAR", "COMP_PROD_KG"} else ""
    return out[cols]


def write_nutrient_slacks(settings: OptimizeSettings, facility_out: pd.DataFrame) -> pd.DataFrame:

    explicit = facility_out.attrs.get("nutrient_slacks") if hasattr(facility_out, "attrs") else None
    cols = ["YEAR", "TO_KEY", "TO_CITY", "CROP", "CULTIVAR", "NUTRIENT", "SURPLUS_KG"]
    if isinstance(explicit, pd.DataFrame) and not explicit.empty:
        out = explicit.copy()
    else:
        out = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0 if c in {"YEAR", "TO_CITY", "SURPLUS_KG"} else ""
    return out[cols]


def write_link_hints(settings: OptimizeSettings, facility_out: pd.DataFrame, loads: pd.DataFrame) -> pd.DataFrame:

    explicit_links = facility_out.attrs.get("links") if hasattr(facility_out, "attrs") else None
    if isinstance(explicit_links, pd.DataFrame) and not explicit_links.empty:
        return explicit_links
    rows = []
    selected = set(facility_out.loc[facility_out["TYPE"].isin(["COMPOST", "MWAD", "TDAD"]), "FACILITY"].astype(str))
    for fac in sorted(selected):
        rows.append({
            "LINK_MODE": settings.link_mode,
            "ORIGIN_KEY": "*",
            "FACILITY": fac,
            "FLOW": "REESTIMATE_IN_01_INVENTORY",
            "SHARE": np.nan,
            "NOTE": "Community-level assignment should be re-estimated using selected 06_facility.csv.",
        })
    return pd.DataFrame(rows)


def parse_args() -> OptimizeSettings:
    ap = argparse.ArgumentParser(description="Optimize facility choices and write 06_facility.csv.")
    ap.add_argument("--target", choices=["GWP", "COST"], required=True)
    ap.add_argument("--mode", choices=["facility_only", "facility_only_dig", "facility_agriculture"], default="facility_only")
    ap.add_argument("--org-mode", choices=["baseline", "free"], default="baseline")
    ap.add_argument("--link-mode", choices=["single", "multi"], default="single")
    ap.add_argument("--compost-mode", choices=["include", "exclude"], default="include",
                    help="Whether TYPE=COMPOST is available as a biowaste-treatment candidate.")
    ap.add_argument("--enable-gridgas", action="store_true")
    ap.add_argument("--min-scale", type=int, default=1, help="Minimum AD scale in t/day.")
    ap.add_argument("--max-scale", type=int, default=100, help="Maximum AD scale in t/day.")
    ap.add_argument("--scale-step", type=int, default=1, help="AD scale interval in t/day.")
    ap.add_argument(
        "--max-digestate-facilities-per-key",
        type=int,
        default=5,
        help=(
            "Keep only nearest N facilities per KEY inside 05_optimize.py as a second guard. "
            "01_inventory.py also controls this upstream via --max-param-facilities-per-key; 0 means no limit."
        ),
    )
    ap.add_argument("--fert-cap-tolerance", type=float, default=0.001, help="Relative upper tolerance for optimized N/P/K crop nutrient caps. Default 0.001 = 0.1%%.")
    ap.add_argument("--imported-compost-penalty", type=float, default=1.0e4, help="Objective penalty per kg imported compost before target-unit scaling; large enough to prioritize pooled local compost without creating 1e9-scale coefficients.")
    ap.add_argument("--food-inc-abstol-t", type=float, default=DEFAULT_FOOD_INC_ABSTOL_T, help="Absolute tolerance, in tonnes, allowed on the first lexicographic objective when optimizing the main objective. Default 1.0 tonne.")
    ap.add_argument("--gurobi-method", type=int, default=DEFAULT_GUROBI_METHOD, choices=[-1, 0, 1, 2, 3, 4, 5], help="Gurobi Method for root relaxation. Default 1 = dual simplex, matching previous versions.")
    ap.add_argument("--gurobi-node-method", type=int, default=DEFAULT_GUROBI_NODE_METHOD, choices=[-1, 0, 1, 2], help="Gurobi NodeMethod. Default 1 = dual simplex for node relaxations.")
    ap.add_argument("--gurobi-crossover", type=int, default=None, choices=[0, 1, 2, 3, 4], help="Optional Gurobi Crossover setting. Use 0 with --gurobi-method 2 to avoid barrier crossover.")
    ap.add_argument("--numeric-focus", type=int, default=2, choices=[0, 1, 2, 3], help="Gurobi NumericFocus. Default 2 for the scaled lexicographic model.")
    ap.add_argument("--mip-focus", type=int, default=None, choices=[0, 1, 2, 3], help="Optional Gurobi MIPFocus.")
    ap.add_argument("--heuristics", type=float, default=None, help="Optional Gurobi Heuristics parameter, e.g. 0.2.")
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--mip-gap", type=float, default=None)
    ap.add_argument("--dataset-dir", default=os.getenv("DATASET_DIR", "."))
    ap.add_argument("--input-dir", default=os.getenv("INPUT_DIR", "."))
    ap.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "."))
    ap.add_argument("--script-dir", default=None)
    ap.add_argument("--threads", type=int, default=0, help="Gurobi Threads. 0 = automatic/max available.")
    ap.add_argument("--log-dir", default=None, help="Directory for Gurobi logs. Default: <output-dir>/logs.")
    ap.add_argument("--cache-dir", default=None, help="Directory for cached flow-explicit model files. Default: <output-dir>/cache.")
    ap.add_argument("--use-model-cache", action="store_true", help="Enable flow-explicit model cache. Disabled by default to avoid long MPS/JSON cache I/O before optimization.")
    ap.add_argument("--no-model-cache", action="store_true", help="Disable flow-explicit model cache. Kept for backward-compatible pipeline calls.")
    ap.add_argument("--rebuild-model-cache", action="store_true", help="Ignore existing model cache and rebuild it.")
    ap.add_argument("--write-model-mps", action="store_true", help="Always export the built flow-explicit model as MPS/JSON metadata.")

    args = ap.parse_args()
    dataset_dir = Path(args.dataset_dir)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    script_dir = Path(args.script_dir) if args.script_dir else Path(__file__).resolve().parent
    log_dir = Path(args.log_dir) if args.log_dir else output_dir / "logs"
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"
    if args.min_scale < 1:
        raise ValueError("--min-scale must be >= 1")
    if args.max_scale < args.min_scale:
        raise ValueError("--max-scale must be >= --min-scale")
    if args.scale_step < 1:
        raise ValueError("--scale-step must be >= 1")
    if args.max_digestate_facilities_per_key < 0:
        raise ValueError("--max-digestate-facilities-per-key must be >= 0")
    if args.fert_cap_tolerance < 0:
        raise ValueError("--fert-cap-tolerance must be >= 0")
    if args.imported_compost_penalty < 0:
        raise ValueError("--imported-compost-penalty must be >= 0")
    if args.food_inc_abstol_t < 0:
        raise ValueError("--food-inc-abstol-t must be >= 0")
    return OptimizeSettings(
        target=args.target.upper(), mode=args.mode, org_mode=args.org_mode,
        link_mode=args.link_mode, compost_mode=args.compost_mode, enable_gridgas=bool(args.enable_gridgas),
        min_scale=int(args.min_scale), max_scale=int(args.max_scale), scale_step=int(args.scale_step),
        max_digestate_facilities_per_key=int(args.max_digestate_facilities_per_key),
        fert_cap_tolerance=float(args.fert_cap_tolerance),
        imported_compost_penalty=float(args.imported_compost_penalty),
        food_inc_abstol_t=float(args.food_inc_abstol_t),
        gurobi_method=int(args.gurobi_method), gurobi_node_method=int(args.gurobi_node_method),
        gurobi_crossover=args.gurobi_crossover,
        numeric_focus=args.numeric_focus, mip_focus=args.mip_focus, heuristics=args.heuristics,
        time_limit=args.time_limit, mip_gap=args.mip_gap, threads=int(args.threads),
        log_dir=log_dir, cache_dir=cache_dir, use_model_cache=(bool(args.use_model_cache) and not args.no_model_cache),
        rebuild_model_cache=bool(args.rebuild_model_cache), write_model_mps=bool(args.write_model_mps),
        dataset_dir=dataset_dir, input_dir=input_dir, output_dir=output_dir, script_dir=script_dir,
    )




def apply_msw_event_years_to_additional_facilities(settings: OptimizeSettings, facility_out: pd.DataFrame) -> pd.DataFrame:

    out = facility_out.copy()
    msw_path = first_existing([settings.input_dir / "00_mswfac.csv", settings.dataset_dir / "00_mswfac.csv"])
    if msw_path is None or out.empty:
        return out
    try:
        msw = pd.read_csv(msw_path)
    except Exception:
        return out
    base_map, city_map, boundary_map = build_mwad_year_lookup_from_mswfac(msw)
    if not (base_map or city_map or boundary_map):
        return out
    out["TYPE"] = out["TYPE"].astype(str).str.upper().str.strip()
    mwad_mask = out["TYPE"].eq("MWAD")
    for idx in out.index[mwad_mask]:
        year = mwad_year_for_facility_row(out.loc[idx], base_map, city_map, boundary_map)
        if year is not None:
            out.at[idx, "BUILT_YEAR"] = int(year)
    return out

def main() -> None:

    
    settings = parse_args()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    log(f"Settings: target={settings.target}, mode={settings.mode}, link_mode={settings.link_mode}, "
        f"compost_mode={settings.compost_mode}, "
        f"max_digestate_facilities_per_key={settings.max_digestate_facilities_per_key}, "
        f"fert_cap_tolerance={settings.fert_cap_tolerance}, imported_compost_penalty={settings.imported_compost_penalty}, "
        f"food_inc_abstol_t={settings.food_inc_abstol_t}, gurobi_method={settings.gurobi_method}, "
        f"gurobi_node_method={settings.gurobi_node_method}, gurobi_crossover={settings.gurobi_crossover}, "
        f"numeric_focus={settings.numeric_focus}, mip_focus={settings.mip_focus}, heuristics={settings.heuristics}, "
        f"time_limit={settings.time_limit}, mip_gap={settings.mip_gap}, threads={settings.threads}, "
        f"model_cache={settings.use_model_cache}, cache_dir={settings.cache_dir}")
    log(f"Loading candidates from {settings.dataset_dir}")
    candidates = build_candidate_facilities(settings)
    waste = load_waste_activity(settings)
    loads = facility_baseline_loads(candidates, waste)

    ef = read_csv_if_exists(settings.dataset_dir / "02_EF.csv")
    energy = read_csv_if_exists(settings.dataset_dir / "02_energy.csv")
    price_y = build_price_table(settings.dataset_dir / "00_cost.csv")
    elec_ef = build_elec_ef(energy, ef)

    log("Generating AD lookup values from MWAD/TDAD modules where available")
    ad_lookup = estimate_ad_lookup(settings, candidates)
    ad_lookup.to_csv(settings.output_dir / "06_ad_lookup.csv", index=False)

    log("Solving facility-choice model")
    facility_out, summary = solve_or_heuristic(settings, candidates, loads, price_y, ef, elec_ef, ad_lookup)
    # ENERGY_USE for AD is set from optimized Ubio in the flow model.
    # Fallback facility-choice model still uses the scenario switch.
    if "ENERGY_USE" not in facility_out.columns or facility_out["ENERGY_USE"].isna().all():
        if settings.enable_gridgas:
            facility_out.loc[facility_out["TYPE"].isin(["MWAD", "TDAD"]), "ENERGY_USE"] = "GRIDGAS"
        else:
            facility_out.loc[facility_out["TYPE"].isin(["MWAD", "TDAD"]), "ENERGY_USE"] = "ELEC"

    facility_out = preserve_protected_dual_type_facilities(settings, facility_out)
    facility_out = apply_msw_event_years_to_additional_facilities(settings, facility_out)

    # Stable output schema requested by user, with explicit type-change diagnostics.
    # TYPE is kept as the downstream-compatible optimized/selected type.
    if "BASE_TYPE" not in facility_out.columns:
        facility_out["BASE_TYPE"] = ""
    facility_out["BASE_TYPE"] = facility_out["BASE_TYPE"].fillna("").astype(str).str.upper().str.strip()
    facility_out["TYPE"] = facility_out["TYPE"].fillna("").astype(str).str.upper().str.strip()
    facility_out["ORIGINAL_TYPE"] = facility_out["BASE_TYPE"]
    facility_out["OPTIMIZED_TYPE"] = facility_out["TYPE"]
    facility_out["TYPE_CHANGED"] = np.where(
        (facility_out["ORIGINAL_TYPE"] != "") & (facility_out["ORIGINAL_TYPE"] != facility_out["OPTIMIZED_TYPE"]),
        1,
        0,
    )
    facility_out["TYPE_CHANGE"] = np.where(
        facility_out["TYPE_CHANGED"].eq(1),
        facility_out["ORIGINAL_TYPE"] + "->" + facility_out["OPTIMIZED_TYPE"],
        facility_out["ORIGINAL_TYPE"] + "->" + facility_out["OPTIMIZED_TYPE"],
    )

    desired_cols = [
        "FACILITY", "LAT", "LON", "CITY", "SOURCE",
        "BASE_TYPE", "ORIGINAL_TYPE", "TYPE", "OPTIMIZED_TYPE", "TYPE_CHANGED", "TYPE_CHANGE",
        "SCALE_T_DAY", "BUILT_YEAR", "ENERGY_USE", "IS_INC_SITE",
    ]
    for c in desired_cols:
        if c not in facility_out.columns:
            facility_out[c] = np.nan
    facility_out = facility_out[desired_cols].sort_values(["FACILITY", "TYPE"]).reset_index(drop=True)
    facility_out.to_csv(settings.output_dir / "06_facility.csv", index=False)

    links = write_link_hints(settings, facility_out, loads)
    links.to_csv(settings.output_dir / "06_links.csv", index=False)

    digestate_links = write_digestate_links(settings, facility_out)
    digestate_links.to_csv(settings.output_dir / "06_digestate_links.csv", index=False)

    fertilizer_links = write_fertilizer_links(settings, facility_out)
    fertilizer_links.to_csv(settings.output_dir / "06_fertilizer_links.csv", index=False)

    fallback_flows = write_fallback_flows(settings, facility_out)
    fallback_flows.to_csv(settings.output_dir / "06_fallback_flows.csv", index=False)
    compost_supply = write_compost_supply(settings, facility_out)
    compost_supply.to_csv(settings.output_dir / "06_compost_supply.csv", index=False)
    nutrient_slacks = write_nutrient_slacks(settings, facility_out)
    nutrient_slacks.to_csv(settings.output_dir / "06_nutrient_slacks.csv", index=False)

    summary_df = pd.DataFrame([summary])
    summary_df["TIME_LIMIT_SEC"] = settings.time_limit
    summary_df["MIP_GAP_ALLOWED"] = settings.mip_gap
    summary_df["THREADS"] = settings.threads
    summary_df["NUMERIC_FOCUS"] = settings.numeric_focus
    summary_df["MIP_FOCUS"] = settings.mip_focus
    summary_df["HEURISTICS"] = settings.heuristics
    summary_df["FERT_CAP_TOLERANCE"] = settings.fert_cap_tolerance
    summary_df["IMPORTED_COMPOST_PENALTY"] = settings.imported_compost_penalty
    summary_df["FOOD_INC_ABSTOL_T"] = settings.food_inc_abstol_t
    summary_df["GUROBI_METHOD"] = settings.gurobi_method
    summary_df["GUROBI_NODE_METHOD"] = settings.gurobi_node_method
    summary_df["GUROBI_CROSSOVER"] = settings.gurobi_crossover
    summary_df["MODEL_CACHE"] = settings.use_model_cache
    summary_df["COMPOST_MODE"] = settings.compost_mode
    summary_df["N_FACILITIES"] = len(facility_out)
    summary_df["N_SELECTED_AD"] = int(facility_out["TYPE"].isin(["MWAD", "TDAD"]).sum())
    summary_df["N_SELECTED_COMPOST"] = int(facility_out["TYPE"].eq("COMPOST").sum())
    summary_df["N_DIGESTATE_LINKS"] = len(digestate_links)
    summary_df["N_FERTILIZER_LINKS"] = len(fertilizer_links)
    summary_df["N_FALLBACK_FLOW_ROWS"] = len(fallback_flows)
    summary_df["N_COMPOST_SUPPLY_ROWS"] = len(compost_supply)
    summary_df["N_NUTRIENT_SLACK_ROWS"] = len(nutrient_slacks)
    summary_df["NUTRIENT_SLACK_KG"] = float(pd.to_numeric(nutrient_slacks.get("SURPLUS_KG", 0.0), errors="coerce").fillna(0.0).sum()) if not nutrient_slacks.empty else 0.0
    summary_df["DIGESTATE_APPLIED_KG"] = float(pd.to_numeric(digestate_links.get("DIGEST_KG", 0.0), errors="coerce").fillna(0.0).sum()) if not digestate_links.empty else 0.0
    summary_df["SYN_N_APPLIED_KG"] = float(pd.to_numeric(fertilizer_links.get("SYN_N_KG", 0.0), errors="coerce").fillna(0.0).sum()) if not fertilizer_links.empty else 0.0
    summary_df["SYN_P_APPLIED_KG"] = float(pd.to_numeric(fertilizer_links.get("SYN_P_KG", 0.0), errors="coerce").fillna(0.0).sum()) if not fertilizer_links.empty else 0.0
    summary_df["SYN_K_APPLIED_KG"] = float(pd.to_numeric(fertilizer_links.get("SYN_K_KG", 0.0), errors="coerce").fillna(0.0).sum()) if not fertilizer_links.empty else 0.0
    summary_df["COMPOST_IMP_APPLIED_KG"] = float(pd.to_numeric(fertilizer_links.get("COMPOST_IMP_KG", 0.0), errors="coerce").fillna(0.0).sum()) if not fertilizer_links.empty else 0.0
    summary_df["COMPOST_LOC_APPLIED_KG"] = float(pd.to_numeric(fertilizer_links.get("COMPOST_LOC_KG", 0.0), errors="coerce").fillna(0.0).sum()) if not fertilizer_links.empty else 0.0
    summary_df.to_csv(settings.output_dir / "06_solution_summary.csv", index=False)

    log(f"Wrote {settings.output_dir / '06_facility.csv'}")
    log(f"Wrote {settings.output_dir / '06_links.csv'}")
    log(f"Wrote {settings.output_dir / '06_digestate_links.csv'}")
    log(f"Wrote {settings.output_dir / '06_fertilizer_links.csv'}")
    log(f"Wrote {settings.output_dir / '06_fallback_flows.csv'}")
    log(f"Wrote {settings.output_dir / '06_compost_supply.csv'}")
    log(f"Wrote {settings.output_dir / '06_nutrient_slacks.csv'}")
    log(f"Wrote {settings.output_dir / '06_solution_summary.csv'}")


if __name__ == "__main__":
    main()
