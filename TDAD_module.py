"高温乾式ADにおける化学量論・物質/エネルギー収支の算定"

import pandas as pd

try:
    df_biowaste = pd.read_csv("00_biowaste.csv").set_index("BIOWASTE")
except Exception:
    df_biowaste = pd.DataFrame()

# 量論式 (Moscoviz & Jimenez, 2021)
def coeffs_for_x(x, h, o, n):
    a = 1 - 0.25 * h - 0.5 * o + 1.75 * n - (x / 4.2) * (2.6 + 0.65 * h - 1.3 * o - 1.95 * n)
    b = (1 - x) * (0.5 + 0.125 * h - 0.25 * o - 0.375 * n)
    c = 0.5 - 0.125 * h + 0.25 * o - 0.625 * n - (x / 4.2) * (1.1 + 0.275 * h - 0.55 * o - 0.825 * n)
    d = n - (x / 4.2) * (0.8 + 0.2 * h - 0.4 * o - 0.6 * n)
    e = (x / 4.2) * (4 + h - 2 * o - 3 * n)
    return {"a": a, "b": b, "c": c, "d": d, "e": e}

# TDADには一廃+糞尿
_DEFAULT_ACCEPTED_FEEDSTOCKS = [
    "FOOD_WASTE",
    "PAPER_WASTE",
    "WOOD_WASTE",
    "CATTLE_F",
    "CATTLE_U",
    "SWINE_F",
    "SWINE_U",
    "POULTRY_E",
]

_DEFAULT_PROPS = {
    "FOOD_WASTE": {
        "WATER": 0.78, "VS": 0.17, "VS_REMOVE": 0.80,
        "CAR": 0.393, "HYD": 0.075, "OXY": 0.413,
        "NIT": 0.037, "PHO": 0.012, "KAL": 0.011,
    },
    "PAPER_WASTE": {
        "WATER": 0.317, "VS": 0.629, "VS_REMOVE": 0.50,
        "CAR": 0.4704, "HYD": 0.0649, "OXY": 0.457,
        "NIT": 0.0065, "PHO": 0.0, "KAL": 0.0,
    },
    "WOOD_WASTE": {
        "WATER": 0.431, "VS": 0.544, "VS_REMOVE": 0.20,
        "CAR": 0.5071, "HYD": 0.0617, "OXY": 0.418,
        "NIT": 0.0112, "PHO": 0.0, "KAL": 0.0,
    },
    "CATTLE_F": {
        "WATER": 0.80, "VS": 0.20, "VS_REMOVE": 0.30,
        "CAR": 0.346, "HYD": 0.060, "OXY": 0.5492,
        "NIT": 0.022, "PHO": 0.0078, "KAL": 0.015,
    },
    "CATTLE_U": {
        "WATER": 0.99, "VS": 0.01, "VS_REMOVE": 0.30,
        "CAR": 0.0, "HYD": 0.0, "OXY": 0.0,
        "NIT": 0.27, "PHO": 0.0, "KAL": 0.73,
    },
    "SWINE_F": {
        "WATER": 0.80, "VS": 0.20, "VS_REMOVE": 0.50,
        "CAR": 0.413, "HYD": 0.060, "OXY": 0.455,
        "NIT": 0.036, "PHO": 0.024, "KAL": 0.012,
    },
    "SWINE_U": {
        "WATER": 0.98, "VS": 0.02, "VS_REMOVE": 0.50,
        "CAR": 0.0, "HYD": 0.0, "OXY": 0.0,
        "NIT": 0.33, "PHO": 0.0, "KAL": 0.0,
    },
    "POULTRY_E": {
        "WATER": 0.64, "VS": 0.36, "VS_REMOVE": 0.50,
        "CAR": 0.347, "HYD": 0.060, "OXY": 0.482,
        "NIT": 0.062, "PHO": 0.023, "KAL": 0.026,
    },
}

def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default

# 受入廃棄物
def _accepted_feedstocks():
    accepted = list(_DEFAULT_ACCEPTED_FEEDSTOCKS)

    if not df_biowaste.empty:
        names = list(df_biowaste.index.astype(str))
        if "CATTLE_F" in names and "POULTRY_E" in names:
            start = names.index("CATTLE_F")
            end = names.index("POULTRY_E")
            if start <= end:
                for name in names[start:end + 1]:
                    if name not in accepted:
                        accepted.append(name)

    return accepted

ACCEPTED_FEEDSTOCKS = _accepted_feedstocks()

# バイオマス組成
def _biowaste_props(feedstock):
    feedstock = str(feedstock)
    base = _DEFAULT_PROPS.get(feedstock, {
        "WATER": 0.78, "VS": 0.17, "VS_REMOVE": 0.80,
        "CAR": 0.393, "HYD": 0.075, "OXY": 0.413,
        "NIT": 0.037, "PHO": 0.012, "KAL": 0.011,
    }).copy()

    if feedstock in df_biowaste.index:
        row = df_biowaste.loc[feedstock]
        for key, default in base.items():
            base[key] = _safe_float(row.get(key, default), default)

    return base

def _normalize_feedstocks(amount_kg,
                          paper_waste_amount=0.0,
                          wood_waste_amount=0.0,
                          cattle_f_amount=0.0,
                          cattle_u_amount=0.0,
                          swine_f_amount=0.0,
                          swine_u_amount=0.0,
                          poultry_e_amount=0.0):
    """Accept either scalar FOOD_WASTE kg/year or dict of TDAD feedstocks."""
    if isinstance(amount_kg, dict):
        feedstocks = {str(k): _safe_float(v, 0.0) for k, v in amount_kg.items()}
    else:
        feedstocks = {"FOOD_WASTE": _safe_float(amount_kg, 0.0)}
        optional = {
            "PAPER_WASTE": paper_waste_amount,
            "WOOD_WASTE": wood_waste_amount,
            "CATTLE_F": cattle_f_amount,
            "CATTLE_U": cattle_u_amount,
            "SWINE_F": swine_f_amount,
            "SWINE_U": swine_u_amount,
            "POULTRY_E": poultry_e_amount,
        }
        for key, value in optional.items():
            value = _safe_float(value, 0.0)
            if value > 0:
                feedstocks[key] = value

    allowed = set(ACCEPTED_FEEDSTOCKS)
    return {k: max(0.0, v) for k, v in feedstocks.items() if k in allowed and v > 0}

# TS濃度算定→反応槽の容量に活用
def _capacity_daily_ts(capacity_amount, refusal_rate, mixed_ts_frac):
    if isinstance(capacity_amount, dict):
        daily_ts_cap = 0.0
        for feedstock, annual_kg in _normalize_feedstocks(capacity_amount).items():
            props = _biowaste_props(feedstock)
            daily_ts_cap += annual_kg / 365.0 * (1.0 - refusal_rate) * (1.0 - props["WATER"])
        cap_total = sum(_normalize_feedstocks(capacity_amount).values())
        return daily_ts_cap, cap_total

    cap_total = _safe_float(capacity_amount, 0.0)
    daily_ts_cap = cap_total / 365.0 * (1.0 - refusal_rate) * mixed_ts_frac
    return daily_ts_cap, cap_total

# 高温乾式AD
def tdad_run(amount_kg,
             avg_air_temp=15.0,
             capacity_amount=None,
             paper_waste_amount=0.0,
             wood_waste_amount=0.0,
             cattle_f_amount=0.0,
             cattle_u_amount=0.0,
             swine_f_amount=0.0,
             swine_u_amount=0.0,
             poultry_e_amount=0.0,
             Biogas_griduse=False):

    feedstocks = _normalize_feedstocks(
        amount_kg,
        paper_waste_amount=paper_waste_amount,
        wood_waste_amount=wood_waste_amount,
        cattle_f_amount=cattle_f_amount,
        cattle_u_amount=cattle_u_amount,
        swine_f_amount=swine_f_amount,
        swine_u_amount=swine_u_amount,
        poultry_e_amount=poultry_e_amount,
    )
    total_amount = sum(feedstocks.values())
    if total_amount <= 0:
        return {}

    if capacity_amount is None:
        capacity_amount = total_amount

    # 基本パラメタ
    reactor_temp = 55.0
    TS_target = 0.20
    refusal_rate = 0.10
    max_TS_load = 5.00  # kg-TS/(m3-day)

    refused = total_amount / 365.0 * refusal_rate  # kg/day

    amount_in = 0.0          # kg/day after refusal
    ts_mass_kg = 0.0         # kg-TS/day
    vs_in = 0.0              # kg-VS/day
    vs_removed = 0.0         # kg-VSremoved/day
    elem_mass = {"C": 0.0, "H": 0.0, "O": 0.0, "N": 0.0, "P": 0.0, "K": 0.0}

    for feedstock, annual_kg in feedstocks.items():
        props = _biowaste_props(feedstock)
        daily_after_refusal = annual_kg / 365.0 * (1.0 - refusal_rate)

        amount_in += daily_after_refusal
        ts_mass_kg += daily_after_refusal * (1.0 - props["WATER"])

        feed_vs = daily_after_refusal * props["VS"]
        feed_vs_removed = feed_vs * props["VS_REMOVE"]
        vs_in += feed_vs
        vs_removed += feed_vs_removed

        elem_mass["C"] += feed_vs * props["CAR"]
        elem_mass["H"] += feed_vs * props["HYD"]
        elem_mass["O"] += feed_vs * props["OXY"]
        elem_mass["N"] += feed_vs * props["NIT"]
        elem_mass["P"] += feed_vs * props["PHO"]
        elem_mass["K"] += feed_vs * props["KAL"]

    # 上水使用量: same logic as MWAD, with TDAD target TS.
    ts_mass = ts_mass_kg * 0.001  # t-TS/day
    target_ts_mass = amount_in * 0.001 * TS_target  # t-TS/day
    water_add = max(0.0, (ts_mass - target_ts_mass) / TS_target)  # m3/day

    # VS除去
    ts_dig = ts_mass_kg - vs_removed  # kg-TSremain/day

    # 量論式
    mw = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007}
    moles = {k: elem_mass[k] / mw[k] for k in mw}
    nC = moles["C"]
    if nC > 0:
        h_r = moles["H"] / nC
        o_r = moles["O"] / nC
        n_r = moles["N"] / nC
    else:
        h_r, o_r, n_r = 0.0, 0.0, 0.0

    cf = coeffs_for_x(0.125, h_r, o_r, n_r)
    b, c, d, e = cf["b"], cf["c"], cf["d"], cf["e"]
    denom = 12 + h_r + 16 * o_r + 14 * n_r - 24.6 * e

    # バイオガス
    ch4_rate = 350.0 * 64.0 * b / denom if denom != 0 else 0.0  # L-CH4/kg-VSremoved
    ch4_total = ch4_rate * vs_removed  # L-CH4/day
    co2_total = ch4_total * (c / b) if b != 0 else 0.0  # L-CO2/day
    biogas_total = ch4_total + co2_total  # L-biogas/day

    c_input = elem_mass["C"]  # kgC/day
    c_gas = ((ch4_total + co2_total) / 22.4) * 12.011 * 0.001  # kgC/day
    c_digest = max(0.0, c_input - c_gas)  # kgC/day

    # Mass balance for Digestate
    gas_mass = ((ch4_total * 16.04 + co2_total * 44.01) / 22.4) * (10 ** -6)  # t/day
    digestate_mass = (amount_in * 0.001) + water_add - gas_mass  # t/day

    # 消化液NPK + NH3揮発
    nmin_rate = 14.0 * d / denom if denom != 0 else 0.0  # kg-Nmin/kg-VSremoved
    nmin_total = nmin_rate * vs_removed  # kg-Nmin/day
    nmin_conc = nmin_total / digestate_mass if digestate_mass > 0 else 0.0  # kg-NH4+/t-dig
    nh4_mol = (nmin_conc * 1000) / 14.007  # mol-NH4+/t-dig

    ad_temp_k = reactor_temp + 273.15
    pka = 0.09018 + (2727.92 / ad_temp_k)
    f_nh3 = 1.0 / (1.0 + 10.0 ** (pka - 7.0))
    nh3_loss = (nh4_mol * f_nh3 * digestate_mass * 17.031) / 1e6  # t-NH3/day

    n_digest = max(0.0, elem_mass["N"] - (nh3_loss / 17.031 * 1000 * 14.007))  # kg-N/day
    p_digest = elem_mass["P"]  # kg-P/day
    k_digest = elem_mass["K"]  # kg-K/day

    # エネルギー使用量
    elec_machinery = 290 * 0.001 * amount_in  # kWh/day
    elec_upgrade = 1.3 * 0.001 * biogas_total if Biogas_griduse else 0.0  # kWh/day
    elec_req_total = elec_machinery + elec_upgrade  # kWh/day

    # 基質加熱
    delta_t = reactor_temp - avg_air_temp
    mcal_to_mj = 4.184
    heat_substrate = delta_t * ((amount_in * 0.001) + water_add) * mcal_to_mj  # MJ/day

    # 反応槽熱損失
    mixed_ts_frac = ts_mass_kg / amount_in if amount_in > 0 else TS_target
    daily_ts_cap, cap_total = _capacity_daily_ts(capacity_amount, refusal_rate, mixed_ts_frac)
    reactor_vol = daily_ts_cap / max_TS_load if max_TS_load > 0 else 0.0

    thick_concrete = 0.2; lambda_concrete = 2.3  # m, W/m-K
    thick_PS = 0.05; lambda_PS = 0.03  # m, W/m-K
    thick_air = 1.0; lambda_air = 25.0  # m, W/m-K
    R_total = (thick_concrete / lambda_concrete) + (thick_PS / lambda_PS) + (thick_air / lambda_air)
    k_combined = 1.0 / R_total  # W/m2-K
    sa_v_ratio = 5.251  # m2/m3

    heat_loss = k_combined * reactor_vol * sa_v_ratio * delta_t * 0.0864  # MJ/day
    heat_req_total = (heat_substrate + heat_loss) * 1.1  # MJ/day

    # CAPEX
    tdad_capex = 112.86 * ((cap_total * 0.001 / 365.0) ** 0.7999) * (10 ** 6) if cap_total > 0 else 0.0

    result = {
        "TDAD_REFUSED": refused * 365,  # kg/year
        "TDAD_ELEC_REQ": elec_req_total * 365,  # kWh/year
        "TDAD_HEAT_REQ": heat_req_total * 365,  # MJ/year
        "TDAD_WATER_REQ": water_add * 365,  # m3/year
        "TDAD_BIOGAS": ch4_total * 0.001 * 365,  # Nm3-CH4/year
        "TDAD_DIGESTATE": digestate_mass * 1000.0 * 365,  # kg/year
        "TDAD_DIG_TS": ts_dig * 365,  # kg/year
        "TDAD_DIG_ORGC": c_digest * 365,  # kgC/year
        "TDAD_DIG_N": n_digest * 365,  # kgN/year
        "TDAD_DIG_P2O5": p_digest * 2.291 * 365,  # kgP2O5/year
        "TDAD_DIG_K2O": k_digest * 1.205 * 365,  # kgK2O/year
        "TDAD_CAPEX": tdad_capex,
        "TDAD_TOTAL_INPUT": total_amount,  # kg/year before refusal
        "TDAD_TOTAL_ACCEPTED_INPUT": amount_in * 365,  # kg/year after refusal
        "TDAD_MIXED_TS_FRAC": mixed_ts_frac,
    }

    for feedstock in ACCEPTED_FEEDSTOCKS:
        result[f"TDAD_{feedstock}_IN"] = feedstocks.get(feedstock, 0.0)  # kg/year before refusal

    result.update({
        "TDAD_refused": result["TDAD_REFUSED"],
        "TDAD_digestate": result["TDAD_DIGESTATE"] * 0.001,  # t/year
        "TDAD_digTS": result["TDAD_DIG_TS"] * 0.001,  # t-TS/year
        "TDAD_digsolid": result["TDAD_DIG_TS"] * 0.001 * 0.95 / (1 - 0.75),  # t/year
        "CH4": result["TDAD_BIOGAS"],
        "ELEC": result["TDAD_ELEC_REQ"],
        "ENERGY_MJ": result["TDAD_HEAT_REQ"],
    })
    return result
