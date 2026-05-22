"中温湿式ADにおける化学量論・物質/エネルギー収支の算定"

import pandas as pd

try:
    df_biowaste = pd.read_csv("00_biowaste.csv").set_index("BIOWASTE")
except:
    df_biowaste = pd.DataFrame()

# 量論式 (Moscoviz & Jimenez, 2021)
def coeffs_for_x(x, h, o, n):
    a = 1 - 0.25 * h - 0.5 * o + 1.75 * n - (x / 4.2) * (2.6 + 0.65 * h - 1.3 * o - 1.95 * n)
    b = (1 - x) * (0.5 + 0.125 * h - 0.25 * o - 0.375 * n)
    c = 0.5 - 0.125 * h + 0.25 * o - 0.625 * n - (x / 4.2) * (1.1 + 0.275 * h - 0.55 * o - 0.825 * n)
    d = n - (x / 4.2) * (0.8 + 0.2 * h - 0.4 * o - 0.6 * n)
    e = (x / 4.2) * (4 + h - 2 * o - 3 * n)
    return {"a": a, "b": b, "c": c, "d": d, "e": e}

def mwad_run(food_waste_amount, capacity_amount, air_temp=15.0,
             Biogas_griduse=False): # バイオガス改質有無
    if food_waste_amount <= 0: return {} 
    
    # 基本パラメタ
    reactor_temp = 35.0
    TS_target = 0.10
    refusal_rate = 0.10
    max_TS_load = 5.00 # kg-TS/(m3-day)   
    
    refused = food_waste_amount / 365.0 * refusal_rate # kg/day
    amount_in = food_waste_amount / 365.0 - refused # kg/day

    # 生ごみの元素組成
    if "FOOD_WASTE" in df_biowaste.index:
        props = df_biowaste.loc["FOOD_WASTE"]
        water_content = props.get("WATER", 0.78)
        vs_content = props.get("VS", 0.17)
        vs_remove = props.get("VS_REMOVE", 0.8)
        comp = {"C": props.get("CAR", 0.393), "H": props.get("HYD", 0.075),
                "O": props.get("OXY", 0.413), "N": props.get("NIT", 0.037),
                "P": props.get("PHO", 0.012), "K": props.get("KAL", 0.011)}
    else:
        water_content, vs_content, vs_remove = 0.78, 0.17, 0.8
        comp = {"C":0.393, "H":0.075, "O":0.413, "N":0.037, "P":0.012, "K":0.011}

    # 上水使用量
    ts_mass = amount_in * (1.0 - water_content) * 0.001 # t-TS/day
    target_ts_mass = amount_in * 0.001 * TS_target # t-TS/day
    water_add = max(0.0, (ts_mass - target_ts_mass)/TS_target) # m3/day

    # VS除去
    vs_in = amount_in * vs_content # kg-VS/day
    vs_removed = vs_in * vs_remove # kg-VSremove/day
    
    ts_dig = ts_mass * 1000.0 - vs_removed # kg-TSremain/day
    
    # 量論式
    mw = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007}
    moles = {k: (vs_in * comp[k]) / mw[k] for k in mw}
    nC = moles["C"]
    if nC > 0:
        h_r, o_r, n_r = moles["H"]/nC, moles["O"]/nC, moles["N"]/nC
    else:
        h_r, o_r, n_r = 0, 0, 0
    
    cf = coeffs_for_x(0.125, h_r, o_r, n_r)
    b, c, d, e = cf["b"], cf["c"], cf["d"], cf["e"]
    denom = 12 + h_r + 16*o_r + 14*n_r - 24.6*e
    
    # バイオガス
    if denom != 0:
        ch4_rate = 350.0 * 64.0 * b / denom # L-CH4/kg-VSremoved
    else:
        ch4_rate = 0
        
    ch4_total = ch4_rate * vs_removed # L-CH4/day
    co2_total = ch4_total * (c / b) if b != 0 else 0 # L-CO2/day
    biogas_total = ch4_total + co2_total # L-biogas/day
    
    c_input = vs_in * comp["C"] # kgC/day
    c_gas = ((ch4_total + co2_total) / 22.4) * 12.011 * 0.001 # kgC/day
    c_digest = max(0.0, c_input - c_gas) # kgC/day
    
    # Mass balance for Digestate
    # Gas mass in tons
    gas_mass = ((ch4_total * 16.04 + co2_total * 44.01) / 22.4) * (10 ** -6) # t/day
    digestate_mass = (amount_in * 0.001) + water_add - gas_mass # t/day
    
    # 消化液NPK+NH3揮発
    if denom != 0:
        nmin_rate = 14.0 * d / denom # kg-Nmin/kg-VSremoved
    else:
        nmin_rate = 0
    nmin_total = nmin_rate * vs_removed # kg-Nmin/day
    nmin_conc = nmin_total / digestate_mass if digestate_mass > 0 else 0 # kg-NH4+/t-dig
    nh4_mol = (nmin_conc * 1000) / 14.007 # mol-NH4+/t-dig
    
    ad_temp_k = reactor_temp + 273.15
    pka = 0.09018 + (2727.92 / ad_temp_k)
    f_nh3 = 1.0 / (1.0 + 10.0**(pka - 7.0))
    nh3_loss = (nh4_mol * f_nh3 * digestate_mass * 17.031) / 1e6 # t-NH3
    
    n_input = vs_in * comp["N"] # kg-N/day
    n_digest = max(0, n_input - (nh3_loss/17.031 * 1000 * 14.007)) # kg-N/day (approx fix)
    p_digest = vs_in * comp["P"] # kg-P/day
    k_digest = vs_in * comp["K"] # kg-K/day
    
    # エネルギー使用量
    # 電力
    elec_machinery = 8.6 * 0.001 * amount_in # kWh/day
    elec_upgrade = 1.3 * 0.001 * biogas_total if Biogas_griduse else 0.0 # kWh/day
    elec_req_total = elec_machinery + elec_upgrade # kWh/day
    
    # 基質加熱
    delta_t = reactor_temp - air_temp
    mcal_to_mj = 4.184
    heat_substrate = delta_t * ((amount_in * 0.001) + water_add) * mcal_to_mj # MJ/day
    
    # 反応槽熱損失
    daily_cap = capacity_amount / 365.0 # kg/day
    daily_ts_cap = daily_cap * (1.0 - refusal_rate) * (1.0 - water_content) # kg-TS/day
    if max_TS_load > 0:
        reactor_vol = daily_ts_cap / max_TS_load # m3
    else:
        reactor_vol = 0

    thick_concrete = 0.2; lambda_concrete = 2.3 # m, W/m-K
    thick_PS = 0.05; lambda_PS = 0.03 # m, W/m-K
    thick_air = 1.0; lambda_air = 25.0 # m, W/m-K
    R_total = (thick_concrete/lambda_concrete) + (thick_PS/lambda_PS) + (thick_air/lambda_air)
    k_combined = 1.0 / R_total # W/m2-K
    sa_v_ratio = 5.251 # m2/m3

    heat_loss = k_combined * reactor_vol * sa_v_ratio * delta_t * 0.0864 # MJ/day
    
    heat_req_total = (heat_substrate + heat_loss) * 1.1 # MJ/day

    # CAPEX
    mwad_capex = 91.129 * ((capacity_amount * 0.001 / 365.0) ** 0.7999) * (10 ** 6)

    return {
        "MWAD_REFUSED": refused * 365, # kg/year
        "MWAD_ELEC_REQ": elec_req_total * 365, # kWh/year
        "MWAD_HEAT_REQ": heat_req_total * 365, # MJ/year
        "MWAD_WATER_REQ": water_add * 365, # m3/year
        "MWAD_BIOGAS": ch4_total * 0.001 * 365, # Nm3-CH4/year (ch4_total is L) -> m3
        "MWAD_DIGESTATE": digestate_mass * 1000.0 * 365, # kg/year
        "MWAD_DIG_TS": ts_dig * 365, # kg/year
        "MWAD_DIG_ORGC": c_digest * 365, # kgC/year
        "MWAD_DIG_N": n_digest * 365, # kgN/year
        "MWAD_DIG_P2O5": p_digest * 2.291 * 365, # P2O5(141.94) / 2*P (2*30.97) = 2.291
        "MWAD_DIG_K2O": k_digest * 1.205 * 365, # K2O(94.2)/2K(78.2) = 1.205
        "MWAD_CAPEX": mwad_capex
    }