# ───────────────────────── 0 · Imports & global style ─────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import to_rgba
from myLibs.backupPowerSystems import EnergyStorageSystem
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel
import numpy as np

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size"  : 13,
    "font.weight": "bold"
})

# ───────────────────────── 1 · Load hourly CSV files ──────────────────────────
pv = (pd.read_csv(r"Data\Test\PowerGeneration_Solar__10_20_Jan_2025.csv",
                  parse_dates=["Timestamp"])
        .set_index("Timestamp")[["PV_kW"]])

wt = (pd.read_csv(r"Data\Test\PowerGeneration_Wind__10_20_Jan_2025.csv",
                  parse_dates=["Timestamp"])
        .set_index("Timestamp")[["WT_kW"]])

load = (pd.read_csv(r"Data\Test\PowerDemand__CBSs_Wollongong_specific_area__10_20_Jan_2025.csv",
                    parse_dates=["Timestamp"])
        .set_index("Timestamp")[["CBSs_kW"]])

df = (pv.join(wt, how="outer")
        .join(load, how="outer")
        .fillna(0.0)
        .sort_index())

df["TRE_kW"]        = df["PV_kW"] + df["WT_kW"]
df["Charge_kW"]     = 0.0
df["Discharge_kW"]  = 0.0
df["Grid_Imp_kW"]   = 0.0
df["Grid_Exp_kW"]   = 0.0
df["SOC_%"]         = None

# ───────────────────────── 2 · ESS model & control marks ──────────────────────
ess = EnergyStorageSystem(stacks=16, autonomy_days=3, soc=0.30)
t_sev_start = pd.Timestamp("2025-01-15 14:20")
t_sev_end   = pd.Timestamp("2025-01-18 22:59")

def hrs_to_severe(ts): 
    return max(0, (t_sev_start - ts).total_seconds() / 3600)

def discharge_to_floor(request_kw):
    """Supply up to *request_kw* kW from ESS (1 h) respecting 20 % SOC floor."""
    soc_before = ess.soc
    ess.discharge(request_kw, dt_hr=1)
    supplied = (soc_before - ess.soc) * ess.E_B
    return supplied, request_kw - supplied

# ───────────────────────── 3 · Quantum annealing dispatcher ───────────────────
# Initialize quantum annealer
sampler = EmbeddingComposite(DWaveSampler())

# Discretization parameters for QUBO (power levels in kW)
power_levels = np.linspace(0, 1000, 10)  # Adjust range based on typical power values
num_bits = len(power_levels)

def encode_power(power, levels):
    """Map continuous power value to closest discrete level."""
    idx = np.argmin(np.abs(levels - power))
    return idx, levels[idx]

def decode_power(idx, levels):
    """Map discrete index back to power value."""
    return levels[idx]

for ts, row in df.iterrows():
    P_res = row["TRE_kW"]
    P_load = row["CBSs_kW"]
    e_prev = ess.energy_kwh

    # Define QUBO for this timestamp
    bqm = BinaryQuadraticModel('BINARY')

    # Decision variables: Binary variables for each power level
    # For P_charge, P_discharge, P_grid_imp, P_grid_exp
    vars_charge = [f'charge_{i}' for i in range(num_bits)]
    vars_discharge = [f'discharge_{i}' for i in range(num_bits)]
    vars_grid_imp = [f'grid_imp_{i}' for i in range(num_bits)]
    vars_grid_exp = [f'grid_exp_{i}' for i in range(num_bits)]

    # Objective: Minimize grid imports, maximize exports (except in severe window)
    w1 = 10.0 if t_sev_start <= ts <= t_sev_end else 1.0  # High penalty for imports during severe weather
    w2 = 100.0  # Penalty for unmet load
    w3 = 0.0 if t_sev_start <= ts <= t_sev_end else 0.5  # Reward exports outside severe window
    w4 = 10.0 if ts < t_sev_start else 0.0  # Encourage pre-event charging

    # Add linear terms for objectives
    for i in range(num_bits):
        bqm.add_variable(vars_grid_imp[i], w1 * power_levels[i])
        bqm.add_variable(vars_grid_exp[i], -w3 * power_levels[i])
        bqm.add_variable(vars_charge[i], w4 * power_levels[i] if ess.soc < 0.90 else 0.0)

    # Constraint 1: Power balance (P_res + P_discharge + P_grid_imp = P_load + P_charge + P_grid_exp)
    for i in range(num_bits):
        for j in range(num_bits):
            for k in range(num_bits):
                for l in range(num_bits):
                    balance = (P_res + power_levels[i] - power_levels[j] - power_levels[k] - power_levels[l])
                    bqm.add_quadratic(
                        vars_discharge[i], vars_charge[j], w2 * balance**2
                    )
                    bqm.add_quadratic(
                        vars_discharge[i], vars_grid_exp[k], w2 * balance**2
                    )
                    bqm.add_quadratic(
                        vars_charge[j], vars_grid_exp[k], w2 * balance**2
                    )
                    bqm.add_quadratic(
                        vars_discharge[i], vars_grid_imp[l], w2 * balance**2
                    )
                    bqm.add_quadratic(
                        vars_charge[j], vars_grid_imp[l], w2 * balance**2
                    )
                    bqm.add_quadratic(
                        vars_grid_imp[l], vars_grid_exp[k], w2 * balance**2
                    )

    # Constraint 2: Only one power level active per variable
    for var_group in [vars_charge, vars_discharge, vars_grid_imp, vars_grid_exp]:
        for i in range(num_bits):
            bqm.add_variable(var_group[i], 0.0)  # Placeholder
        for i in range(num_bits):
            for j in range(i + 1, num_bits):
                bqm.add_quadratic(var_group[i], var_group[j], 1000.0)  # Penalty for multiple selections

    # Constraint 3: ESS SOC constraints (20%–90%)
    soc_target = 0.90 if ts < t_sev_start else ess.soc
    for i in range(num_bits):
        delta_soc = (power_levels[i] / ess.E_B) * (1 if i in vars_charge else -1 / ess.eta_d)
        bqm.add_variable(vars_charge[i], 10.0 * (ess.soc + delta_soc - soc_target)**2)
        bqm.add_variable(vars_discharge[i], 10.0 * (ess.soc + delta_soc - soc_target)**2)

    # Solve QUBO
    response = sampler.sample(bqm, num_reads=100)
    solution = response.first.sample

    # Extract results
    charge_idx = next((i for i, v in enumerate(vars_charge) if solution[v]), 0)
    discharge_idx = next((i for i, v in enumerate(vars_discharge) if solution[v]), 0)
    grid_imp_idx = next((i for i, v in enumerate(vars_grid_imp) if solution[v]), 0)
    grid_exp_idx = next((i for i, v in enumerate(vars_grid_exp) if solution[v]), 0)

    P_charge = decode_power(charge_idx, power_levels)
    P_discharge = decode_power(discharge_idx, power_levels)
    P_grid_imp = decode_power(grid_imp_idx, power_levels)
    P_grid_exp = decode_power(grid_exp_idx, power_levels)

    # Update ESS and dataframe
    ess.charge(P_charge, 1)
    supplied, _ = discharge_to_floor(P_discharge)
    df.at[ts, "Charge_kW"] = P_charge
    df.at[ts, "Discharge_kW"] = supplied
    df.at[ts, "Grid_Imp_kW"] = P_grid_imp
    df.at[ts, "Grid_Exp_kW"] = P_grid_exp
    df.at[ts, "SOC_%"] = round(ess.soc * 100, 2)

# ───────────────────────── 4 · Energy accounting & console report ────────────
def nearest_val(series: pd.Series, ts: pd.Timestamp) -> float:
    idx = series.index.get_indexer([ts], method="nearest")[0]
    return series.iloc[idx]

E_res = df["TRE_kW"].sum()
E_load = df["CBSs_kW"].sum()
E_imp = df["Grid_Imp_kW"].sum()
E_exp = df["Grid_Exp_kW"].sum()
E_noRES = E_load + 0.60 * ess.E_B / ess.eta_c
net_grid = E_imp - E_exp

print("────────── Dispatch Summary (10–20 Jan 2025) ──────────")
print(f"Renewable energy available      : {E_res:8.0f} kWh")
print(f"Grid import (strategy)          : {E_imp:8.0f} kWh")
print(f"Grid export (surplus RES sold)  : {E_exp:8.0f} kWh")
print(f"Net grid balance                : {net_grid:8.0f} kWh")
print(f"Grid import (no-RES baseline)   : {E_noRES:8.0f} kWh")
print(f"SOC at window open (≈14:20)     : {nearest_val(df['SOC_%'], t_sev_start):5.1f} %")
print(f"SOC at window close (≈22:59)    : {nearest_val(df['SOC_%'], t_sev_end):5.1f} %")
print("────────────────────────────────────────────────────────")

# ───────────────────────── 5 · Build five-panel figure ──────────────────────
fig, axes = plt.subplots(
    5, 1, figsize=(8, 9),
    sharex=True, constrained_layout=True
)

fig.patch.set_facecolor(to_rgba('lightgray', 0.50))

def shade(ax):
    """Highlight the severe-weather window."""
    ax.axvspan(t_sev_start, t_sev_end, color="brown", alpha=0.30,
               label="Severe Weather Period")

# Row 0 — CBSs (critical load)
axes[0].plot(df.index, df["CBSs_kW"], color="purple", lw=2.2,
             label="CBSs (kW)")
axes[0].set_ylabel("CBSs (kW)", fontweight="bold")
shade(axes[0])

# Row 1 — Grid import / export
imp_line, = axes[1].plot(df.index, df["Grid_Imp_kW"],
                         color="black", lw=2.2, label="Grid Import")
exp_line, = axes[1].plot(df.index, -df["Grid_Exp_kW"],
                         color="gold", lw=2.2, label="Grid Export (−)")
axes[1].set_ylabel("Grid (kW)", fontweight="bold")
shade(axes[1])

# Row 2 — Solar power
axes[2].plot(df.index, df["PV_kW"], color="orange", lw=2.2,
             label="Solar (kW)")
axes[2].set_ylabel("Solar (kW)", fontweight="bold")
shade(axes[2])

# Row 3 — Wind power
axes[3].plot(df.index, df["WT_kW"], color="steelblue", lw=2.2,
             label="Wind (kW)")
axes[3].set_ylabel("Wind (kW)", fontweight="bold")
shade(axes[3])

# Row 4 — ESS charge / discharge
ch_line, = axes[4].plot(df.index, df["Charge_kW"],
                        color="green", lw=2, label="ESS Charge")
ds_line, = axes[4].plot(df.index, df["Discharge_kW"],
                        color="red", lw=2, label="ESS Discharge")
axes[4].set_ylabel("ESS (kW)", fontweight="bold")
shade(axes[4])

# X-axis ticks & label
axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=12))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%y"))
axes[-1].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
axes[-1].tick_params(axis="x", rotation=45, labelsize=12)
axes[-1].set_xlabel("Timestamp (h)", fontweight="bold")

# Uniform grid style
for ax in axes:
    ax.grid(which="minor", axis="x", linestyle="--",
            color="coral", alpha=0.5)
    ax.grid(which="major", axis="y", linestyle="--",
            color="coral", alpha=0.5)

# Combined legend
handles, labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

seen = set()
uniq_handles, uniq_labels = [], []
for h, l in zip(handles, labels):
    if l not in seen:
        uniq_handles.append(h)
        uniq_labels.append(l)
        seen.add(l)

fig.legend(uniq_handles, uniq_labels,
           loc="upper center", bbox_to_anchor=(0.5, 1.12),
           ncol=3, frameon=True)

fig.savefig("Figures/Wollongong_Resilient_Power_System_10_20_Jan_2025.png",
            dpi=600, bbox_inches='tight')

# plt.show()
