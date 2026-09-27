import base64
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import streamlit as st

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Day-Ahead Market Clearing & Alpine Hydro Simulator",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Day-Ahead Market Clearing & Alpine Hydro Simulator")
st.markdown(
    "Interactive engineering and economic application modeling Day-Ahead market"
    " clearing, social welfare maximization, and Alpine cross-border dynamics."
)

# --- Sidebar Controls ---
st.sidebar.header("🎛️ Simulation Parameters")

num_gens = st.sidebar.slider(
    "Number of Supply Generators", min_value=5, max_value=20, value=12, step=1
)
num_demands = st.sidebar.slider(
    "Number of Demand Blocks", min_value=5, max_value=15, value=10, step=1
)

# --- Optimization Engine (SciPy Market Clearing) ---
with st.spinner("Running Day-Ahead Market Clearing Engine (SciPy)..."):
    np.random.seed(42)

    # Generate Supply Bids Data
    gen_capacities = np.random.uniform(50, 250, num_gens)
    gen_costs = np.sort(np.random.uniform(15, 110, num_gens))
    supply_bids = pd.DataFrame(
        {
            "Gen_ID": [f"G_{i+1}" for i in range(num_gens)],
            "Capacity_MW": gen_capacities,
            "Marginal_Cost": gen_costs,
        }
    )

    # Generate Demand Bids Data
    dem_volumes = np.random.uniform(40, 200, num_demands)
    dem_values = np.sort(np.random.uniform(50, 130, num_demands))[::-1]
    demand_bids = pd.DataFrame(
        {
            "Demand_ID": [f"D_{j+1}" for j in range(num_demands)],
            "Volume_MW": dem_volumes,
            "Willingness_to_Pay": dem_values,
        }
    )

    # =========================================================================
    # BULLETPROOF SCIPY OPTIMIZATION (No PuLP, No Rust-Core Crashes)
    # =========================================================================
    # Objective: Minimize (Gen_Cost * Gen_Vol) - (Dem_WTP * Dem_Vol)
    c_gen = supply_bids["Marginal_Cost"].values
    c_dem = -demand_bids["Willingness_to_Pay"].values
    c = np.concatenate([c_gen, c_dem])

    # Equality constraint: sum(Gen_Vol) - sum(Dem_Vol) == 0
    A_eq = np.concatenate([np.ones(num_gens), -np.ones(num_demands)]).reshape(1, -1)
    b_eq = np.array([0.0])

    # Variable Bounds: (0, Max_Capacity)
    bounds_gen = [(0, cap) for cap in supply_bids["Capacity_MW"].values]
    bounds_dem = [(0, vol) for vol in demand_bids["Volume_MW"].values]
    bounds = bounds_gen + bounds_dem

    # Solve using SciPy's highly optimized HiGHS solver
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        # Extract cleared volumes
        supply_bids["Cleared_Volume_MW"] = res.x[:num_gens]
        demand_bids["Cleared_Volume_MW"] = res.x[num_gens:]
        
        # Calculate Social Welfare (Negative of the minimized objective)
        social_welfare = -res.fun
        
        # Calculate Market Clearing Price (MCP) via Shadow Price or Fallback
        try:
            mcp = abs(res.eqlin.marginals[0])
        except (AttributeError, TypeError, IndexError):
            active_gen = supply_bids[supply_bids["Cleared_Volume_MW"] > 0.1]
            mcp = active_gen["Marginal_Cost"].max() if not active_gen.empty else 0.0
    else:
        st.error("Market Clearing Failed to Converge!")
        st.stop()

    # Save Outputs to 'outputs' Directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    supply_bids.to_csv(
        os.path.join(output_dir, "market_clearing_supply.csv"), index=False
    )
    demand_bids.to_csv(
        os.path.join(output_dir, "market_clearing_demand.csv"), index=False
    )

    # Plot Market Equilibrium Curves
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_supply = supply_bids.sort_values("Marginal_Cost").reset_index(drop=True)
    sorted_supply["Cumulative_Capacity"] = sorted_supply["Capacity_MW"].cumsum()
    sorted_demand = demand_bids.sort_values(
        "Willingness_to_Pay", ascending=False
    ).reset_index(drop=True)
    sorted_demand["Cumulative_Volume"] = sorted_demand["Volume_MW"].cumsum()

    ax.step(
        sorted_supply["Cumulative_Capacity"],
        sorted_supply["Marginal_Cost"],
        where="post",
        label="Supply Curve",
        color="red",
        linewidth=2,
    )
    ax.step(
        sorted_demand["Cumulative_Volume"],
        sorted_demand["Willingness_to_Pay"],
        where="post",
        label="Demand Curve",
        color="blue",
        linewidth=2,
    )
    total_cleared_vol = supply_bids["Cleared_Volume_MW"].sum()
    ax.axhline(
        y=mcp,
        color="green",
        linestyle="--",
        label=f"Market Clearing Price (MCP): {mcp:.2f} €/MWh",
    )
    ax.axvline(
        x=total_cleared_vol,
        color="gray",
        linestyle=":",
        label=f"Cleared Volume: {total_cleared_vol:.1f} MW",
    )

    ax.set_title(
        "Day-Ahead Market Clearing & Marginal Pricing",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Volume (MW)")
    ax.set_ylabel("Price (€/MWh)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    chart_path = os.path.join(output_dir, "market_clearing_dynamics.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()

# --- Main Dashboard Layout ---
col1, col2, col3 = st.columns(3)
col1.metric("Market Clearing Price (MCP)", f"{mcp:.2f} €/MWh")
col2.metric("Total Cleared Volume", f"{total_cleared_vol:.1f} MW")
col3.metric("Social Welfare", f"€{social_welfare:,.2f}")

st.markdown("---")
st.subheader("📈 Market Equilibrium & Clearing Dynamics")
if os.path.exists(chart_path):
    st.image(chart_path, use_container_width=True)

st.markdown("---")
col_s, col_d = st.columns(2)
with col_s:
    st.subheader("📊 Cleared Supply Bids")
    st.dataframe(supply_bids, use_container_width=True)
with col_d:
    st.subheader("📊 Cleared Demand Bids")
    st.dataframe(demand_bids, use_container_width=True)

# --- Sidebar Download Bundle ---
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Download Outputs Bundle")
zip_filename = "alpine_hydro_outputs_bundle"
if os.path.exists(output_dir):
    shutil.make_archive(zip_filename, "zip", output_dir)
    zip_path = f"{zip_filename}.zip"
    if os.path.exists(zip_path):
        with open(zip_path, "rb") as f:
            bytes_data = f.read()
        b64 = base64.b64encode(bytes_data).decode()
        href = f'<a href="data:file/zip;base64,{b64}" download="{zip_path}" style="text-decoration: none;"><button style="background-color: #ff4b4b; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">📦 Download All Outputs (.zip)</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
