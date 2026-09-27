import base64
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
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

# --- Optimization Engine (Market Clearing) ---
with st.spinner("Running Day-Ahead Market Clearing Optimization..."):
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

    # Formulate Optimization Problem using PuLP (Using safe explicit for-loops)
    market_model = pulp.LpProblem("Day_Ahead_Market_Clearing", pulp.LpMaximize)

    p_gen = {}
    for i in supply_bids.index:
        gen_name = str(supply_bids.loc[i, "Gen_ID"])
        cap = float(supply_bids.loc[i, "Capacity_MW"])
        p_gen[i] = pulp.LpVariable(
            f"Gen_{gen_name}", lowBound=0.0, upBound=cap, cat="Continuous"
        )

    p_dem = {}
    for j in demand_bids.index:
        dem_name = str(demand_bids.loc[j, "Demand_ID"])
        vol = float(demand_bids.loc[j, "Volume_MW"])
        p_dem[j] = pulp.LpVariable(
            f"Dem_{dem_name}", lowBound=0.0, upBound=vol, cat="Continuous"
        )

    social_welfare = pulp.lpSum(
        p_dem[j] * float(demand_bids.loc[j, "Willingness_to_Pay"])
        for j in demand_bids.index
    ) - pulp.lpSum(
        p_gen[i] * float(supply_bids.loc[i, "Marginal_Cost"])
        for i in supply_bids.index
    )

    market_model += social_welfare
    market_model += (
        pulp.lpSum(p_gen[i] for i in supply_bids.index)
        == pulp.lpSum(p_dem[j] for j in demand_bids.index),
        "Market_Balance",
    )

    market_model.solve(pulp.PULP_CBC_CMD(msg=0))

    mcp = market_model.constraints["Market_Balance"].pi
    supply_bids["Cleared_Volume_MW"] = [
        pulp.value(p_gen[i]) for i in supply_bids.index
    ]
    demand_bids["Cleared_Volume_MW"] = [
        pulp.value(p_dem[j]) for j in demand_bids.index
    ]

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
col3.metric(
    "Social Welfare", f"€{pulp.value(market_model.objective):,.2f}"
)

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
