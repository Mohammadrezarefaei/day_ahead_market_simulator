import pulp
import pandas as pd
import numpy as np


def run_market_clearing_model(num_gens=12, num_demands=10):
  """اجرای مدل بهینه‌سازی خطی برای پاکسازی بازار Day-Ahead و محاسبه MCP"""
  np.random.seed(42)

  gen_capacities = np.random.uniform(50, 250, num_gens)
  gen_costs = np.sort(np.random.uniform(15, 110, num_gens))
  supply_bids = pd.DataFrame(
      {
          "Gen_ID": [f"G_{i+1}" for i in range(num_gens)],
          "Capacity_MW": gen_capacities,
          "Marginal_Cost": gen_costs,
      }
  )

  dem_volumes = np.random.uniform(40, 200, num_demands)
  dem_values = np.sort(np.random.uniform(50, 130, num_demands))[::-1]
  demand_bids = pd.DataFrame(
      {
          "Demand_ID": [f"D_{j+1}" for j in range(num_demands)],
          "Volume_MW": dem_volumes,
          "Willingness_to_Pay": dem_values,
      }
  )

  market_model = pulp.LpProblem("Day_Ahead_Market_Clearing", pulp.LpMaximize)

  p_gen = {
      i: pulp.LpVariable(
          f"Gen_{i}",
          lowBound=0,
          upBound=supply_bids.loc[i, "Capacity_MW"],
          cat="Continuous",
      )
      for i in supply_bids.index
  }
  p_dem = {
      j: pulp.LpVariable(
          f"Dem_{j}",
          lowBound=0,
          upBound=demand_bids.loc[j, "Volume_MW"],
          cat="Continuous",
      )
      for j in demand_bids.index
  }

  social_welfare = pulp.lpSum(
      p_dem[j] * demand_bids.loc[j, "Willingness_to_Pay"]
      for j in demand_bids.index
  ) - pulp.lpSum(
      p_gen[i] * supply_bids.loc[i, "Marginal_Cost"] for i in supply_bids.index
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

  return supply_bids, demand_bids, mcp, pulp.value(market_model.objective)
