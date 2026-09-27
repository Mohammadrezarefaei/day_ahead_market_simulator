import pandas as pd
import pytest
from src.market_engine import run_market_clearing_model


def test_run_market_clearing_model():
  # اجرای مدل با مقادیر کوچک‌تر برای تست سریع
  supply_df, demand_df, mcp, objective = run_market_clearing_model(
      num_gens=5, num_demands=5
  )

  # بررسی اینکه خروجی‌ها ساختار دیتافریم دارند
  assert isinstance(supply_df, pd.DataFrame)
  assert isinstance(demand_df, pd.DataFrame)

  # بررسی وجود ستون‌های کلیدی مربوط به حجم‌های پاکسازی‌شده
  assert "Cleared_Volume_MW" in supply_df.columns
  assert "Cleared_Volume_MW" in demand_df.columns

  # بررسی صحت محاسبات اقتصادی و قیمت تعادلی (MCP)
  assert mcp >= 0
  assert objective > 0

  # بررسی عدم وجود حجم‌های منفی در نتایج
  assert (supply_df["Cleared_Volume_MW"] >= 0).all()
  assert (demand_df["Cleared_Volume_MW"] >= 0).all()
