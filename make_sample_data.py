"""Generate a realistic sample CSV (about 1 year of hourly household power data).

Usage:  python make_sample_data.py
Creates sample_data/household_power_hourly.csv
"""
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
idx = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h")

hour = idx.hour.values
dow = idx.dayofweek.values
doy = idx.dayofyear.values

# morning and evening peaks, higher on weekends, winter heating/summer cooling bump
daily = 0.6 + 0.9 * np.exp(-((hour - 7.5) ** 2) / 4) + 1.4 * np.exp(-((hour - 20) ** 2) / 6)
weekend = np.where(dow >= 5, 1.15, 1.0)
season = 1.0 + 0.25 * np.cos(2 * np.pi * (doy - 15) / 365) + 0.1 * np.cos(4 * np.pi * (doy - 15) / 365)
power = daily * weekend * season + rng.normal(0, 0.12, len(idx))
power = np.clip(power, 0.15, None)

# a few abnormal spikes so the alert feature has something to find
spikes = rng.choice(len(idx), size=12, replace=False)
power[spikes] += rng.uniform(2.5, 4.0, size=12)

df = pd.DataFrame({"timestamp": idx, "global_active_power_kw": power.round(3)})

# make it a bit messy on purpose so validation has work to do
df.loc[rng.choice(len(df), 40, replace=False), "global_active_power_kw"] = np.nan
df = pd.concat([df, df.sample(10, random_state=1)]).sort_values("timestamp")

os.makedirs("sample_data", exist_ok=True)
out = os.path.join("sample_data", "household_power_hourly.csv")
df.to_csv(out, index=False)
print(f"Wrote {out} with {len(df)} rows")
