import pandas as pd
import re

def normalize_key(bus: str, tech: str) -> str:
    """Normalize key to match between different CSVs. Remove region numbers like _00000_."""
    # Remove pattern _XXXXX_ (region numbers)
    bus_clean = re.sub(r'_\d+_', '_', bus)
    return f"{bus_clean}_{tech}"

# Read CFs
print("Reading CFs...")
cfs = pd.read_csv('../results/thesis-wake-2030-10-northsea-dominant-6h/base-s100000-biasFalse/csvs/nodal_cfs.csv', skiprows=4)
offshore_cfs = cfs[cfs.iloc[:, 2].astype(str).str.contains('offwind', case=False, na=False)]
cf_keys = {}
for i, row in offshore_cfs.iterrows():
    bus = str(row.iloc[1])
    tech = str(row.iloc[2])
    cf = float(row.iloc[3])
    key = normalize_key(bus, tech)
    cf_keys[key] = cf
    print(f"CF: bus='{bus}', tech='{tech}', normalized key='{key}'")

print("\n" + "="*80 + "\n")

# Read capacities
print("Reading capacities...")
caps = pd.read_csv('../results/thesis-wake-2030-10-northsea-dominant-6h/base-s100000-biasFalse/csvs/nodal_capacities.csv', skiprows=4)
offshore_caps = caps[caps.iloc[:, 2].astype(str).str.contains('offwind', case=False, na=False)]
cap_keys = {}
for i, row in offshore_caps.head(3).iterrows():
    bus = str(row.iloc[1])
    tech = str(row.iloc[2])
    cap = float(row.iloc[3])
    key = normalize_key(bus, tech)
    cap_keys[key] = cap
    print(f"CAP: bus='{bus}', tech='{tech}', normalized key='{key}'")

print("\n" + "="*80 + "\n")

# Read energy
print("Reading energy...")
energy = pd.read_csv('../results/thesis-wake-2030-10-northsea-dominant-6h/base-s100000-biasFalse/csvs/nodal_supply_energy.csv', skiprows=4)
offshore_energy = energy[energy.iloc[:, 3].astype(str).str.contains('offwind', case=False, na=False)]
en_keys = {}
for i, row in offshore_energy.head(3).iterrows():
    bus = str(row.iloc[2])
    tech = str(row.iloc[3])
    enrg = float(row.iloc[4])
    key = normalize_key(bus, tech)
    en_keys[key] = enrg
    print(f"ENERGY: bus='{bus}', tech='{tech}', normalized key='{key}'")

print("\n" + "="*80 + "\n")

print(f"CF keys: {len(cf_keys)}")
print(f"Cap keys: {len(cap_keys)}")
print(f"Energy keys: {len(en_keys)}")

print("\nKey matching:")
for key in list(cf_keys.keys())[:3]:
    print(f"\n  key='{key}'")
    print(f"    in cf_keys: {key in cf_keys}")
    print(f"    in cap_keys: {key in cap_keys}")
    print(f"    in en_keys: {key in en_keys}")
