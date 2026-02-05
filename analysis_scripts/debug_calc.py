import pandas as pd
import re

# For one scenario, let's check the numbers
cfs = pd.read_csv('../results/thesis-wake-2030-10-northsea-dominant-6h/base-s100000-biasFalse/csvs/nodal_cfs.csv', skiprows=4)
caps = pd.read_csv('../results/thesis-wake-2030-10-northsea-dominant-6h/base-s100000-biasFalse/csvs/nodal_capacities.csv', skiprows=4)
energy = pd.read_csv('../results/thesis-wake-2030-10-northsea-dominant-6h/base-s100000-biasFalse/csvs/nodal_supply_energy.csv', skiprows=4)

offshore_energy = energy[energy.iloc[:, 3].astype(str).str.contains('offwind', case=False, na=False)]
print('First offshore energy row:')
row = offshore_energy.iloc[0]
bus = row.iloc[2]
tech = row.iloc[3]
enrg = row.iloc[4]
print(f'  Bus: {bus}')
print(f'  Tech: {tech}')
print(f'  Energy: {enrg} MWh')

# Find matching capacity
bus_norm = re.sub(r'_\d+$', '', bus)
print(f'  Normalized bus: {bus_norm}')

offshore_caps = caps[caps.iloc[:, 2].astype(str).str.contains('offwind', case=False, na=False)]
for i, cap_row in offshore_caps.iterrows():
    cap_bus = str(cap_row.iloc[1])
    cap_tech = str(cap_row.iloc[2])
    cap_val = float(cap_row.iloc[3])
    cap_bus_norm = re.sub(r'_\d+$', '', cap_bus)
    if cap_bus_norm == bus_norm and cap_tech == tech:
        print(f'  Found matching capacity: {cap_val} MW')
        print(f'  CF calculation: {enrg} / ({cap_val} * 8760) = {enrg / (cap_val * 8760):.4f}')
        
        # Also check CF from nodal_cfs
        offshore_cfs = cfs[cfs.iloc[:, 2].astype(str).str.contains('offwind', case=False, na=False)]
        for j, cf_row in offshore_cfs.iterrows():
            cf_bus = str(cf_row.iloc[1])
            cf_tech = str(cf_row.iloc[2])
            cf_val = float(cf_row.iloc[3])
            cf_bus_norm = re.sub(r'_\d+$', '', cf_bus)
            if cf_bus_norm == bus_norm and cf_tech == tech:
                print(f'  Available CF from file: {cf_val:.4f}')
                print(f'  Curtailment CF: {cf_val - (enrg / (cap_val * 8760)):.4f}')
                break
        break
