import pandas as pd

# Read CFs
cfs = pd.read_csv('../results/thesis-wake-2030-10-northsea-dominant-6h/base-s100000-biasFalse/csvs/nodal_cfs.csv', skiprows=4)
print('nodal_cfs shape:', cfs.shape)
offshore_cfs = cfs[cfs.iloc[:, 2].astype(str).str.contains('offwind', case=False, na=False)]
print('Offshore CF rows:', len(offshore_cfs))
print('First 3:')
for i, row in offshore_cfs.head(3).iterrows():
    bus = row.iloc[1]
    tech = row.iloc[2]
    cf = row.iloc[3]
    key = f"{bus}_{tech}"
    print(f"  {key}: cf={cf}")

print('\n' + '='*80 + '\n')

# Read capacities
caps = pd.read_csv('../results/thesis-wake-2030-10-northsea-dominant-6h/base-s100000-biasFalse/csvs/nodal_capacities.csv', skiprows=4)
print('nodal_capacities shape:', caps.shape)
offshore_caps = caps[caps.iloc[:, 2].astype(str).str.contains('offwind', case=False, na=False)]
print('Offshore capacity rows:', len(offshore_caps))
print('First 3:')
for i, row in offshore_caps.head(3).iterrows():
    bus = row.iloc[1]
    tech = row.iloc[2]
    cap = row.iloc[3]
    key = f"{bus}_{tech}"
    print(f"  {key}: cap={cap}")

print('\n' + '='*80 + '\n')

# Read energy
energy = pd.read_csv('../results/thesis-wake-2030-10-northsea-dominant-6h/base-s100000-biasFalse/csvs/nodal_supply_energy.csv', skiprows=4)
print('nodal_supply_energy shape:', energy.shape)
print('Columns:', energy.columns.tolist())
print('\nFirst 10 rows:')
print(energy.head(10))

offshore_energy = energy[energy.iloc[:, 3].astype(str).str.contains('offwind', case=False, na=False)]
print('\nOffshore energy rows:', len(offshore_energy))
print('First 3:')
for i, row in offshore_energy.head(3).iterrows():
    bus = row.iloc[2]
    tech = row.iloc[3]
    enrg = row.iloc[4]
    key = f"{bus}_{tech}"
    print(f"  {key}: energy={enrg}")
