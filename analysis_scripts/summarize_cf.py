import pandas as pd

df = pd.read_csv('data/wake_extracted/cf_metrics.csv')

print('=' * 80)
print('Offshore Wind Capacity Factor Analysis Summary')
print('=' * 80)
print()
print('Three CF metrics extracted:')
print('  1. Available CF:    Resource availability after wake losses')
print('  2. Dispatch CF:     Actual generation / (capacity × 8760h)')
print('  3. Curtailment CF:  Available CF - Dispatch CF')
print()
print('=' * 80)
print('Statistics by scenario (mean values):')
print('=' * 80)
print()
print(f"{'Scenario':<12} {'Available CF':<14} {'Dispatch CF':<14} {'Curtailment CF':<16} {'Utilization':<12}")
print('-' * 80)

for scenario in ['base', 'standard', 'glaum', 'new_more']:
    s_data = df[df['scenario'] == scenario]
    avail = s_data['available_cf'].mean()
    dispatch = s_data['dispatch_cf'].mean()
    curtail = s_data['curtailment_cf'].mean()
    utilization = (dispatch / avail * 100) if avail > 0 else 0
    print(f"{scenario:<12} {avail:>6.2%}{'':8} {dispatch:>6.2%}{'':8} {curtail:>6.2%}{'':10} {utilization:>5.1f}%")

print()
print('Notes:')
print('  - Utilization = Dispatch CF / Available CF × 100%')
print('  - Values > 100% indicate system optimization effects (storage, transmission)')
print('  - Wake models reduce Available CF, which flows through to lower Dispatch CF')
