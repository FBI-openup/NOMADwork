#!/usr/bin/env python3
import json
from pathlib import Path
p = Path('test_inputs_large/outputs/report.json')
if not p.exists():
    p = Path(__file__).resolve().parent / 'test_inputs_large' / 'outputs' / 'report.json'
if not p.exists():
    print('report.json not found at', p)
    raise SystemExit(1)
with open(p,'r',encoding='utf-8') as f:
    data = json.load(f)

rows = []
for entry in data:
    case = entry.get('case')
    summ = entry.get('summary')
    if not summ:
        rows.append({'case':case,'ok':False,'S_total':None,'violations':None})
        continue
    ok = summ.get('ok', False)
    S = summ.get('S_total', 0.0)
    vio = len(summ.get('capacity_violations', [])) if summ.get('capacity_violations') is not None else 0
    rows.append({'case':case,'ok':ok,'S_total':S,'violations':vio})

valid = [r for r in rows if r['S_total'] is not None]
count = len(rows)
count_ok = sum(1 for r in rows if r['ok'])
avg = sum(r['S_total'] for r in valid)/len(valid) if valid else 0.0
mx = max((r['S_total'] for r in valid), default=None)
mn = min((r['S_total'] for r in valid), default=None)

sorted_by_S = sorted([r for r in valid], key=lambda x: (x['S_total'] if x['S_total'] is not None else -1), reverse=True)
print('cases_total:', count)
print('cases_ok:', count_ok)
print('avg_S_total: {:.4f}'.format(avg))
print('max_S_total: {:.4f}'.format(mx) if mx is not None else 'N/A')
print('min_S_total: {:.4f}'.format(mn) if mn is not None else 'N/A')
print('\nTop 5 cases:')
for r in sorted_by_S[:5]:
    print(' ', r['case'], r['S_total'], 'violations=', r['violations'])
print('\nBottom 5 cases:')
for r in sorted_by_S[-5:]:
    print(' ', r['case'], r['S_total'], 'violations=', r['violations'])

