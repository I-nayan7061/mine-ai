# Reports generator script
import os
from pathlib import Path

REPORTS_DIR = Path('mine-ai/reports')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def write_report(filename, content):
    p = REPORTS_DIR / filename
    with open(p, 'w', encoding='utf-8') as f:
        f.write(content.strip() + '\n')
    print(f'Successfully wrote {p}')

print('build_reports.py ready')
