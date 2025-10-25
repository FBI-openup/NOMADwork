#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_and_validate.py

批量运行 solver（prior_solver.py）对一组输入文件，并调用 validate_and_score.py 生成报告。

Usage:
  python run_and_validate.py --in-dir test_inputs_large --cases 100 --workers 1

注意：当前实现为串行（workers=1）。可扩展为并行（multiprocessing）如果需要。
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path


def run_case(prior_solver_py, input_path, output_path):
    # Run solver: type input | python prior_solver.py > output
    # Use shell piping via reading the file to avoid shell-specific type command
    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        proc = subprocess.run([sys.executable, prior_solver_py], stdin=fin, stdout=fout, stderr=subprocess.PIPE)
    return proc.returncode, proc.stderr.decode('utf-8')


def validate_case(validator_py, input_path, output_path):
    proc = subprocess.run([sys.executable, validator_py, '--input', input_path, '--output', output_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode, proc.stdout.decode('utf-8'), proc.stderr.decode('utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', required=True)
    p.add_argument('--out-dir', help='where to write outputs', default=None)
    p.add_argument('--cases', type=int, default=None)
    # default solver/validator path: same directory as this script
    base_dir = Path(__file__).resolve().parent
    p.add_argument('--solver', default=str(base_dir / 'prior_solver.py'))
    p.add_argument('--validator', default=str(base_dir / 'validate_and_score.py'))
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob('case_*.txt'))
    if args.cases is not None:
        files = files[:args.cases]

    summary = []
    for pth in files:
        outname = out_dir / (pth.stem + '.out')
        print(f'Running {pth.name} -> {outname.name}')
        code, err = run_case(args.solver, str(pth), str(outname))
        if code != 0:
            print(f'  Solver failed for {pth.name}:', err)
            summary.append({'case': pth.name, 'solver_error': err})
            continue
        # validate
        vcode, vout, verr = validate_case(args.validator, str(pth), str(outname))
        if vcode != 0:
            print(f'  Validator failed for {pth.name}:', verr)
            summary.append({'case': pth.name, 'validator_error': verr})
            continue
        try:
            j = json.loads(vout)
        except Exception as e:
            print('  Failed to parse validator output JSON:', e)
            summary.append({'case': pth.name, 'validator_parse_error': str(e), 'raw': vout})
            continue
        # write per-case summary file
        with open(out_dir / (pth.stem + '.summary.json'), 'w', encoding='utf-8') as f:
            json.dump(j, f, indent=2)
        summary.append({'case': pth.name, 'summary': j})

    # write aggregated report
    with open(out_dir / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('All done. Report:', out_dir / 'report.json')

if __name__ == '__main__':
    main()
