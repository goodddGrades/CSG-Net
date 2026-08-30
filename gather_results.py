import csv
import glob
import os
import re

FIELDS = ['dir', 'exp', 'data', 'svd', 'cl', 'xgb', 'pre', 'p', 'sim',
          'abg', 'cw', 'dss', 'nsr', 'acc', 'recall', 'prec', 'f1',
          'auc', 'aupr', 'pool11_ap', 'pool11_auc',
          'fold11_ap', 'fold11_auc', 'full_ap', 'full_auc']


def parse_dir(name):
    m = re.match(
        r'(.+?)_d(\d+)_svd(\d+)_cl(\d+)_xgb(\d+)_pre([\d.]+)_p(\d+)_sim(\w+)_abg([\d.]+)_([\d.]+)_([\d.]+)_cw([\d.]+)_dss(\d+)_nsr(-?\d+)_',
        name)
    if not m:
        return {}
    exp, data, svd, cl, xgb, pre, p, sim, a, b, g, cw, dss, nsr = m.groups()
    return dict(exp=exp, data=data, svd=svd, cl=cl, xgb=xgb, pre=pre, p=p,
                sim=sim, abg=f'{a}/{b}/{g}', cw=cw, dss=dss, nsr=nsr)


def main():
    rows = []
    for f in sorted(glob.glob('results/*/summary.txt')):
        d = os.path.basename(os.path.dirname(f))
        txt = open(f, encoding='utf-8').read()
        row = {'dir': d, **parse_dir(d)}
        patterns = {
            'acc': r'accuracy: ([\d.]+)',
            'recall': r'recall: ([\d.]+)',
            'prec': r'precision: ([\d.]+)',
            'f1': r'F1: ([\d.]+)',
            'auc': r'AUC: ([\d.]+)',
            'aupr': r'AUPR: ([\d.]+)',
            'pool11_ap': r'1:1 池: AP=([\d.]+)±',
            'pool11_auc': r'1:1 池: AP=[\d.]+±[\d.]+  AUC=([\d.]+)±',
            'fold11_ap': r'逐折重采样池.*?1:1 池: AP=([\d.]+)±',
            'fold11_auc': r'逐折重采样池.*?1:1 池: AP=[\d.]+±[\d.]+  AUC=([\d.]+)±',
            'full_ap': r'全量池: AP=([\d.]+)',
            'full_auc': r'全量池: AP=[\d.]+  AUC=([\d.]+)',
        }
        for k, pat in patterns.items():
            m = re.search(pat, txt)
            row[k] = m.group(1) if m else ''
        rows.append(row)

    with open('experiments_summary.csv', 'w', newline='', encoding='utf-8-sig') as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"{'exp':<16}{'data':<5}{'xgb':<4}{'sim':<9}{'AUC':<8}{'AUPR':<8}{'recall':<8}")
    for r in rows:
        if r.get('exp'):
            print(f"{r['exp']:<16}{r['data']:<5}{r['xgb']:<4}{r['sim']:<9}"
                  f"{r.get('auc', ''):<8}{r.get('aupr', ''):<8}{r.get('recall', ''):<8}")
    print(f"\n共 {len(rows)} 个结果目录, 已写入 experiments_summary.csv")


if __name__ == '__main__':
    main()
