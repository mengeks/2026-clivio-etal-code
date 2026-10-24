"""Paper tables, LaTeX number macros, ablation summaries, replication check.

    python scripts/make_paper_tables.py

Reads   outputs/results/<experiment>/results.csv  (scripts/run_simulation.py)
        outputs/results/legacy/*.csv               (original runs, Sep 2026)
Writes  outputs/tables/paper_numbers.tex           (\\newcommand macros used by the manuscript)
        outputs/tables/tab_<dataset>_seeds.tex     (per-seed breakdowns)
        outputs/tables/ablations.md, ablations.csv (research-note comparisons)
        outputs/tables/replication_check.md        (package rerun vs original CSVs)
and prints the success criteria of configs/default.yaml.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from causal_embedding.utils.config import evaluation_criteria

REPO = Path(__file__).resolve().parents[1]
RES = REPO / 'outputs' / 'results'
TAB = REPO / 'outputs' / 'tables'

# Extra columns reported per research note (besides the two direction metrics).
NOTE_COLUMNS = {
    'A01_treatment_objective': ['score_error_A', 'selected_epoch_A'],
    'A03_initialization': ['beta_init_cos2', 'alpha_init_cos2'],
    'A04_propensity_rank': ['score_error_A'],
    'A05_joint_score_rank': ['selected_epoch_AY', 'score_error_YA'],
    'A06_wa_normalization': ['norm_W_A', 'norm_wA', 'score_error_A'],
    'A08_propensity_hidden': ['score_error_A', 'selected_epoch_A'],
    'A09_h0_construction': ['h0_per_sample_cos2', 'h0_difference_ctrl_cos2',
                            'h0_difference_all_cos2', 'h0_per_sample_min_eig',
                            'h0_difference_ctrl_min_eig', 'h0_difference_all_min_eig'],
    'A10_trace_estimator': ['selected_epoch_AY', 'trace_exact', 'trace_hutch4_sd',
                            'trace_hutch8_sd'],
}


def load(name):
    path = RES / name / 'results.csv'
    return pd.read_csv(path) if path.exists() else None


def med(x):
    return float(np.median(x))


def fmt_range(x, nd=3):
    x = np.asarray(x, dtype=float)
    return f"{np.median(x):.{nd}f} [{x.min():.{nd}f}, {x.max():.{nd}f}]"


def criterion(df, crit):
    b, a = df['beta_sr_cos2'].values, df['alpha_sr_cos2'].values
    n_ok = int(np.sum((b >= crit['floor']) & (a >= crit['floor'])))
    ok = (med(b) >= crit['median_beta'] and med(a) >= crit['median_alpha']
          and n_ok >= crit['min_seeds_above_floor'])
    return ok, n_ok


def per_seed_table(df, groups, caption, label):
    """Booktabs table: one row per seed, (beta, alpha) column pairs per method."""
    spec = 'l' + 'cc' * len(groups)
    top = ' & '.join([''] + [rf'\multicolumn{{2}}{{c}}{{{g}}}' for g, _ in groups]) + r'\\'
    rules = ''.join(rf'\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}' for i in range(len(groups)))
    sub = ' & '.join(['Seed'] + [r'$\hat\beta$ & $\hat\alpha$'] * len(groups)) + r'\\'
    lines = [r'\begin{table}[h]', r'\centering', rf'\caption{{{caption}}}',
             rf'\label{{{label}}}', r'\small', rf'\begin{{tabular}}{{{spec}}}', r'\toprule',
             top, rules, sub, r'\midrule']
    cols = [c for _, pair in groups for c in pair]
    for _, r in df.sort_values('seed').iterrows():
        lines.append(' & '.join([str(int(r['seed']))] + [f"{r[c]:.3f}" for c in cols]) + r'\\')
    lines += [r'\midrule',
              ' & '.join(['Median'] + [f"{med(df[c]):.3f}" for c in cols]) + r'\\',
              r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def main_tables(macros, report):
    crit = evaluation_criteria()
    names = {'linear_rank1': 'Lin', 'nonlinear_rank1': 'Nl'}
    for ds, tag in names.items():
        df = load(ds)
        if df is None:
            report.append(f"- {ds}: no results yet")
            continue
        ok, n_ok = criterion(df, crit[ds])
        c = crit[ds]
        report.append(f"- {ds}: beta {fmt_range(df['beta_sr_cos2'])}, alpha "
                      f"{fmt_range(df['alpha_sr_cos2'])}, seeds >= {c['floor']}: "
                      f"{n_ok}/{len(df)} -> {'PASS' if ok else 'FAIL'}")
        macros.update({
            f'{tag}NSeeds': str(len(df)),
            f'{tag}SRBetaMed': f"{med(df['beta_sr_cos2']):.3f}",
            f'{tag}SRAlphaMed': f"{med(df['alpha_sr_cos2']):.3f}",
            f'{tag}SRSeedsAboveFloor': str(n_ok),
            f'{tag}InitBetaMed': f"{med(df['beta_init_cos2']):.3f}",
            f'{tag}InitAlphaMed': f"{med(df['alpha_init_cos2']):.3f}",
        })
        for m in ('save', 'sir', 'lasso'):
            macros[f'{tag}{m.capitalize()}BetaMed'] = f"{med(df[f'beta_{m}_cos2']):.3f}"
            macros[f'{tag}{m.capitalize()}AlphaMed'] = f"{med(df[f'alpha_{m}_cos2']):.3f}"
        groups = [('Score ratio', ('beta_sr_cos2', 'alpha_sr_cos2'))]
        groups += [(lab, (f'beta_{m}_cos2', f'alpha_{m}_cos2'))
                   for m, lab in (('save', 'SAVE'), ('sir', 'SIR'), ('lasso', 'LASSO'))]
        caption = (r"Per-seed $\cos^2$ between estimated and true directions, "
                   + ('linear' if ds == 'linear_rank1' else 'nonlinear')
                   + r" rank-one model. Source: \texttt{outputs/results/"
                   + ds.replace('_', r'\_') + r"/results.csv}.")
        (TAB / f'tab_{ds}_seeds.tex').write_text(
            per_seed_table(df, groups, caption, f'tab:{ds}_seeds'))

    a03 = load('A03_initialization')
    if a03 is not None:
        for ds, tag in names.items():
            g = a03[(a03.dataset == ds) & (a03.arm == 'random')]
            if len(g):
                macros[f'{tag}RandomBetaMed'] = f"{med(g['beta_sr_cos2']):.3f}"
                macros[f'{tag}RandomAlphaMed'] = f"{med(g['alpha_sr_cos2']):.3f}"
                macros[f'{tag}RandomNSeeds'] = str(len(g))


def main_hashes():
    """Config hash of each dataset's main configuration."""
    from causal_embedding.utils.config import config_hash, expand_experiment
    out = {}
    for ds in ('linear_rank1', 'nonlinear_rank1'):
        _, jobs, _ = expand_experiment(REPO / 'configs' / 'experiments' / f'{ds}.yaml')
        out[ds] = config_hash(jobs[0][2])
    return out


def paired(g_arm, g_ref, col):
    """Median paired difference (arm - main) and #seeds where arm > main."""
    m = g_arm[['seed', col]].merge(g_ref[['seed', col]], on='seed', suffixes=('', '_ref'))
    diff = m[col] - m[f'{col}_ref']
    return f"{np.median(diff):+.3f} ({int((diff > 0).sum())}/{len(diff)})"


def ablation_tables(suffix=''):
    md, rows, ref_hash = [], [], main_hashes()
    for name, extra in NOTE_COLUMNS.items():
        df = load(name + suffix)
        if df is None:
            md.append(f"## {name}{suffix}\n\n_no results yet_\n")
            continue
        cols = ['beta_sr_cos2', 'alpha_sr_cos2'] + [c for c in extra if c in df]
        md.append(f"## {name}{suffix}\n")
        md.append('| dataset | arm | n | ' + ' | '.join(cols)
                  + ' | Δβ vs main (wins) | Δα vs main (wins) |')
        md.append('|' + '---|' * (5 + len(cols)))
        for (ds, arm), g in df.groupby(['dataset', 'arm'], sort=False):
            ref = df[(df.dataset == ds) & (df.config_hash == ref_hash[ds])]
            is_main = bool(len(g)) and g.config_hash.iloc[0] == ref_hash[ds]
            nd = {c: (1 if 'epoch' in c else 3) for c in cols}
            if is_main or ref.empty:
                cmp_b = cmp_a = 'main' if is_main else '-'
            else:
                cmp_b, cmp_a = paired(g, ref, 'beta_sr_cos2'), paired(g, ref, 'alpha_sr_cos2')
            md.append(f"| {ds} | {arm}{' (main)' if is_main else ''} | {len(g)} | "
                      + ' | '.join(fmt_range(g[c], nd[c]) for c in cols)
                      + f" | {cmp_b} | {cmp_a} |")
            for c in cols:
                rows.append(dict(note=name, seeds=suffix or '_5seeds', dataset=ds, arm=arm,
                                 is_main=is_main, metric=c, n=len(g), median=med(g[c]),
                                 min=g[c].min(), max=g[c].max()))
        md.append('')
    header = (f"# Research-note comparisons{' (seeds 1-20)' if suffix else ' (seeds 1-5)'}\n\n"
              "Median [min, max] over seeds. Δ = median over seeds of (arm − main) on the same "
              "data; wins = seeds where the arm beats the main configuration. Generated by "
              f"scripts/make_paper_tables.py from outputs/results/A*{suffix}/results.csv.\n")
    (TAB / f'ablations{suffix}.md').write_text(header + '\n' + '\n'.join(md))
    pd.DataFrame(rows).to_csv(TAB / f'ablations{suffix}.csv', index=False)


def replication_check():
    pairs = [
        ('linear_rank1', 'sr_nn_linear_stable.csv',
         lambda d: d[d['hidden_A'] == '(8,)'], 'beta_final_cos2', 'alpha_final_cos2'),
        ('nonlinear_rank1', 'sr_nn_nonlinear_rank1_stable.csv',
         lambda d: d[d['init_mode'] == 'save'], 'beta_sr_cos2', 'alpha_sr_cos2'),
    ]
    lines = ["# Replication check: package pipeline vs original stable scripts\n",
             "Original: outputs/results/legacy (Sep 2026, Aries GPU, unseeded TensorFlow).",
             "Rerun: outputs/results/<dataset>/results.csv (this repository, seeded).",
             "The original scripts did not seed TensorFlow or the mini-batch order, so",
             "per-seed values are not expected to match; data, warm starts and baselines",
             "are deterministic and should match exactly.\n",
             "| dataset | quantity | original | rerun |", "|---|---|---|---|"]
    for ds, legacy_csv, sel, bcol, acol in pairs:
        new = load(ds)
        old_path = RES / 'legacy' / legacy_csv
        if new is None or not old_path.exists():
            continue
        old = sel(pd.read_csv(old_path))
        lines.append(f"| {ds} | SR cos2 beta | {fmt_range(old[bcol])} | {fmt_range(new['beta_sr_cos2'])} |")
        lines.append(f"| {ds} | SR cos2 alpha | {fmt_range(old[acol])} | {fmt_range(new['alpha_sr_cos2'])} |")
        for col in ('beta_init_cos2', 'alpha_init_cos2'):
            if col in old:
                lines.append(f"| {ds} | {col} (original init) | {fmt_range(old[col])} | "
                             f"{fmt_range(new[col])} |")
    for ds in ('linear_rank1', 'nonlinear_rank1'):
        d20 = load(ds + '_20seeds')
        if d20 is not None:
            lines.append(f"| {ds} | package, seeds 1-{len(d20)}, beta | - | {fmt_range(d20['beta_sr_cos2'])} |")
            lines.append(f"| {ds} | package, seeds 1-{len(d20)}, alpha | - | {fmt_range(d20['alpha_sr_cos2'])} |")
    rerun_dir = RES / 'exploration'
    for ds, legacy_csv, sel, bcol, acol in pairs:
        path = rerun_dir / legacy_csv
        if path.exists():
            d = sel(pd.read_csv(path))
            lines.append(f"| {ds} | original script rerun here, beta | - | {fmt_range(d[bcol])} |")
            lines.append(f"| {ds} | original script rerun here, alpha | - | {fmt_range(d[acol])} |")
    (TAB / 'replication_check.md').write_text('\n'.join(lines) + '\n')


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    macros, report = {}, []
    main_tables(macros, report)
    lines = ['% Generated by scripts/make_paper_tables.py -- do not edit by hand.',
             '% Sources: outputs/results/{linear_rank1,nonlinear_rank1,A03_initialization}/results.csv']
    lines += [rf'\newcommand{{\{k}}}{{{v}}}' for k, v in sorted(macros.items())]
    (TAB / 'paper_numbers.tex').write_text('\n'.join(lines) + '\n')
    ablation_tables()
    ablation_tables('_20seeds')
    replication_check()
    print('Success criteria (configs/default.yaml: evaluation):')
    print('\n'.join(report))
    print(f'wrote {TAB.relative_to(REPO)}/: paper_numbers.tex, tab_*_seeds.tex, '
          'ablations{,_20seeds}.{md,csv}, replication_check.md')


if __name__ == '__main__':
    main()
