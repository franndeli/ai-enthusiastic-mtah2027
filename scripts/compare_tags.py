"""Aggregate check of decoder knobs on the public window (coordinator experiment ledger helper)."""
import sys, warnings, json, copy
sys.path.insert(0, '.'); sys.path.insert(0, 'evaluation')
import numpy as np, possession_metrics as pm
from run import label_codes_from_config, load_config, DEFAULT_CONFIG
from possession import io
warnings.simplefilter('ignore')
base = load_config(DEFAULT_CONFIG); match = io.load_match(); gt = io.load_public_gt(); tags = io.load_smart_tagging()
runs=[(a,b+1) for _,a,b in match.halves]
def report(name, config):
    codes = label_codes_from_config(config, match, tags)
    r = pm.evaluate(gt, codes); st=pm.possession_stats(codes, runs)
    print(f"{name:<36} spell {r['spell_f1_mean']:.4f} macro {r['macro_f1']:.4f} pub poss {r['n_pred_spells']} full poss {st['n_spells']} dur {st['mean_duration_s']:.1f}")
for disc in (0.0, 0.5, 0.8):
    for md in (1, 2):
        c = copy.deepcopy(base); c['decoder']['restart_discount'] = disc; c['decoder']['min_duration_frames'] = md
        report(f"discount {disc} min_dur {md}", c)
for costs in ((8,8,8), (12,3,12), (10,6,6)):
    c = copy.deepcopy(base); c['decoder']['switch_cost'] = dict(zip(('team_switch','to_dead','from_dead'), map(float, costs)))
    report(f"costs {costs}", c)
