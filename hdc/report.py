"""Build reports/hdc.html from output/hdc/results.json and curves.csv."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tfs.config import load_config, resolve


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float):
        return None if not np.isfinite(x) else round(x, 6)
    if isinstance(x, (np.floating, np.integer)):
        return clean(x.item())
    return x


def main():
    cfg = load_config()
    od = resolve(cfg, "output") / "hdc"
    r = json.loads((od / "results.json").read_text())
    cur = pd.read_csv(od / "curves.csv", index_col=0)
    keep = ["A_pre_tax", "A_short_term", "A_short_term_offset", "hybrid_pre_tax", "spy_pre_tax", "unhedged_pre_tax",
            "high_yield_basket", "B_pre_tax"]
    cur = cur[[c for c in keep if c in cur.columns]]
    wk = cur[pd.to_datetime(cur.index).weekday == 4]                 # Fridays: a weekly series is plenty
    wk = pd.concat([cur.iloc[[0]], wk, cur.iloc[[-1]]]).loc[lambda d: ~d.index.duplicated()]
    curves = {"dates": list(wk.index), **{c: [None if pd.isna(v) else round(float(v) / 1e6, 5) for v in wk[c]] for c in wk.columns}}
    tt = pd.read_csv(od / "trades_model.csv")
    edges = np.arange(-0.06, 0.0401, 0.0025)
    hist = np.histogram(tt.net.clip(-0.06, 0.04), bins=edges)[0].tolist()
    itm = tt[tt.itm_at_exm1.astype(bool)]
    exercise = {k: {"n": int(len(g)), "kept": float(g.net.mean()), "assigned": float(g.assigned_net.mean())}
                for k, g in (("rational", itm[itm.rational.astype(bool)]), ("other", itm[~itm.rational.astype(bool)]))}
    data = clean({**r, "curves": curves, "hist": {"edges": edges.round(4).tolist(), "counts": hist}, "exercise": exercise})
    html = (resolve(cfg, "dashboard_page") / "hdc_template.html").read_text()
    out = resolve(cfg, "reports") / "hdc.html"
    out.write_text(html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    print(out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
