"""Reproducible postprocessing of the fixed 24 fits. Never trains or loads pickle.
Run from project root: python risk_6306_levels/review_6306_thesis.py
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from common import PROJECT, load_data, baselines, metrics, alarm_weights, digest, save_json
from run_6306_levels import select_period, identity, code_identity, LEVELS

SEEDS = (42, 43, 44)
BUDGETS = (5, 10, 20)
LABELS = {"training_climatology": "Training climatology",
          "history_high_occupancy": "Recent exceedance occupancy",
          "history_count": "Recent count", "history_energy": "Recent energy",
          "history_count_energy_equal": "Count + energy",
          "ordered_low": "Single low", "ordered_high": "Single high",
          "ordered_dual": "Dual ordered", "shuffled_dual": "Dual shuffled"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def csv(out, name, rows):
    pd.DataFrame(rows).to_csv(out / name, index=False, encoding="utf-8-sig")


def coverage_curves(scores, events):
    """Expected hit count at EVERY integer area budget, with exact fractional ties."""
    scores, events = np.asarray(scores), np.asarray(events)
    if scores.shape != events.shape or scores.ndim != 3:
        raise ValueError("Expected matching [day,y,x] arrays")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Invalid scores")
    g = scores.shape[1] * scores.shape[2]
    result = np.zeros((len(scores), g + 1))
    for d, (s, n) in enumerate(zip(scores, events)):
        order = np.argsort(-s.ravel(), kind="stable")
        values, counts = s.ravel()[order], n.ravel()[order]
        starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
        ends = np.r_[starts[1:], g]
        total = 0.
        for start, end in zip(starts, ends):
            mass = float(counts[start:end].sum())
            result[d, start:end+1] = total + mass * np.arange(end-start+1)/(end-start)
            total += mass
    return result


def block_totals(dates, values, block_days):
    """Disjoint calendar blocks. Excluded dates are absent, never imputed as monitored zeros."""
    dates = pd.DatetimeIndex(dates)
    labels = ((dates - dates.min()).days // block_days).to_numpy()
    return np.stack([values[labels == b].sum(axis=0) for b in np.unique(labels)])


def paired_bootstrap(parts, block_days, repeats, seed):
    """Parts are distinct A/B strata; columns = hit difference, observed event count."""
    rng = np.random.default_rng(seed)
    samples = np.zeros((repeats, 2))
    point = np.zeros(2)
    for dates, difference, events in parts:
        values = np.column_stack([difference, events])
        point += values.sum(axis=0)
        blocks = block_totals(dates, values, block_days)
        indices = rng.integers(0, len(blocks), size=(repeats, len(blocks)))
        samples += blocks[indices].sum(axis=1)
    valid = samples[:, 1] > 0
    ratios = samples[valid, 0] / samples[valid, 1]
    lo, hi = np.quantile(ratios, [.025, .975]) if len(ratios) else (np.nan, np.nan)
    return dict(delta=point[0]/point[1] if point[1] else np.nan,
                ci_low=lo, ci_high=hi, valid_replicates=int(valid.sum()),
                zero_event_replicates=int((~valid).sum()), events=int(point[1]))


def assert_metric_match(actual, old, context):
    for key in ("days", "events", "positive_cells", "cell_average_precision"):
        if not np.isclose(actual[key], old[key], rtol=1e-8, atol=1e-10):
            raise ValueError(f"Metric mismatch: {context} {key}: {actual[key]} vs {old[key]}")
    for b in BUDGETS:
        for key in ("event_recall", "positive_cell_recall", "mean_alarm_area"):
            if not np.isclose(actual[f"top{b}"][key], old[f"top{b}"][key], rtol=1e-8, atol=1e-10):
                raise ValueError(f"Metric mismatch: {context} top{b} {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT/"input_6306_levels")
    parser.add_argument("--runs", type=Path, default=PROJECT/"output_6306_levels")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be >=100")
    out = args.output or args.runs/("thesis_review_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    cfg, arr, samples, _ = load_data(args.data)
    data_id = identity(args.data)
    core_id = code_identity()
    cache, source_files, checks, results, curves, cells, daily = {}, {}, [], [], [], [], []
    g = int(np.prod(arr["target"].shape[-2:]))
    observed_counts = []
    count_runs = 0

    def record(path):
        source_files[str(Path(path).resolve())] = digest(path)

    def add(period, level, method, seed, scores, probability, old, folder):
        ids = cache[period]["ids"]
        y, n = arr["target"][ids, level], arr["event_counts"][ids, level]
        dates = arr["dates"][ids].astype(str)
        actual = metrics(scores, y, n, probability)
        assert_metric_match(actual, old, f"{folder.name}/{LEVELS[level]}/{method}")
        hits = coverage_curves(scores, n)
        for b in BUDGETS:
            k = int(np.floor(g*b/100))
            np.testing.assert_allclose(hits[:, k].sum()/n.sum(),
                                       actual[f"top{b}"]["event_recall"], rtol=1e-10, atol=1e-10)
        row = dict(period=period, level=LEVELS[level], method=method, seed=seed,
                   days=len(ids), events=int(n.sum()), positive_cells=int(y.sum()),
                   AP=actual["cell_average_precision"], brier=actual.get("brier"),
                   log_loss=actual.get("log_loss"))
        for b in BUDGETS:
            row[f"hits_{b}"] = float(hits[:, int(np.floor(g*b/100))].sum())
            row[f"recall_{b}"] = actual[f"top{b}"]["event_recall"]
            row[f"actual_area_{b}"] = int(np.floor(g*b/100))/g
        results.append(row)
        for k in range(g+1):
            curves.append(dict(period=period, level=LEVELS[level], method=method, seed=seed,
                               cells_selected=k, actual_area=k/g, events=int(n.sum()),
                               expected_hits=float(hits[:, k].sum()),
                               event_recall=float(hits[:, k].sum()/n.sum())))
        for d, date in enumerate(dates):
            r = dict(period=period, level=LEVELS[level], method=method, seed=seed,
                     date=date, events=int(n[d].sum()))
            for b in BUDGETS:
                r[f"hits_{b}"] = float(hits[d, int(np.floor(g*b/100))])
            daily.append(r)
            for iy, ix in zip(*np.nonzero(n[d])):
                s = scores[d]
                above = int((s > s[iy, ix]).sum())
                tied = int((s == s[iy, ix]).sum())
                c = dict(period=period, date=date, level=LEVELS[level], method=method,
                         seed=seed, iy=int(iy), ix=int(ix), events=int(n[d, iy, ix]),
                         rank_best=above+1, rank_worst=above+tied,
                         score=float(s[iy, ix]))
                for b in BUDGETS:
                    k = int(np.floor(g*b/100))
                    c[f"coverage_{b}"] = float(np.clip((k-above)/tied, 0, 1))
                cells.append(c)
        cache[period]["maps"].setdefault(level, {})[(method, seed)] = scores
        cache[period]["hits"].setdefault(level, {})[(method, seed)] = hits
        # Also check stored day-by-day hits, including excluded-date alignment.
        frame = pd.read_csv(folder/"evaluation/daily_metrics.csv")
        original_method = "convlstm" if method.startswith(("ordered_", "shuffled_")) else method
        f = frame[frame.level.eq(LEVELS[level]) & frame.method.eq(original_method)]
        if f.date.tolist() != dates.tolist():
            raise ValueError(f"Stored date mismatch: {folder}")
        np.testing.assert_array_equal(f.events.to_numpy(), n.sum((1, 2)))
        for b in BUDGETS:
            np.testing.assert_allclose(f[f"hit_events_top{b}"], hits[:, int(np.floor(g*b/100))],
                                       rtol=1e-8, atol=1e-10)
        checks.append(dict(run=folder.name, level=LEVELS[level], method=method,
                           status="AP, coverage, daily dates/counts/hits matched"))

    for period in ("A", "B"):
        train_ids, ids = select_period(samples, period)
        cache[period] = dict(ids=ids, maps={}, hits={})
        observed_counts.append(dict(period=period, start=str(arr["dates"][ids[0]]),
                                    end=str(arr["dates"][ids[-1]]), days=len(ids),
                                    events_low=int(arr["event_counts"][ids, 0].sum()),
                                    events_high=int(arr["event_counts"][ids, 1].sum())))
        baseline = args.runs/f"baselines_{period}"
        basecfg = read_json(baseline/"run_config.json")
        if basecfg["identity"] != data_id or basecfg["code_identity"] != core_id:
            raise ValueError(f"Baseline data/code identity mismatch: {period}")
        base_old = read_json(baseline/"evaluation/summary.json")
        for name in ("run_config.json", "completed.json", "evaluation/summary.json", "evaluation/daily_metrics.csv"):
            record(baseline/name)
        for level in range(2):
            single = dict(arr, target=arr["target"][:, level], event_counts=arr["event_counts"][:, level])
            for method, (score, prob) in baselines(single, ids, train_ids, cfg["seq_len"]).items():
                add(period, level, method, -1, score, prob, base_old[LEVELS[level]][method], baseline)
        for order, task in (("ordered", "low"), ("ordered", "high"),
                            ("ordered", "dual"), ("shuffled", "dual")):
            for seed in SEEDS:
                folder = args.runs/f"{'train' if order == 'ordered' else 'shuffled'}_{period}_{task}_all_seed{seed}"
                rc = read_json(folder/"run_config.json")
                train = read_json(folder/"training_summary.json")
                complete = read_json(folder/"completed.json")
                expected_code = dict(core_id)
                if order == "shuffled":
                    expected_code["run_6306_shuffled.py"] = digest(Path(__file__).parent/"run_6306_shuffled.py")
                if rc["identity"] != data_id or rc["code_identity"] != expected_code:
                    raise ValueError(f"Run data/code mismatch: {folder.name}")
                if (rc["period"], rc["task"], rc["seed"], rc["epochs"], rc["input_mode"]) != (period, task, seed, 15, "all"):
                    raise ValueError(f"Unexpected fixed experiment configuration: {folder.name}")
                if complete.get("action") != "train" or complete.get("period") != period:
                    raise ValueError(f"Unexpected completion record: {folder.name}")
                np.testing.assert_array_equal(train["train_ids"], train_ids)
                np.testing.assert_array_equal(train["eval_ids"], ids)
                old = read_json(folder/"evaluation/summary.json")
                with np.load(folder/"evaluation/predictions.npz", allow_pickle=False) as z:
                    levels = [LEVELS.index(str(s)) for s in z["level_names"]]
                    expected_levels = {"low": [0], "high": [1], "dual": [0, 1]}[task]
                    if levels != expected_levels:
                        raise ValueError(f"Output levels mismatch: {folder.name}")
                    np.testing.assert_array_equal(z["dates"].astype(str), arr["dates"][ids].astype(str))
                    np.testing.assert_array_equal(z["targets"], arr["target"][ids][:, levels])
                    np.testing.assert_array_equal(z["event_counts"], arr["event_counts"][ids][:, levels])
                    np.testing.assert_array_equal(z["x_edges"], arr["x_edges"])
                    np.testing.assert_array_equal(z["y_edges"], arr["y_edges"])
                    scores = z["scores"].copy()
                if scores.shape != arr["target"][ids][:, levels].shape:
                    raise ValueError("Prediction shape mismatch")
                if len(levels) == 2 and np.any(scores[:, 1] > scores[:, 0]+1e-7):
                    raise ValueError("Non-nested scores")
                for j, level in enumerate(levels):
                    add(period, level, f"{order}_{task}", seed, scores[:, j], True,
                        old[LEVELS[level]]["convlstm"], folder)
                for name in ("run_config.json", "training_summary.json", "completed.json",
                             "evaluation/summary.json", "evaluation/daily_metrics.csv", "evaluation/predictions.npz"):
                    record(folder/name)
                count_runs += 1
                print("Verified:", folder.name, flush=True)
    if count_runs != 24:
        raise ValueError("Expected exactly 24 fixed fits")
    out.mkdir(parents=True)
    (out/"figures").mkdir()
    (out/"cases").mkdir()
    for name in ("config.json", "dataset.npz", "sample_manifest.csv", "daily_audit.csv"):
        record(args.data/name)
    record(Path(__file__))
    csv(out, "verification.csv", checks)
    csv(out, "period_counts.csv", observed_counts)
    csv(out, "metrics_per_run.csv", results)
    csv(out, "daily_hits.csv", daily)
    csv(out, "positive_cell_ranks.csv", cells)
    frame = pd.DataFrame(results)
    # Combined coverage sums hits/events per seed. Pooled AP is intentionally not computed.
    combined = []
    for (level, method, seed), group in frame.groupby(["level", "method", "seed"]):
        if set(group.period) != {"A", "B"}:
            raise ValueError("Missing A/B pair")
        r = dict(period="A+B", level=level, method=method, seed=seed,
                 events=int(group.events.sum()), days=int(group.days.sum()), AP=np.nan)
        for b in BUDGETS:
            r[f"hits_{b}"] = group[f"hits_{b}"].sum()
            r[f"recall_{b}"] = r[f"hits_{b}"]/r["events"]
        combined.append(r)
    all_metrics = pd.concat([frame, pd.DataFrame(combined)], ignore_index=True)
    csv(out, "metrics_with_combined.csv", all_metrics)
    summary = all_metrics.groupby(["period", "level", "method"])[
        ["AP", "recall_5", "recall_10", "recall_20"]].agg(["count", "mean", "min", "max"])
    summary.to_csv(out/"seed_summary.csv", encoding="utf-8-sig")
    cf = pd.DataFrame(curves)
    for (level, method, seed, k), group in cf.groupby(["level", "method", "seed", "cells_selected"]):
        curves.append(dict(period="A+B", level=level, method=method, seed=seed,
                           cells_selected=k, actual_area=k/g, events=int(group.events.sum()),
                           expected_hits=group.expected_hits.sum(),
                           event_recall=group.expected_hits.sum()/group.events.sum()))
    csv(out, "area_coverage_all_cells.csv", curves)
    uncertainty = []
    for level in range(2):
        single = "ordered_low" if level == 0 else "ordered_high"
        comparisons = [(m, ref) for m in (single, "ordered_dual")
                       for ref in ("history_count", "history_energy", "history_count_energy_equal")]
        comparisons += [("ordered_dual", single), ("ordered_dual", "shuffled_dual")]
        for scope in ("A", "B", "A+B"):
            periods = ("A", "B") if scope == "A+B" else (scope,)
            for left, right in comparisons:
                for b in BUDGETS:
                    parts = []
                    k = int(np.floor(g*b/100))
                    for period in periods:
                        item = cache[period]
                        def seed_mean(method):
                            return np.mean([v[:, k] for (m, s), v in item["hits"][level].items()
                                            if m == method], axis=0)
                        ids = item["ids"]
                        parts.append((arr["dates"][ids], seed_mean(left)-seed_mean(right),
                                      arr["event_counts"][ids, level].sum((1, 2))))
                    for block_days in (7, 14, 28):
                        stat = paired_bootstrap(parts, block_days, args.bootstrap, 6306)
                        uncertainty.append(dict(period=scope, level=LEVELS[level], left=left, right=right,
                                                budget=b, block_days=block_days, **stat))
    csv(out, "paired_block_bootstrap.csv", uncertainty)
    make_figures(out, pd.DataFrame(curves), all_metrics, pd.DataFrame(uncertainty))
    make_cases(out, cache, arr)
    report(out, all_metrics, pd.DataFrame(uncertainty), count_runs)
    save_json(out/"provenance.json", dict(runs_verified=count_runs, source_sha256=source_files,
              data_identity=data_id, core_code_identity=core_id, bootstrap=args.bootstrap,
              bootstrap_seed=6306, block_days=[7, 14, 28], primary_block_days=14,
              primary_budget_percent=10, supplementary_budgets_percent=[5, 20],
              ties="fractional expected coverage; no arbitrary tie-breaking",
              seeds="mean performance across 3 fits; not ensemble predictions or independent events",
              AP="all date-cell pairs within each period; A+B AP deliberately omitted",
              inference="exploratory, pointwise intervals, conditional on fitted models; no multiplicity correction",
              excluded_dates="not imputed; block totals use only originally evaluated dates",
              case_selection="seed42 fixed; all high-event dates; first low-event gain/loss vs energy in each period"))
    save_json(out/"completed.json", dict(status="postprocessing_complete", fits=24))
    print("Completed postprocessing:", out, flush=True)


def make_figures(out, curves, frame, uncertainty):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    for level in LEVELS:
        for period in ("A", "B", "A+B"):
            part = curves[curves.level.eq(level) & curves.period.eq(period)]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for method in part.method.unique():
                f = part[part.method.eq(method)]
                p = f.pivot(index="actual_area", columns="seed", values="event_recall").sort_index()
                x = p.index.to_numpy()*100
                mean, lo, hi = p.mean(axis=1).to_numpy()*100, p.min(axis=1).to_numpy()*100, p.max(axis=1).to_numpy()*100
                for ax in axes:
                    line, = ax.step(x, mean, where="post", label=LABELS[method])
                    if p.shape[1] > 1:
                        ax.fill_between(x, lo, hi, alpha=.12, step="post", color=line.get_color())
            for ax, xmax in zip(axes, (20, 100)):
                ax.set(xlabel="Actual daily area (%)", ylabel="Expected event coverage (%)",
                       xlim=(0, xmax), ylim=(0, 101))
                ax.grid(alpha=.2)
            axes[1].legend(fontsize=7, loc="lower right")
            fig.suptitle(f"{period} / {level}: mean and seed range (not confidence intervals)")
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(out/"figures"/f"coverage_{period}_{level}.{ext}", dpi=180)
            plt.close(fig)
    for level in LEVELS:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, period in zip(axes, ("A", "B")):
            f = frame[frame.period.eq(period) & frame.level.eq(level)]
            names = list(f.method.unique())
            for i, method in enumerate(names):
                values = f.loc[f.method.eq(method), "AP"].to_numpy()
                ax.scatter(np.full(len(values), i), values, s=25)
            ax.set_xticks(range(len(names)), [LABELS[n] for n in names], rotation=65, ha="right")
            ax.set(title=period, ylabel="All date-cell AP")
            ax.grid(axis="y", alpha=.2)
        fig.suptitle(f"{level}: every seed shown; baseline once")
        fig.tight_layout()
        fig.savefig(out/"figures"/f"AP_{level}.png", dpi=180)
        plt.close(fig)
    for level in LEVELS:
        f = uncertainty[uncertainty.level.eq(level) & uncertainty.period.eq("A+B")
                        & uncertainty.block_days.eq(14) & uncertainty.budget.eq(5)]
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, r in enumerate(f.itertuples()):
            ax.plot([r.ci_low*100, r.ci_high*100], [i, i], color="steelblue")
            ax.scatter(r.delta*100, i, color="black")
        ax.axvline(0, color="gray", ls="--")
        ax.set_yticks(range(len(f)), [f"{LABELS[r.left]} - {LABELS[r.right]}" for r in f.itertuples()])
        ax.set(xlabel="Coverage difference (percentage points), 95% pointwise interval",
               title=f"A+B {level}: 5% budget, 14-calendar-day blocks")
        fig.tight_layout()
        fig.savefig(out/"figures"/f"paired_difference_{level}.png", dpi=180)
        plt.close(fig)


def make_cases(out, cache, arr):
    import matplotlib.pyplot as plt
    case_rows = []
    for period, item in cache.items():
        ids = item["ids"]
        for level in range(2):
            n = arr["event_counts"][ids, level]
            selected = {}
            if level == 1:
                selected = {int(i): "all high-event dates" for i in np.flatnonzero(n.sum((1, 2)))}
            else:
                k = int(np.floor(n.shape[1]*n.shape[2]*.05))
                diff = (item["hits"][level][("ordered_dual", 42)][:, k]
                        - item["hits"][level][("history_energy", -1)][:, k])
                for mask, reason in ((diff > 1e-9, "first low gain"), (diff < -1e-9, "first low loss")):
                    choices = np.flatnonzero(mask)
                    if len(choices):
                        selected[int(choices[0])] = reason
                if not selected:
                    selected[int(np.flatnonzero(n.sum((1, 2)))[0])] = "first low positive; no gain/loss at 5%"
            methods = [("history_energy", -1),
                       ("ordered_low" if level == 0 else "ordered_high", 42),
                       ("ordered_dual", 42), ("shuffled_dual", 42)]
            xe, ye = arr["x_edges"], arr["y_edges"]
            x, y = xe-xe[0], ye-ye[0]
            for d, reason in sorted(selected.items()):
                date = str(arr["dates"][ids[d]])
                fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
                for ax, (method, seed) in zip(axes.ravel(), methods):
                    s = item["maps"][level][(method, seed)][d]
                    flat = s.ravel()
                    ranks = np.array([((flat < v).sum()+.5*(flat == v).sum())/len(flat) for v in flat]).reshape(s.shape)
                    im = ax.pcolormesh(x, y, ranks, cmap="magma", vmin=0, vmax=1)
                    w = alarm_weights(s, .05)
                    if w.min() < .5 < w.max():
                        ax.contour((x[:-1]+x[1:])/2, (y[:-1]+y[1:])/2, w, levels=[.5], colors="cyan", linewidths=1)
                    iy, ix = np.nonzero(n[d])
                    ax.scatter((x[:-1]+x[1:])[ix]/2, (y[:-1]+y[1:])[iy]/2,
                               marker="x", color="lime", s=50)
                    hits = float((w*n[d]).sum())
                    ax.set(title=f"{LABELS[method]}: hits {hits:.2f}/{n[d].sum():.0f} @5%",
                           xlabel="X offset (m)", ylabel="Y offset (m)", aspect="equal")
                    case_rows.append(dict(period=period, date=date, level=LEVELS[level], reason=reason,
                                          method=method, seed=seed, events=int(n[d].sum()), expected_hits_5=hits))
                fig.colorbar(im, ax=axes.ravel().tolist(), label="Within-day score percentile; not probability")
                fig.suptitle(f"{period} {date} {LEVELS[level]}; seed42; cell-center observations\n"
                             f"Coordinate origin: X={xe[0]:.2f}, Y={ye[0]:.2f}; cyan: 5% area boundary")
                fig.savefig(out/"cases"/f"{period}_{date}_{LEVELS[level]}.png", dpi=160)
                plt.close(fig)
    csv(out, "case_index.csv", case_rows)


def report(out, frame, uncertainty, count_runs, verification=None):
    lines = ["# 6306 固定实验后处理报告", "",
             verification or f"已核验 {count_runs} 次固定训练；本次未训练网络。原有AP、覆盖率、逐日日期/事件数/命中数均复算通过。",
             "", "## 核心结果（网络为三种子均值，括号内为范围）", "",
             "|阶段|能级|方法|AP|5%预算覆盖|10%预算覆盖|",
             "|---|---|---|---|---|---|"]
    for (period, level, method), group in frame.groupby(["period", "level", "method"], sort=False):
        def fmt(col, percent=False):
            v = group[col].dropna()
            if not len(v):
                return "不合并AP"
            scale = 100 if percent else 1
            values = v*scale
            result = f"{values.mean():.2f}%" if percent else f"{values.mean():.5f}"
            if len(values) > 1:
                result += f" ({values.min():.2f}–{values.max():.2f})" if percent else f" ({values.min():.5f}–{values.max():.5f})"
            return result
        lines.append(f"|{period}|{level}|{LABELS[method]}|{fmt('AP')}|{fmt('recall_5', True)}|{fmt('recall_10', True)}|")
    lines += ["", "## 配对差异（A+B，三种子平均表现，14日分块）", "",
              "以下为覆盖率差，单位为百分点；不是三个模型平均预测图的表现。5%、20%为补充；10%仍为原主要预算。",
              "", "|能级|比较|预算|差异|95%区间|无事件重采样次数|",
              "|---|---|---|---|---|---|"]
    f = uncertainty[uncertainty.period.eq("A+B") & uncertainty.block_days.eq(14)]
    for r in f.itertuples():
        lines.append(f"|{r.level}|{LABELS[r.left]} − {LABELS[r.right]}|{r.budget}%|"
                     f"{r.delta*100:.2f}|[{r.ci_low*100:.2f}, {r.ci_high*100:.2f}]|{r.zero_event_replicates}|")
    lines += ["", "## 按阶段计算的结果解读", ""]
    for level in LEVELS:
        single = "ordered_low" if level == LEVELS[0] else "ordered_high"
        for period in ("A", "B"):
            f = frame[frame.period.eq(period) & frame.level.eq(level)]
            energy_ap = f.loc[f.method.eq("history_energy"), "AP"].mean()
            for method in (single, "ordered_dual"):
                vals = f.loc[f.method.eq(method), "AP"]
                direction = "三个种子均高于" if (vals > energy_ap).all() else (
                    "三个种子均低于" if (vals < energy_ap).all() else "不同种子有交叉，不能一致判为高于或低于")
                lines.append(f"- {period} / {level}：{LABELS[method]} AP均值 {vals.mean():.5f}，"
                             f"历史能量AP {energy_ap:.5f}；{direction}历史能量。")
            normal = f.loc[f.method.eq("ordered_dual"), "AP"].mean()
            shuffled = f.loc[f.method.eq("shuffled_dual"), "AP"].mean()
            lines.append(f"- {period} / {level}：正常顺序减打乱顺序的平均AP差为 {normal-shuffled:+.5f}；"
                         "需结合所有阶段和种子判断，不能由单段差异推断稳定顺序收益。")
    selected = uncertainty[uncertainty.period.eq("A+B") & uncertainty.block_days.eq(14)
                           & uncertainty.right.eq("history_energy")]
    for r in selected.itertuples():
        interpretation = ("区间跨越或接触0，尚不足以支持覆盖率稳定增加" if r.ci_low <= 0 <= r.ci_high
                          else ("区间在0以上，但仍属探索性、条件于现有模型的证据" if r.ci_low > 0
                                else "区间在0以下，当前比较指向覆盖减少"))
        lines.append(f"- A+B / {r.level} / {r.budget}%：{LABELS[r.left]}相对历史能量"
                     f"覆盖差 {r.delta*100:+.2f} 个百分点，95%区间"
                     f"[{r.ci_low*100:.2f}, {r.ci_high*100:.2f}]；{interpretation}。")
    lines += ["", "## 可复核的结论与限制", "",
              "- 历史热点基线与网络在相同日期、网格和面积规则下比较；具体优势须分别按阶段、能级、指标判断。",
              "- 联合学习是否帮助：查看 Dual ordered 与 Single low/high 的同阶段AP、种子范围和配对覆盖差。",
              "- 时间顺序是否帮助：查看 Dual ordered 与 Dual shuffled；不能把保留历史集合的打乱试验解释为去掉历史信息。",
              "- 曲线包含0至全部网格的每一个整数面积；不据此重新挑主要预算。5/10/20%实际面积向下取整。",
              "- 相同分数采用分数权重，覆盖是期望覆盖；个别案例命中数可能为小数。",
              "- AP对每段全部日期—网格计算。A+B仅按事件数合并覆盖，不平均AP，也不拼接不同阶段模型分数计算AP。",
              "- 不确定性采用7/14/28自然日非重叠分块；14日为主，分阶段重采样后合并A+B。缺测/排除日期不补为正常监测日。",
              "- 区间针对给定训练模型、已观测目录的配对覆盖差；不包含重新训练的不确定性。三种子展示重复拟合波动，不能扩大独立事件数。",
              "- 区间为探索性逐项百分位区间，未作多重比较校正；高能事件少或全命中时区间可退化为0，不能据此证明等效或总体完美预测。",
              "- 固定每日面积不能证明时间预警有效。概率校准、监测完整性、爆破筛除与工程边界仍需分别核实。",
              "- 案例固定seed42；展示全部高能事件日期，以及每期低阈值相对历史能量首次改善/退步日期。案例用于解释，不替代全量评价。",
              "", "## 文件导航", "",
              "- verification.csv：复算核对记录；provenance.json：输入、代码和预测文件哈希及分析参数。",
              "- metrics_per_run.csv / metrics_with_combined.csv / seed_summary.csv：逐次和种子汇总。",
              "- area_coverage_all_cells.csv / figures/coverage_*：全网格预算曲线。",
              "- paired_block_bootstrap.csv：分块长度敏感性、差异与区间、无事件重采样次数。",
              "- daily_hits.csv / positive_cell_ranks.csv：逐日与阳性单元明细；单元并非独立事件ID。",
              "- case_index.csv / cases/：预定规则选出的案例及同日四方法图。"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
