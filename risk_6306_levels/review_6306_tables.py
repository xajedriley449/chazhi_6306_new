"""Text-only numerical review of existing summaries and daily hits; no predictions or training required.
Run: python risk_6306_levels/review_6306_tables.py
"""
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from common import PROJECT, digest, save_json
from review_6306_thesis import LABELS, LEVELS, SEEDS, BUDGETS, read_json, csv, paired_bootstrap, report


def compute(bundle, out, repeats=2000):
    results, daily, checks, sources = [], [], [], {}
    canonical = {}
    for item in bundle:
        folder, summary = item["folder"], item["summary"]
        data = pd.DataFrame(item["daily"])
        period = "A" if folder.startswith("baselines_A") or "_A_" in folder else "B"
        if folder.startswith("baselines"):
            wanted, seed = list(LABELS)[:5], -1
        else:
            fields = folder.split("_")
            task, seed = fields[2], int(fields[-1].replace("seed", ""))
            wanted = ["convlstm"]
        for level, methods in summary.items():
            for method in wanted:
                if method not in methods:
                    continue
                f = data[data.level.eq(level) & data.method.eq(method)].copy()
                f = f.sort_values("date")
                if f.date.duplicated().any() or not len(f):
                    raise ValueError("Empty/duplicate dates: "+folder)
                old = methods[method]
                np.testing.assert_equal(len(f), old["days"])
                np.testing.assert_equal(f.events.sum(), old["events"])
                np.testing.assert_equal(f.positive_cells.sum(), old["positive_cells"])
                name = method if method != "convlstm" else (
                    ("shuffled_" if folder.startswith("shuffled") else "ordered_")+task)
                row = dict(period=period, level=level, method=name, seed=seed, days=len(f),
                           events=int(f.events.sum()), positive_cells=int(f.positive_cells.sum()),
                           AP=old["cell_average_precision"], brier=old.get("brier"), log_loss=old.get("log_loss"))
                for b in BUDGETS:
                    hit = f[f"hit_events_top{b}"].sum()
                    np.testing.assert_allclose(hit/row["events"], old[f"top{b}"]["event_recall"],
                                               rtol=1e-8, atol=1e-10)
                    np.testing.assert_allclose(f[f"area_top{b}"].mean(), old[f"top{b}"]["mean_alarm_area"],
                                               rtol=1e-8, atol=1e-10)
                    row[f"hits_{b}"], row[f"recall_{b}"] = hit, hit/row["events"]
                    row[f"actual_area_{b}"] = f[f"area_top{b}"].mean()
                key = period, level
                reference = canonical.setdefault(key, f[["date", "events", "positive_cells"]].reset_index(drop=True))
                pd.testing.assert_frame_equal(reference, f[["date", "events", "positive_cells"]].reset_index(drop=True),
                                              check_dtype=False)
                results.append(row)
                for r in f.itertuples():
                    daily.append(dict(period=period, level=level, method=name, seed=seed,
                                      date=r.date, events=r.events,
                                      **{f"hits_{b}":getattr(r, f"hit_events_top{b}") for b in BUDGETS}))
                checks.append(dict(run=folder, level=level, method=name,
                                   status="daily totals/coverage/area/date alignment matched; AP read from summary"))
        sources[folder] = {k: v for k, v in item.items() if k.endswith("sha256")}
    frame, days = pd.DataFrame(results), pd.DataFrame(daily)
    if len(frame[frame.method.str.startswith(("ordered_", "shuffled_"))]) != 36:
        raise ValueError("Expected 24 fits, 36 output branches")
    combined = []
    for (level, method, seed), f in frame.groupby(["level", "method", "seed"]):
        if set(f.period) != {"A", "B"}:
            raise ValueError("Missing period")
        row = dict(period="A+B", level=level, method=method, seed=seed,
                   events=int(f.events.sum()), days=int(f.days.sum()), AP=np.nan)
        for b in BUDGETS:
            row[f"hits_{b}"] = f[f"hits_{b}"].sum()
            row[f"recall_{b}"] = row[f"hits_{b}"]/row["events"]
        combined.append(row)
    frame = pd.concat([frame, pd.DataFrame(combined)], ignore_index=True)
    uncertainty = []
    # Average hit count per day across fits. This is not an ensemble score map.
    means = days.groupby(["period", "level", "method", "date"], sort=True)[
        ["events", "hits_5", "hits_10", "hits_20"]].mean().reset_index()
    for level in LEVELS:
        single = "ordered_low" if level == LEVELS[0] else "ordered_high"
        pairs = [(m, ref) for m in (single, "ordered_dual")
                 for ref in ("history_count", "history_energy", "history_count_energy_equal")]
        pairs += [("ordered_dual", single), ("ordered_dual", "shuffled_dual")]
        for scope in ("A", "B", "A+B"):
            for left, right in pairs:
                for b in BUDGETS:
                    parts = []
                    for period in (("A", "B") if scope == "A+B" else (scope,)):
                        f = means[means.period.eq(period) & means.level.eq(level)]
                        l, r = (f[f.method.eq(m)].set_index("date").sort_index() for m in (left, right))
                        if not l.index.equals(r.index):
                            raise ValueError("Nonpaired dates")
                        parts.append((l.index, l[f"hits_{b}"].to_numpy()-r[f"hits_{b}"].to_numpy(),
                                      l.events.to_numpy()))
                    for block in (7, 14, 28):
                        uncertainty.append(dict(period=scope, level=level, left=left, right=right,
                                                budget=b, block_days=block,
                                                **paired_bootstrap(parts, block, repeats, 6306)))
    u = pd.DataFrame(uncertainty)
    out.mkdir(parents=True, exist_ok=False)
    csv(out, "metrics_with_combined.csv", frame)
    csv(out, "paired_block_bootstrap.csv", u)
    csv(out, "daily_hits.csv", days)
    csv(out, "verification.csv", checks)
    frame.groupby(["period", "level", "method"])[["AP", "recall_5", "recall_10", "recall_20"]].agg(
        ["count", "mean", "min", "max"]).to_csv(out/"seed_summary.csv", encoding="utf-8-sig")
    report(out, frame, u, 24, verification=(
        "本报告使用24次真实实验和A/B基线的文本输出。逐日事件数、阳性单元数、覆盖率、面积和评价日期已交叉核对；"
        "AP沿用原summary.json，尚未从完整预测数组重算。本次未训练网络。"))
    path = out/"REPORT.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace("## 文件导航", "## 文件导航（本目录为文本结果分析）")
    text = text[:text.index("- verification.csv：")] + (
        "- verification.csv：文本结果一致性核对；provenance.json：来源哈希和统计设置。\n"
        "- metrics_with_combined.csv / seed_summary.csv：各阶段、能级、方法与种子指标。\n"
        "- paired_block_bootstrap.csv：7/14/28日分块的配对覆盖差及区间。\n"
        "- daily_hits.csv：所有方法逐日命中，保留事件数和种子。\n"
        "- figures/：AP散点、5/10/20%三个预算点、配对覆盖差。三个预算点不代表完整面积曲线。\n"
        "- 完整曲线、空间案例和预测数组复核：运行 review_6306_thesis.py，在独立thesis_review_时间戳目录生成。\n")
    text = text.replace("- 曲线包含0至全部网格的每一个整数面积；不据此重新挑主要预算。5/10/20%实际面积向下取整。",
                        "- 本目录只展示原定5/10/20%三个预算点；完整整数网格曲线由另一脚本生成。")
    text = text.replace("- 案例固定seed42；展示全部高能事件日期，以及每期低阈值相对历史能量首次改善/退步日期。案例用于解释，不替代全量评价。",
                        "- 本目录尚无空间案例图；完整脚本固定seed42生成全部高能日期和低阈值首次改善/退步案例。")
    path.write_text(text, encoding="utf-8")
    save_json(out/"provenance.json", dict(source_sha256=sources, fits=24,
              scope="text outputs only; AP not recomputed; no full-map or label verification",
              bootstrap_repeats=repeats, random_seed=6306, block_days=[7,14,28],
              primary_budget=10, primary_block_days=14, seed_handling="mean daily hits, not ensemble",
              interval="pointwise percentile, exploratory, conditional on fitted models"))
    plots(out, frame, u)
    save_json(out/"completed.json", dict(status="text_results_review_complete", fits=24))
    return frame, u


def plots(out, frame, u):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (out/"figures").mkdir()
    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(out/"figures"/f"{name}.{ext}", dpi=170)
        plt.close(fig)
    for level in LEVELS:
        fig, axes = plt.subplots(1,2,figsize=(12,5))
        for ax, period in zip(axes, ("A","B")):
            f = frame[frame.level.eq(level) & frame.period.eq(period)]
            names = f.method.unique()
            for i, m in enumerate(names):
                values = f.loc[f.method.eq(m),"AP"]
                ax.scatter([i]*len(values),values,s=25)
            ax.set_xticks(range(len(names)),[LABELS[m] for m in names],rotation=65,ha="right")
            ax.set(title=period,ylabel="All date-cell AP")
            ax.grid(axis="y",alpha=.2)
        fig.suptitle(f"{level}: baseline once; every network seed shown")
        save(fig, f"AP_{level}")
        for period in ("A","B","A+B"):
            fig, ax = plt.subplots(figsize=(9,5))
            f = frame[frame.level.eq(level) & frame.period.eq(period)]
            for i, m in enumerate(f.method.unique()):
                sub = f[f.method.eq(m)]
                x = np.array(BUDGETS)+(i-3.5)*.18
                vals = sub[["recall_5","recall_10","recall_20"]].to_numpy()*100
                mean, lo, hi = vals.mean(0), vals.min(0), vals.max(0)
                ax.errorbar(x,mean,yerr=np.stack([mean-lo,hi-mean]),fmt="o",capsize=2,label=LABELS[m])
            ax.set(xlabel="Nominal daily area budget (%)",ylabel="Expected event coverage (%)",
                   title=f"{period} / {level}: three pre-existing budgets; mean and seed range",
                   ylim=(0,103))
            ax.set_xticks(BUDGETS); ax.legend(fontsize=7,loc="lower right"); ax.grid(alpha=.2)
            save(fig,f"budget_points_{period}_{level}")
        for b in BUDGETS:
            f = u[u.level.eq(level) & u.period.eq("A+B") & u.block_days.eq(14) & u.budget.eq(b)]
            fig,ax=plt.subplots(figsize=(10,5))
            for i,r in enumerate(f.itertuples()):
                ax.plot([r.ci_low*100,r.ci_high*100],[i,i],color="steelblue")
                ax.scatter(r.delta*100,i,color="black")
            ax.set_yticks(range(len(f)),[f"{LABELS[r.left]} - {LABELS[r.right]}" for r in f.itertuples()])
            ax.axvline(0,color="gray",ls="--")
            ax.set(xlabel="Coverage difference (percentage points); 95% pointwise interval",
                   title=f"A+B / {level} / {b}% budget: 14-calendar-day blocks")
            save(fig,f"paired_difference_{level}_budget{b}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,default=PROJECT/"output_6306_levels")
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    folders=["baselines_A","baselines_B"]+[
        f"{prefix}_{period}_{task}_all_seed{seed}"
        for period in ("A","B")
        for prefix,task in (("train","low"),("train","high"),("train","dual"),("shuffled","dual"))
        for seed in SEEDS]
    bundle=[]
    for folder in folders:
        base=a.runs/folder/"evaluation"
        summary_path,daily_path=base/"summary.json",base/"daily_metrics.csv"
        bundle.append(dict(folder=folder,summary=read_json(summary_path),
                           daily=pd.read_csv(daily_path).to_dict("records"),
                           summary_sha256=digest(summary_path),daily_sha256=digest(daily_path)))
    out=a.output or a.runs/("thesis_text_review_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    compute(bundle,out)
    print("Completed text result review:",out)


if __name__=="__main__":
    main()
