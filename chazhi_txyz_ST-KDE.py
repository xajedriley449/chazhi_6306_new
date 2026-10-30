#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""6306微震：能量分级 -> 北京时间日切片 -> XY能量守恒高斯网格化。

Python >=3.10。安装：pip install numpy pandas openpyxl xlrd matplotlib pillow
默认运行：python chazhi_txyz_ST-KDE.py --input data/6306.csv
不训练模型；不旋转坐标；不加载7303距离协变量；不做逐帧Min-Max/CDF。

使用：
  1. 把脚本放到项目根目录，原始文件放到 data/6306.csv。
  2. 运行 python chazhi_txyz_ST-KDE.py
  3. 如需重跑，指定新的 --output，避免新旧结果混杂。
  4. 源ENERGIA默认按J解释；若源单位为kJ，传 --energy-multiplier 1000。
  5. 默认范围只使用2022-01-01之前的事件，再外扩150m。
     已有矿图边界时可传 --bounds XMIN XMAX YMIN YMAX。
     请在训练前固定空间范围，不根据测试结果反复调整。

读取每天矩阵：
  with np.load("output_6306_daily_xy/daily/2020-02-14.npz") as f:
      frame = f["energy_log10"]     # float32 (2,64,64)，低能/高能两个通道
      joules = f["energy_J"]        # float32 (2,64,64)，平滑后的J/格
      target = frame[1:2]           # 当天高能标签；未来训练须取目标日而非输入日
      raw_grid = f["energy_grid_J"] # float64，未平滑事件能量之和
      counts = f["count_grid"]      # int32，各格各等级事件数
  连续10天frame堆叠可得(10,2,64,64)，下一日frame[1:2]为(1,64,64)。
  本脚本仅准备日帧，不自动认定这些序列可用于训练，不修改旧训练脚本。

主要输出：
  events_clean.csv / source_event_map.csv：清洗事件与原文件行号追溯。
  duplicate_records.csv / rejected_records.csv：去重与拒收记录。
  daily_manifest.csv / daily_event_ids.json：完整日历与每日事件ID。
  out_of_bounds.csv：网格外事件，不夹到网格边缘，不隐瞒丢失能量。
  daily/YYYY-MM-DD.npz：含双通道平滑能量、对数能量、原始网格与占据标签。
  preview/low, preview/high：固定训练期色标PNG，仅用于查看。
  dataset_config.json / summary.json：参数、原文件SHA256与统计。
  monthly_summary.csv / empty_day_runs.json：月度统计与连续无事件时段。

监测状态：
  默认unknown，首末事件时间不等于完整监测覆盖边界。
  --coverage monitoring.csv 可提供date,status两列；status支持
  normal（完整日正常监测）、missing、unknown。未列日期仍为unknown。
  只有normal且没有网格外事件的日帧才标usable_with_confirmed_coverage=True。
  “高能”只是本实验阈值分组，不自动表示已发生冲击地压灾害。

高斯网格化是对事件能量的空间平滑分配，不是应力场或物理传播模拟。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

# 可以在VSCode直接修改这里，也可以用命令行覆盖。
DEFAULT_INPUT = ["data/6306_all.csv"]
DEFAULT_OUTPUT = "input_6306_daily_xy"
THRESHOLD_J = 100_000.0
ENERGY_REFERENCE_J = 100_000.0
GRID_SIZE = 64
SIGMA_M = 50.0                 # 试验平滑带宽，不是危险半径
TRAIN_END_EXCLUSIVE = "2022-01-01"  # 仅此日期之前用于拟合范围和PNG色标
TIMEZONE = "Asia/Shanghai"

ALIASES = {
    "datetime": ["datetime", "日期时间", "发生时间", "事件时间"],
    "date": ["data", "date", "日期"],
    "time": ["czas", "time", "时间"],
    "x": ["x", "x坐标", "坐标x"],
    "y": ["y", "y坐标", "坐标y"],
    "z": ["z", "z坐标", "坐标z"],
    "energy": ["energia", "energy_j", "energy", "e", "能量", "能量(j)", "能量（j）"],
}


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def read_table(path, sheet=0, header=0):
    """按文件签名读Excel，避免把伪装成CSV的xlsx当文本读取。"""
    path = Path(path)
    raw = path.read_bytes()
    if raw.startswith(b"PK\x03\x04"):
        return pd.read_excel(io.BytesIO(raw), engine="openpyxl", sheet_name=sheet, header=header)
    if raw.startswith(b"\xd0\xcf\x11\xe0"):
        return pd.read_excel(io.BytesIO(raw), engine="xlrd", sheet_name=sheet, header=header)
    errors = []
    for encoding in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding, sep=None, engine="python", header=header)
        except (UnicodeError, pd.errors.ParserError) as exc:
            errors.append(str(exc))
    raise ValueError(f"无法读取 {path}：{errors}")


def resolve_columns(df, explicit):
    names = {str(c).strip().lower(): c for c in df.columns}
    if len(names) != len(df.columns):
        raise ValueError("存在重复列名，请先修正表头。")
    result = {}
    for key, aliases in ALIASES.items():
        if key in explicit:
            if explicit[key] not in df.columns:
                raise ValueError(f"指定列不存在：{key}={explicit[key]}")
            result[key] = explicit[key]
        else:
            matches = [names[a] for a in aliases if a in names]
            if len(matches) > 1:
                raise ValueError(f"{key}匹配到多个列{matches}，请用--columns明确指定。")
            result[key] = matches[0] if matches else None
    required = ["x", "y", "energy"]
    if result["datetime"] is None:
        required += ["date", "time"]
    missing = [k for k in required if result[k] is None]
    if missing:
        raise ValueError(f"缺少{missing}，实际列名：{list(df.columns)}；可用--columns传入JSON映射。")
    return result


def local_timestamp(value):
    if pd.isna(value):
        return pd.NaT
    t = pd.Timestamp(value)
    return t.tz_localize(TIMEZONE) if t.tzinfo is None else t.tz_convert(TIMEZONE)


def parse_datetime(row, cols):
    try:
        if cols["datetime"] is not None:
            value = row[cols["datetime"]]
            if isinstance(value, (float, int, np.number)):
                value = pd.Timestamp("1899-12-30") + pd.to_timedelta(float(value), unit="D")
            return local_timestamp(value)
        date = row[cols["date"]]
        if isinstance(date, (float, int, np.number)):
            day = pd.Timestamp("1899-12-30") + pd.to_timedelta(float(date), unit="D")
        else:
            day = pd.Timestamp(date)
        day = local_timestamp(day).normalize()
        clock = row[cols["time"]]
        if isinstance(clock, (datetime, pd.Timestamp)):
            clock = clock.time()
        if isinstance(clock, time):
            seconds = clock.hour * 3600 + clock.minute * 60 + clock.second + clock.microsecond / 1e6
        elif isinstance(clock, (float, int, np.number)):
            if not 0 <= float(clock) < 1:
                return pd.NaT
            seconds = float(clock) * 86400
        else:
            seconds = pd.to_timedelta(str(clock).strip()).total_seconds()
        if not math.isfinite(seconds) or not 0 <= seconds < 86400:
            return pd.NaT
        return day + pd.to_timedelta(seconds, unit="s")
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def canonical_id(row):
    values = [row["datetime"].isoformat()] + [format(float(row[c]), ".17g") for c in ("x", "y", "z", "energy_J")]
    return "ev_" + hashlib.sha256("|".join(values).encode()).hexdigest()[:24]


def clean_events(args, out):
    frames, input_info = [], []
    explicit = json.loads(args.columns) if args.columns else {}
    if set(explicit) - set(ALIASES):
        raise ValueError("--columns键只能为datetime/date/time/x/y/z/energy。")
    for name in args.input:
        path = Path(name).resolve()
        df = read_table(path, args.sheet, args.header)
        cols = resolve_columns(df, explicit)
        clean = pd.DataFrame(index=df.index)
        clean["datetime"] = pd.to_datetime(df.apply(lambda r: parse_datetime(r, cols), axis=1), utc=True).dt.tz_convert(TIMEZONE)
        for c in ("x", "y", "z", "energy"):
            clean[c if c != "energy" else "energy_J"] = pd.to_numeric(df[cols[c]], errors="coerce") if cols[c] is not None else np.nan
        clean["energy_J"] *= args.energy_multiplier
        clean["source_file"] = str(path)
        clean["source_row"] = np.arange(len(df)) + args.header + 2
        clean["reject_reason"] = ""
        clean.loc[clean.datetime.isna(), "reject_reason"] += "invalid_datetime;"
        for c in ("x", "y", "energy_J"):
            clean.loc[~np.isfinite(clean[c]), "reject_reason"] += f"invalid_{c};"
        if cols["z"] is not None:
            clean.loc[~np.isfinite(clean.z), "reject_reason"] += "invalid_z;"
        clean.loc[clean.energy_J <= 0, "reject_reason"] += "nonpositive_energy;"
        frames.append(clean)
        input_info.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "rows": len(df), "columns": cols})
    all_rows = pd.concat(frames, ignore_index=True)
    rejected = all_rows[all_rows.reject_reason != ""]
    rejected.to_csv(out / "rejected_records.csv", index=False, encoding="utf-8-sig")
    good = all_rows[all_rows.reject_reason == ""].drop(columns="reject_reason").copy()
    if good.empty:
        raise ValueError("没有有效事件，请检查rejected_records.csv和列映射。")
    good["event_id"] = good.apply(canonical_id, axis=1)
    # 同时刻同XYZ同能量才去重，不按近似时间/距离误删事件。
    dup = good.duplicated(["datetime", "x", "y", "z", "energy_J"], keep="first")
    good.assign(is_duplicate=dup).to_csv(out / "source_event_map.csv", index=False, encoding="utf-8-sig")
    good[dup].to_csv(out / "duplicate_records.csv", index=False, encoding="utf-8-sig")
    good = good[~dup].sort_values("datetime").reset_index(drop=True)
    good["energy_class"] = (good.energy_J >= args.threshold).astype(np.int8)
    good["date"] = good.datetime.dt.strftime("%Y-%m-%d")
    return good, {"input_files": input_info, "raw_rows": len(all_rows), "rejected_rows": len(rejected),
                  "duplicate_rows": int(dup.sum()), "clean_events": len(good),
                  "high_events": int(good.energy_class.sum())}


def day_boundary(value):
    t = local_timestamp(value)
    if pd.isna(t) or t != t.normalize():
        raise ValueError(f"日期边界必须为北京时间00:00：{value}")
    return t


def build_grid(events, args):
    train = events[events.datetime < day_boundary(args.train_end)]
    if args.bounds:
        xmin, xmax, ymin, ymax = args.bounds
        source = "explicit_bounds"
    else:
        if train.empty:
            raise ValueError("训练期没有事件，无法拟合网格；请设置--train-end或--bounds。")
        xmin, xmax = train.x.min() - args.padding, train.x.max() + args.padding
        ymin, ymax = train.y.min() - args.padding, train.y.max() + args.padding
        source = "events_before_train_end_plus_padding"
    if not np.isfinite([xmin, xmax, ymin, ymax]).all() or xmax <= xmin or ymax <= ymin:
        raise ValueError("网格边界无效。")
    xe = np.linspace(xmin, xmax, args.grid + 1)
    ye = np.linspace(ymin, ymax, args.grid + 1)
    return xe, ye, source


def render_day(events, xe, ye, sigma, truncate):
    """每个事件的核在当前矩形网格内归一化，sum(kernel)=1。
    数组索引为[channel,y,x]。energy_grid是未平滑J/格，energy_J是平滑J/格。
    网格外事件由调用者剔出并单独记录，禁止夹到边缘冒充网格内事件。
    """
    n = len(xe) - 1
    grid = np.zeros((2, n, n), dtype=np.float64)
    smooth = np.zeros_like(grid)
    counts = np.zeros((2, n, n), dtype=np.int32)
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    radius = sigma * truncate
    for row in events.itertuples(index=False):
        c, energy = int(row.energy_class), float(row.energy_J)
        ix = min(n - 1, int(np.searchsorted(xe, row.x, side="right") - 1))
        iy = min(n - 1, int(np.searchsorted(ye, row.y, side="right") - 1))
        if ix < 0 or iy < 0 or row.x > xe[-1] or row.y > ye[-1]:
            raise ValueError("render_day收到网格外事件。")
        grid[c, iy, ix] += energy
        counts[c, iy, ix] += 1
        xs = np.flatnonzero(np.abs(xc - row.x) <= radius)
        ys = np.flatnonzero(np.abs(yc - row.y) <= radius)
        if len(xs) and len(ys):
            d2 = ((xc[xs][None, :] - row.x) ** 2 + (yc[ys][:, None] - row.y) ** 2) / sigma**2
            kernel = np.exp(-0.5 * d2) * (d2 <= truncate**2)
            total = kernel.sum()
        else:
            total = 0.0
        if total <= 0:
            # 极粗网格/极小带宽时，退化为所在格点能量，不丢事件。
            smooth[c, iy, ix] += energy
        else:
            smooth[c][np.ix_(ys, xs)] += energy * kernel / total
    if not np.allclose(smooth.sum(axis=(1, 2)), grid.sum(axis=(1, 2)), rtol=1e-12, atol=1e-7):
        raise AssertionError("能量守恒检查失败。")
    return smooth, grid, counts


def load_coverage(path):
    """可选CSV：date,status；status为normal/missing/unknown。
    normal必须来自完整日监测记录，不能由当天有事件推断。
    """
    if path is None:
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    if not {"date", "status"}.issubset(df.columns):
        raise ValueError("coverage文件需要date,status列。")
    df["date"] = df.date.map(lambda v: day_boundary(v).strftime("%Y-%m-%d"))
    if df.date.duplicated().any() or not df.status.isin(["normal", "missing", "unknown"]).all():
        raise ValueError("coverage日期重复或status不在normal/missing/unknown中。")
    return dict(zip(df.date, df.status))


def make_previews(out, records):
    """从energy_J生成便于查看的PNG，不修改NPZ数据。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    # 固定色标。低能和高能分别设定，跨日期使用同一尺度。
    # 超过上限的像素只在图上显示为顶色，不修改实际数值。
    channel_settings = {
        0: ("low", 10_000.0),
        1: ("high", 250_000.0),
    }
    linthresh = 1.0  # 1 J/格以下线性显示，以上使用非线性色标

    for row in records:
        matrix_path = out / row["matrix_file"]

        with np.load(matrix_path) as data:
            energy = data["energy_J"].copy()
            x_edges = data["x_edges"].copy()
            y_edges = data["y_edges"].copy()
            count_grid = data["count_grid"].copy()
            energy_grid_J = data["energy_grid_J"].copy()

        x0, y0 = x_edges[0], y_edges[0]

        for channel, (name, vmax) in channel_settings.items():
            image = energy[channel].astype(np.float64)
            masked = np.ma.masked_equal(image, 0.0)

            fig, ax = plt.subplots(figsize=(7, 5))
            cmap = plt.get_cmap("YlOrRd").copy()
            cmap.set_bad("white")

            norm = SymLogNorm(
                linthresh=linthresh,
                linscale=1.0,
                base=10,
                vmin=0.0,
                vmax=vmax,
            )

            mesh = ax.pcolormesh(
                x_edges - x0,
                y_edges - y0,
                masked,
                cmap=cmap,
                norm=norm,
                shading="flat",
            )

            clipped = int((image > vmax).sum())
            count = int(count_grid[channel].sum())
            total_energy = float(energy_grid_J[channel].sum())

            colorbar = fig.colorbar(
                mesh,
                ax=ax,
                extend="max" if clipped else "neither",
            )
            colorbar.set_label(
                "Allocated energy (J/cell; nonlinear scale)"
            )

            ax.set_title(
                f"{row['date']} | {name} energy events\n"
                f"Count: {count} | Total: {total_energy:.3g} J"
            )
            ax.set_xlabel("Relative X (m)")
            ax.set_ylabel("Relative Y (m)")
            ax.set_aspect("equal")

            if count == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No events in this energy class",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="gray",
                )

            if clipped:
                ax.text(
                    0.99,
                    0.01,
                    f"{clipped} cells exceed display limit",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="black",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )

            ax.set_xlim(0, x_edges[-1] - x0)
            ax.set_ylim(0, y_edges[-1] - y0)

            save_dir = out / "preview" / name
            save_dir.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                save_dir / f"{row['date']}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", default=DEFAULT_INPUT, help="一个或多个同结构CSV/XLS/XLSX事件文件")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="输出目录；非空时拒绝覆盖")
    p.add_argument("--sheet", default="0", help="Excel工作表编号或名称，默认0")
    p.add_argument("--header", type=int, default=0, help="0起算表头行")
    p.add_argument("--columns", help='JSON列映射，例如 {"date":"Data","time":"CZAS","energy":"ENERGIA"}')
    p.add_argument("--energy-multiplier", type=float, default=1.0, help="源能量换算到J的倍数；kJ设1000")
    p.add_argument("--threshold", type=float, default=THRESHOLD_J)
    p.add_argument("--energy-ref", type=float, default=ENERGY_REFERENCE_J)
    p.add_argument("--grid", type=int, default=GRID_SIZE)
    p.add_argument("--sigma-m", type=float, default=SIGMA_M)
    p.add_argument("--truncate", type=float, default=3.0, help="高斯径向截断半径，单位sigma")
    p.add_argument("--train-end", default=TRAIN_END_EXCLUSIVE, help="训练拟合截止日期，不包含当天")
    p.add_argument("--bounds", nargs=4, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    p.add_argument("--padding", type=float, default=150.0, help="训练事件坐标范围外扩米数")
    p.add_argument("--start", help="首个日窗口日期，包含；默认首事件所在日")
    p.add_argument("--end", help="最后边界日期，不包含；默认末事件次日")
    p.add_argument("--coverage", help="完整日监测状态CSV：date,status")
    p.add_argument("--png-every", type=int, default=1, help="每N天预览两张PNG，0关闭")
    args = p.parse_args()
    args.sheet = int(args.sheet) if args.sheet.isdigit() else args.sheet
    if args.grid < 2 or args.header < 0 or args.png_every < 0:
        p.error("grid>=2，header>=0，png-every>=0")
    for k in ("energy_multiplier", "threshold", "energy_ref", "sigma_m", "truncate", "padding"):
        if not math.isfinite(getattr(args, k)) or getattr(args, k) <= 0:
            p.error(f"{k}必须为有限正数")
    return args


def main():
    args = parse_args()
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"输出目录非空，避免混入旧结果，请换--output：{out}")
    out.mkdir(parents=True, exist_ok=True)
    events, summary = clean_events(args, out)
    start = day_boundary(args.start) if args.start else events.datetime.min().normalize()
    end = day_boundary(args.end) if args.end else events.datetime.max().normalize() + pd.Timedelta(days=1)
    if end <= start:
        raise ValueError("end必须晚于start，且end是排除边界。")
    selected = events[(events.datetime >= start) & (events.datetime < end)].copy()
    if selected.empty:
        raise ValueError("指定日期范围没有事件。")
    xe, ye, bounds_source = build_grid(selected, args)
    inside = selected.x.between(xe[0], xe[-1]) & selected.y.between(ye[0], ye[-1])
    selected["inside_grid"] = inside
    events.to_csv(out / "events_clean.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(out / "events_selected.csv", index=False, encoding="utf-8-sig")
    selected[~inside].to_csv(out / "out_of_bounds.csv", index=False, encoding="utf-8-sig")
    excluded = events[(events.datetime < start) | (events.datetime >= end)]
    excluded.to_csv(out / "outside_time_range.csv", index=False, encoding="utf-8-sig")
    (out / "daily").mkdir()
    coverage = load_coverage(args.coverage)
    groups = {k: v for k, v in selected.groupby("date")}
    empty = selected.iloc[:0]
    records, preview_records, event_index = [], [], {}
    train_vmax = 0.0
    max_energy_error = 0.0
    days = pd.date_range(start, end, freq="D", inclusive="left")
    for j, day in enumerate(days):
        date = day.strftime("%Y-%m-%d")
        day_events = groups.get(date, empty)
        frame_events = day_events[day_events.inside_grid]
        smooth, grid, counts = render_day(frame_events, xe, ye, args.sigma_m, args.truncate)
        log_map = np.log10(1 + smooth / args.energy_ref).astype(np.float32)
        stored = smooth.astype(np.float32)
        if not np.isfinite(stored).all() or not np.isfinite(log_map).all():
            raise OverflowError(f"{date}: float32溢出，请检查源能量单位。")
        error = np.max(np.abs(stored.astype(np.float64).sum(axis=(1, 2)) - grid.sum(axis=(1, 2))))
        max_energy_error = max(max_energy_error, float(error))
        np.testing.assert_allclose(stored.astype(np.float64).sum(axis=(1, 2)), grid.sum(axis=(1, 2)), rtol=1e-6, atol=1e-3)
        fname = f"daily/{date}.npz"
        next_day = day + pd.Timedelta(days=1)
        np.savez_compressed(out / fname, energy_J=stored, energy_log10=log_map,
                            energy_grid_J=grid, count_grid=counts,
                            high_occupancy=(counts[1:2] > 0).astype(np.uint8),
                            x_edges=xe, y_edges=ye,
                            window_start=np.array(day.isoformat()), window_end=np.array(next_day.isoformat()))
        status = coverage.get(date, "unknown")
        oob = int((~day_events.inside_grid).sum())
        if status == "missing" and len(day_events):
            raise ValueError(f"{date}: coverage标记missing但存在事件，请核查监测记录。")
        row = {"date": date, "window_start": day.isoformat(), "window_end": next_day.isoformat(),
               "matrix_file": fname, "n_events": len(day_events), "n_low": int((day_events.energy_class == 0).sum()),
               "n_high": int((day_events.energy_class == 1).sum()), "n_outside_grid": oob,
               "total_energy_J": float(day_events.energy_J.sum()),
               "high_energy_J": float(day_events.loc[day_events.energy_class == 1, "energy_J"].sum()),
               "raster_low_J": float(grid[0].sum()), "raster_high_J": float(grid[1].sum()),
               "monitoring_status": status,
               "usable_with_confirmed_coverage": status == "normal" and oob == 0,
               "catalog_edge_day": date in {events.date.iloc[0], events.date.iloc[-1]}}
        records.append(row)
        event_index[date] = {"all_event_ids": day_events.event_id.tolist(),
                             "raster_event_ids": frame_events.event_id.tolist(),
                             "high_event_ids": day_events.loc[day_events.energy_class == 1, "event_id"].tolist()}
        if day < day_boundary(args.train_end):
            train_vmax = max(train_vmax, float(log_map.max()))
        if args.png_every and j % args.png_every == 0:
            preview_records.append(row)
        if j % 100 == 0 or j == len(days) - 1:
            print(f"[{j+1}/{len(days)}] {date}: low={row['n_low']} high={row['n_high']}", flush=True)
    daily = pd.DataFrame(records)
    daily.to_csv(out / "daily_manifest.csv", index=False, encoding="utf-8-sig")
    daily.assign(month=daily.date.str[:7]).groupby("month")[["n_events", "n_low", "n_high", "total_energy_J"]].sum().to_csv(out / "monthly_summary.csv", encoding="utf-8-sig")
    # 连续无事件日仅报告，不自动判定为停机或可靠负样本。
    blank = daily.n_events.eq(0)
    gaps = []
    for _, block in daily[blank].groupby((~blank).cumsum()):
        gaps.append({"start": block.date.iloc[0], "end_inclusive": block.date.iloc[-1], "days": len(block)})
    dump_json(out / "empty_day_runs.json", gaps)
    dump_json(out / "daily_event_ids.json", event_index)
    cfg = vars(args).copy()
    cfg.update({"timezone": TIMEZONE, "channel_order": ["low_E_lt_threshold", "high_E_ge_threshold"],
                "axis_order": ["channel", "y", "x"], "row_zero": "smallest_y", "bounds_source": bounds_source,
                "bounds_xy": [float(xe[0]), float(xe[-1]), float(ye[0]), float(ye[-1])],
                "cell_size_m": [float(xe[1] - xe[0]), float(ye[1] - ye[0])],
                "start_inclusive": start.isoformat(), "end_exclusive": end.isoformat(),
                "kernel": "event-centered isotropic Gaussian; radial truncation; per-event in-grid sum normalized to 1",
                "energy_map_unit": "J per cell (spatially allocated)",
                "transform": "log10(1 + energy_J / energy_ref)",
                "inverse": "energy_ref * (10**energy_log10 - 1)",
                "png_fixed_vmax": max(train_vmax, 1.0e-6), "png_scale_source": "training frames only; future overflow clipped for display"})
    dump_json(out / "dataset_config.json", cfg)
    summary.update({"selected_events": len(selected), "excluded_time_events": len(excluded), "days": len(days),
                    "empty_days": int(blank.sum()), "high_event_days": int((daily.n_high > 0).sum()),
                    "out_of_bounds_events": int((~inside).sum()),
                    "out_of_bounds_high_events": int(((~inside) & (selected.energy_class == 1)).sum()),
                    "confirmed_usable_days": int(daily.usable_with_confirmed_coverage.sum()),
                    "max_float32_total_energy_error_J": max_energy_error})
    if preview_records:
        make_previews(out, preview_records)
    summary["completed"] = True
    dump_json(out / "summary.json", summary)  # 只在全部完成后写完成标记
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果已保存：{out}")
    if coverage == {}:
        print("未提供完整日监测记录：所有日帧监测状态为unknown，零图不能自动当作可靠负样本。")
    if (~inside).any():
        print("存在网格外事件，见out_of_bounds.csv；受影响日不能作为完整空间标签。")


if __name__ == "__main__":
    main()
