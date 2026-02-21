# src/tools/score_dist_tune.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
score_dist_tune.py (moved under src/tools)

「現状の分布」→「目標分布に寄せる新閾値」→「raw→score再計算」→「分布比較」を一気に出す。

要件:
- temp/evaluation_results_*.csv をそのまま食わせて処理（デフォルトglob）
- 直近(更新時刻)のNファイルのみ対象（デフォルト5）
- *_raw と *_score を使用（対象は固定）
- score妥当性チェック導入（A確定）:
    - 値域 0..1（微小許容）
    - 離散率 >= 95%（{0,0.25,0.5,0.75,1.0}）
    - *_raw 併存必須
    - 満たさない metric はスキップ＋警告

追加（退行検知・健全性チェック強化） (#701-5):
- metric_summary.csv に以下を追加/強化
    - current/new の sat(0+1) と target_l1 を出力し退行を検知
    - accepted率の current/new（ranking CSV から推定。可能なら acceptance.py で再計算）
    - direction 整合性ラベル（unknown/conflict）
    - score_discrete_ratio / in_range_ratio の current/new
    - validation_pass / validation_fail_reasons を出力

対象メトリクス（固定）:
- 技術系: sharpness, blurriness, contrast, noise, local_contrast
- face系: face_sharpness, face_blurriness, face_contrast
※ local_contrast 以外の "local_*" や body_composition は対象外（誤認混入を避ける）

目標分布（技術系）:
- 1.0: 5-10%, 0.75: 20-30%, 0.5: 25-35%, 0.25: 15-25%, 0.0: 10-15%
→ 自動閾値提案は「目標中心」に合わせた分位点境界を使う:
  境界: t0=12.5%, t1=32.5%, t2=62.5%, t3=87.5%

face系:
- 現段階は「極端寄り抑制」を“監視”に留める（警告と可視化）。
- 自動閾値提案の適用は --apply-face-auto で切替可能（デフォルトOFF推奨）。

使い方:
  python -m src.tools.score_dist_tune
  python -m src.tools.score_dist_tune --n-files 5
  python -m src.tools.score_dist_tune --params-json temp/new_params.json
  python -m src.tools.score_dist_tune --ranking-glob "output/**/evaluation_ranking_*.csv"
  python -m src.tools.score_dist_tune --validate-require-target-l1-improve --validate-direction-conflict-fail
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import matplotlib

matplotlib.use("Agg")  # ★ headless環境対応
import matplotlib.pyplot as plt  # noqa: E402

# -------------------------------------------------
# ensure import paths (src is importable)
# -------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../photo_insight
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# -------------------------------------------------
# #701-5 SSOT validation
# -------------------------------------------------
from .score_dist_validation import Criteria7015, validate_row_701_5  # noqa: E402

try:
    import yaml
except Exception:
    yaml = None

DISCRETE_SCORES = [0.0, 0.25, 0.5, 0.75, 1.0]
DISCRETE_SET = set(DISCRETE_SCORES)

# raw_transform の列挙（将来拡張しても互換が壊れにくいように固定）
RAW_TRANSFORM_ENUM = {
    "identity",  # 値はそのまま（高いほど良い）
    "lower_is_better",  # 値は変えず、score割当ロジックを逆向きにする
    "negate",  # effective = -raw（将来導入する場合）
    "affine",  # effective = a*raw + b（将来導入する場合）
    "clip",  # effective = clip(raw, lo, hi)（将来導入する場合）
}

# 対象固定（score_col -> raw_col を自動派生）
TARGET_SCORE_COLS = [
    "sharpness_score",
    "blurriness_score",
    "contrast_score",
    "noise_score",
    "local_contrast_score",
    "face_sharpness_score",
    "face_blurriness_score",
    "face_contrast_score",
]

TECH_SCORE_COLS = {
    "sharpness_score",
    "blurriness_score",
    "contrast_score",
    "noise_score",
    "local_contrast_score",
}
FACE_SCORE_COLS = {
    "face_sharpness_score",
    "face_blurriness_score",
    "face_contrast_score",
}

# 技術系の目標分布レンジ（割合）
TECH_TARGET_RANGES = {
    1.0: (0.05, 0.10),
    0.75: (0.20, 0.30),
    0.5: (0.25, 0.35),
    0.25: (0.15, 0.25),
    0.0: (0.10, 0.15),
}
TECH_TARGET_CENTER = {
    lv: (lo + hi) / 2.0 for lv, (lo, hi) in TECH_TARGET_RANGES.items()
}

# 目標中心に合わせた分位点境界（rawが高いほど良い想定）
# t0: 12.5%, t1: 32.5%, t2: 62.5%, t3: 87.5%
TECH_TARGET_QUANTILES = [0.125, 0.325, 0.625, 0.875]


@dataclass
class Thresholds:
    """
    5段階離散スコア用の閾値（rawが高いほど良い想定）
      raw < t0       -> 0.0
      t0 <= raw < t1 -> 0.25
      t1 <= raw < t2 -> 0.5
      t2 <= raw < t3 -> 0.75
      t3 <= raw      -> 1.0

    ※ noise_sigma_used のように「低いほど良い」場合は
       score_from_raw(..., higher_is_better=False) で反転評価する。
    """

    t0: float
    t1: float
    t2: float
    t3: float

    def to_list(self) -> List[float]:
        return [self.t0, self.t1, self.t2, self.t3]

    @staticmethod
    def from_list(xs: List[float]) -> "Thresholds":
        if len(xs) != 4:
            raise ValueError(f"thresholds must have 4 values, got {len(xs)}")
        return Thresholds(float(xs[0]), float(xs[1]), float(xs[2]), float(xs[3]))


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def find_latest_files(glob_pattern: str, n_files: int) -> List[Path]:
    paths = [Path(p) for p in sorted(Path().glob(glob_pattern))]
    if not paths:
        base = Path(glob_pattern).parent
        pat = Path(glob_pattern).name
        if base.exists():
            paths = list(base.glob(pat))

    if not paths:
        raise FileNotFoundError(f"No files found for pattern: {glob_pattern}")

    paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:n_files]
    return paths


def read_concat_csv(files: List[Path]) -> pd.DataFrame:
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__source_file__"] = f.name
            dfs.append(df)
        except Exception as e:
            warn(f"failed to read: {f} ({e})")
    if not dfs:
        raise RuntimeError("All CSV reads failed.")
    return pd.concat(dfs, ignore_index=True)


def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def round_to_quarter(x: float) -> float:
    return round(x * 4.0) / 4.0


def score_from_raw(
    raw: float, thr: Thresholds, *, higher_is_better: bool = True
) -> float:
    """
    raw + thresholds から 5段階離散スコアを算出する。

    - higher_is_better=True : raw が高いほど良い（通常）
    - higher_is_better=False: raw が低いほど良い（例: noise_sigma_used）
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return float("nan")

    if not higher_is_better:
        # 低いほど良い: 閾値に対する割当を逆向きにする（raw値の符号反転はしない）
        if raw < thr.t0:
            return 1.0
        if raw < thr.t1:
            return 0.75
        if raw < thr.t2:
            return 0.5
        if raw < thr.t3:
            return 0.25
        return 0.0

    # 通常（高いほど良い）
    if raw < thr.t0:
        return 0.0
    if raw < thr.t1:
        return 0.25
    if raw < thr.t2:
        return 0.5
    if raw < thr.t3:
        return 0.75
    return 1.0


def compute_score_ratios(scores: pd.Series) -> Dict[float, float]:
    s = coerce_numeric(scores).dropna()
    if s.empty:
        return {v: 0.0 for v in DISCRETE_SCORES}

    # 0.25刻みに丸めて離散判定
    s = s.map(lambda x: round_to_quarter(float(x)))
    vc = s.value_counts()
    total = int(vc.sum())
    ratios = {v: float(vc.get(v, 0)) / total for v in DISCRETE_SCORES}
    return ratios


def compute_score_counts(scores: pd.Series) -> Dict[float, int]:
    s = coerce_numeric(scores).dropna()
    if s.empty:
        return {v: 0 for v in DISCRETE_SCORES}

    s = s.map(lambda x: round_to_quarter(float(x)))
    vc = s.value_counts()
    return {v: int(vc.get(v, 0)) for v in DISCRETE_SCORES}


def saturation_ratio(ratios: Dict[float, float]) -> float:
    return ratios.get(0.0, 0.0) + ratios.get(1.0, 0.0)


def discrete_ratio(scores: pd.Series) -> float:
    """
    離散率は「母数からrange外を除外しない」。
    range外は in_range_ratio が落とす責務。
    """
    s = coerce_numeric(scores).dropna()
    if s.empty:
        return 0.0
    r = s.map(lambda x: round_to_quarter(float(x)))
    ok = r.isin(list(DISCRETE_SET))
    return float(ok.mean())


def in_range_ratio(scores: pd.Series, *, eps: float = 1e-6) -> float:
    s = coerce_numeric(scores).dropna()
    if s.empty:
        return 0.0
    ok = (s >= -eps) & (s <= 1.0 + eps)
    return float(ok.mean())


def infer_direction_by_score(
    raw_num: pd.Series, score: pd.Series
) -> Tuple[bool, Optional[float], int, bool]:
    """
    raw と score の相関から「高いほど良いか」を推定する（根拠付き）。

    戻り値:
      (higher_is_better, corr, n_for_corr, direction_inferred)
    """
    r = coerce_numeric(raw_num)
    s = coerce_numeric(score)
    tmp = pd.DataFrame({"r": r, "s": s}).dropna()
    n = int(len(tmp))

    if n < 30:
        return True, None, n, False  # デフォルトは「高いほど良い」（推定はしていない）

    corr = tmp["r"].corr(tmp["s"])
    if corr is None or (isinstance(corr, float) and math.isnan(corr)):
        return True, None, n, True  # 計算は試みたが corr が無効

    return bool(corr >= 0), float(corr), n, True


def resolve_raw_col(
    df: pd.DataFrame, score_col: str
) -> Tuple[Optional[str], str, str, str, Dict[str, Any]]:
    metric = score_col.replace("_score", "")

    direction_meta: Dict[str, Any] = {
        "direction_inferred": False,
        "corr": None,
        "n_for_corr": 0,
        "direction_note": "",
    }

    def _direction_to_transform(direction: str) -> str:
        if direction == "higher_is_better":
            return "identity"
        if direction == "lower_is_better":
            return "lower_is_better"
        raise ValueError(f"invalid raw_direction: {direction}")

    # =========================
    # ★ noise は sigma 優先
    # =========================
    if metric == "noise":
        if "noise_sigma_used" in df.columns:
            raw_source = "noise_sigma_used"
            raw_direction = "lower_is_better"
            raw_transform = _direction_to_transform(raw_direction)

            direction_meta.update(
                {
                    "direction_inferred": False,
                    "corr": None,
                    "n_for_corr": 0,
                    "direction_note": "sigma_axis",
                }
            )

            return (
                "noise_sigma_used",
                raw_source,
                raw_direction,
                raw_transform,
                direction_meta,
            )

    # =========================
    # 通常系: xxx_raw
    # =========================
    c1 = f"{metric}_raw"
    if c1 in df.columns:
        raw_source = "raw"
        raw_direction = "higher_is_better"
        raw_transform = _direction_to_transform(raw_direction)
        return c1, raw_source, raw_direction, raw_transform, direction_meta

    return None, "fallback", "higher_is_better", "identity", direction_meta


def build_raw_transform_spec(
    metric: str,
    source_raw_col: str,
    *,
    higher_is_better: bool,
    raw_source: str,
    direction_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    raw の '変換仕様' を JSON に明示するためのメタ情報（raw_spec）。
    """
    raw_direction = "higher_is_better" if higher_is_better else "lower_is_better"
    raw_transform = "identity" if higher_is_better else "lower_is_better"

    if raw_transform not in RAW_TRANSFORM_ENUM:
        raise ValueError(f"invalid raw_transform: {raw_transform}")

    effective_raw_col = source_raw_col

    dm = direction_meta or {}
    return {
        "raw_direction": raw_direction,
        "raw_transform": raw_transform,
        "source_raw_col": source_raw_col,
        "effective_raw_col": effective_raw_col,
        "raw_source": raw_source,
        "higher_is_better": higher_is_better,
        "direction_inferred": dm.get("direction_inferred", False),
        "corr": dm.get("corr", None),
        "n_for_corr": dm.get("n_for_corr", 0),
        "direction_note": dm.get("direction_note", ""),
    }


def validate_score_column(
    df: pd.DataFrame,
    score_col: str,
    raw_col: str,
    *,
    min_samples: int,
    min_discrete_ratio: float,
    eps: float = 1e-6,
) -> Tuple[bool, str]:
    # raw併存必須
    if raw_col not in df.columns:
        return False, f"missing required raw column: {raw_col}"
    if score_col not in df.columns:
        return False, f"missing score column: {score_col}"

    s = coerce_numeric(df[score_col]).dropna()
    n = int(s.shape[0])
    if n < min_samples:
        return False, f"too few samples (n={n} < {min_samples})"

    r_in = in_range_ratio(df[score_col], eps=eps)
    if r_in < 0.98:
        return False, f"in-range ratio too low ({r_in:.3f} < 0.98); suspicious column"

    r_disc = discrete_ratio(df[score_col])
    if r_disc < min_discrete_ratio:
        return (
            False,
            f"discrete ratio too low ({r_disc:.3f} < {min_discrete_ratio}); suspicious column",
        )

    return True, "ok"


def propose_thresholds_for_target_distribution(raw_values: pd.Series) -> Thresholds:
    r = coerce_numeric(raw_values).dropna()
    if len(r) < 20:
        # データが少ない場合は等間隔でフォールバック（落ちないための保険）
        mn, mx = (float(r.min()), float(r.max())) if len(r) else (0.0, 1.0)
        step = (mx - mn) / 5.0 if mx > mn else 1.0
        return Thresholds(mn + step, mn + 2 * step, mn + 3 * step, mn + 4 * step)

    qs = r.quantile(TECH_TARGET_QUANTILES).tolist()
    return Thresholds.from_list([float(x) for x in qs])


def tech_target_flags(ratios: Dict[float, float]) -> Dict[float, str]:
    flags: Dict[float, str] = {}
    for lv, (lo, hi) in TECH_TARGET_RANGES.items():
        v = ratios.get(lv, 0.0)
        flags[lv] = "in" if (lo <= v <= hi) else ("low" if v < lo else "high")
    return flags


def tech_target_l1(ratios: Dict[float, float]) -> float:
    return sum(
        abs(ratios.get(lv, 0.0) - TECH_TARGET_CENTER[lv]) for lv in DISCRETE_SCORES
    )


def make_plots(
    out_dir: Path,
    metric: str,
    raw: pd.Series,
    cur_score: pd.Series,
    new_score: pd.Series,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_n = coerce_numeric(raw).dropna()
    cur_n = coerce_numeric(cur_score).dropna().map(lambda x: round_to_quarter(float(x)))
    new_n = coerce_numeric(new_score).dropna().map(lambda x: round_to_quarter(float(x)))

    # raw histogram
    plt.figure(figsize=(8, 4.2))
    if not raw_n.empty:
        plt.hist(raw_n, bins=50)
    plt.title(f"{metric}: raw distribution")
    plt.xlabel("raw")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_raw_hist.png", dpi=150)
    plt.close()

    def counts_for(series: pd.Series) -> List[int]:
        vc = series.value_counts()
        return [int(vc.get(v, 0)) for v in DISCRETE_SCORES]

    cur_counts = counts_for(cur_n)
    new_counts = counts_for(new_n)

    x = list(range(len(DISCRETE_SCORES)))
    width = 0.4
    plt.figure(figsize=(8, 4.2))
    plt.bar([i - width / 2 for i in x], cur_counts, width=width, label="current")
    plt.bar([i + width / 2 for i in x], new_counts, width=width, label="new")
    plt.title(f"{metric}: score distribution (current vs new)")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.xticks(x, [str(v) for v in DISCRETE_SCORES])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_score_compare.png", dpi=150)
    plt.close()


def thresholds_list_to_mapping(
    metric: str, ts: List[float]
) -> Optional[Dict[str, float]]:
    """
    evaluator_thresholds.yaml 形式の mapping に変換（tech/face共通）
    """
    if len(ts) != 4:
        return None

    t0, t1, t2, t3 = [float(x) for x in ts]

    if metric in (
        "sharpness",
        "contrast",
        "local_contrast",
        "face_sharpness",
        "face_contrast",
    ):
        return {"poor": t0, "fair": t1, "good": t2, "excellent": t3}

    if metric in ("blurriness", "face_blurriness"):
        return {"bad": t0, "poor": t1, "fair": t2, "good": t3}

    # noise は mapping 化しない（suggestions に “材料” として保存する）
    if metric == "noise":
        return None

    return None


def build_evaluator_config_from_chosen_params(
    chosen_params_out: Dict[str, Dict]
) -> Dict[str, Dict]:
    """
    chosen_params_out（new_params_used.json 相当）から evaluator config（YAML用dict）を組み立てる。
    """
    out: Dict[str, Dict] = {}
    noise_sugg: Dict[str, Any] = {}

    for metric, spec in chosen_params_out.items():
        ts = spec.get("thresholds")
        if not ts:
            continue

        # face_* は混乱回避でスキップ
        if metric.startswith("face_"):
            continue

        # ★ noise は mapping 判定より前に処理する
        if metric == "noise":
            rs = spec.get("raw_spec", {})

            noise_sugg.setdefault("noise", {})
            noise_sugg["noise"] = {
                "axis": "sigma",
                "thresholds_5bin": ts,
                "good_sigma_suggestion": ts[1] if len(ts) == 4 else None,
                "warn_sigma_suggestion": ts[3] if len(ts) == 4 else None,
                "threshold_source": spec.get("source"),
                "score_col": spec.get("score_col"),
                "raw_col": spec.get("raw_col"),
                "raw_source": spec.get("raw_source"),
                "raw_spec": rs,
                "note": "suggestion only; apply tool converts to config.noise.good_sigma/warn_sigma",
            }
            continue

        mapping = thresholds_list_to_mapping(metric, ts)
        if mapping is None:
            continue

        out.setdefault(metric, {})
        out[metric]["discretize_thresholds_raw"] = mapping

    if noise_sugg:
        out["noise_suggestions"] = noise_sugg

    return out


def build_thresholds_dict_for_acceptance(
    chosen_params_out: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    acceptance.decide_accept(..., thresholds=...) に渡すための dict を（できる範囲で）構築する。
    """
    out: Dict[str, Any] = {}

    for metric, spec in chosen_params_out.items():
        ts = spec.get("thresholds")
        if not ts:
            continue

        mapping = thresholds_list_to_mapping(metric, ts)
        if mapping is None:
            continue

        out.setdefault(metric, {})
        out[metric]["discretize_thresholds_raw"] = mapping

    return out


def _unique_non_null_values(series: pd.Series) -> List[Any]:
    s = series.dropna()
    if s.empty:
        return []
    return list(pd.unique(s))


def validate_dataframe_contract(df: pd.DataFrame) -> bool:
    """
    CSV列契約チェック（追加②）
    """
    ok = True

    # 0) score_dist_tune の前提: 対象スコア列が存在するか
    missing_score_cols = [c for c in TARGET_SCORE_COLS if c not in df.columns]
    if missing_score_cols:
        warn(f"Contract violation: missing score columns: {missing_score_cols}")
        return False  # 致命（ツールが成立しない）

    # 1) raw 列が resolve できるか（最低限）
    for score_col in TARGET_SCORE_COLS:
        metric = score_col.replace("_score", "")
        raw_col, raw_source, raw_direction, raw_transform, meta = resolve_raw_col(
            df, score_col
        )
        if raw_col is None or raw_col not in df.columns:
            warn(
                f"Contract violation: raw resolve failed for metric='{metric}' "
                f"(score_col='{score_col}', resolved_raw='{raw_col}')"
            )
            ok = False

    # 2) blurriness の raw direction 契約（厳密）
    expected = {
        "blurriness_raw_direction": "higher_is_better",
        "blurriness_raw_transform": "identity",
        "blurriness_higher_is_better": True,
    }

    missing = [k for k in expected.keys() if k not in df.columns]
    if missing:
        warn(
            "Contract warning: missing blurriness contract columns (allowed for now): "
            + str(missing)
        )
        return ok  # 欠損はまだ止めない（段階導入）

    for col, exp in expected.items():
        uniq = _unique_non_null_values(df[col])
        if not uniq:
            warn(f"Contract violation: '{col}' has no non-null values")
            ok = False
            continue

        if len(uniq) != 1:
            warn(f"Contract violation: '{col}' has multiple values: {uniq}")
            ok = False
            continue

        got = uniq[0]
        if isinstance(exp, bool):
            try:
                got_bool = bool(got)
            except Exception:
                got_bool = got
            if got_bool != exp:
                warn(f"Contract violation: '{col}' expected {exp} but got {got}")
                ok = False
        else:
            if str(got) != str(exp):
                warn(f"Contract violation: '{col}' expected '{exp}' but got '{got}'")
                ok = False

    return ok


def dump_yaml_like(data: Dict[str, Any]) -> str:
    """
    PyYAMLが無い環境でも落ちない簡易YAML（この用途に十分）。
    """

    def _emit(obj, indent=0):
        sp = "  " * indent
        if isinstance(obj, dict):
            lines = []
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{sp}{k}:")
                    lines.extend(_emit(v, indent + 1))
                else:
                    lines.append(f"{sp}{k}: {v}")
            return lines
        if isinstance(obj, list):
            lines = []
            for v in obj:
                if isinstance(v, (dict, list)):
                    lines.append(f"{sp}-")
                    lines.extend(_emit(v, indent + 1))
                else:
                    lines.append(f"{sp}- {v}")
            return lines
        return [f"{sp}{obj}"]

    return "\n".join(_emit(data)) + "\n"


# -----------------------------
# accepted/direction/targets helpers (#701-5)
# -----------------------------
def accepted_ratio_from_ranking(
    df: Optional[pd.DataFrame],
    *,
    accepted_col: str,
    secondary_col: str,
    mode: str,
) -> float:
    if df is None or df.empty:
        return float("nan")
    if accepted_col not in df.columns:
        return float("nan")

    pri = pd.to_numeric(df[accepted_col], errors="coerce").fillna(0) > 0

    if mode == "primary":
        return float(pri.mean())

    if secondary_col not in df.columns:
        return float(pri.mean())

    sec = pd.to_numeric(df[secondary_col], errors="coerce").fillna(0) > 0
    return float((pri | sec).mean())


def direction_final_label(higher_is_better: bool) -> str:
    return "higher_is_better" if higher_is_better else "lower_is_better"


def decide_direction_final(
    *,
    higher_is_better: bool,
    direction_inferred: bool,
    corr,
    n_for_corr: int,
    direction_note: str,
) -> str:
    # noise sigma axis is always lower_is_better
    if direction_note == "sigma_axis":
        return "lower_is_better"

    # Not enough evidence
    if (not direction_inferred) or (corr in ("", None)) or (int(n_for_corr) < 30):
        return "unknown"

    try:
        c = float(corr)
    except Exception:
        return "unknown"

    inferred = "higher_is_better" if c > 0 else "lower_is_better"
    cfg = direction_final_label(higher_is_better)
    return inferred if inferred == cfg else "conflict"


def target_match_ratio(flags: Optional[Dict[float, str]]) -> float:
    if not flags:
        return float("nan")
    total = len(DISCRETE_SCORES)
    if total == 0:
        return float("nan")
    ok = sum(1 for sv in DISCRETE_SCORES if flags.get(sv) == "in")
    return float(ok / total)


def _nan_to_empty(x: float) -> Any:
    return "" if x != x else float(x)


def compute_new_accepted_ratio_via_acceptance_from_results(
    df_results_with_new_scores: pd.DataFrame,
    *,
    thresholds_for_acceptance: Optional[Dict[str, Any]] = None,
) -> float:
    """
    df_results_with_new_scores に new_score を反映済みとして、
    ポートレート受け入れ判定を再実行し new_accepted_ratio を推定する。

    NOTE:
    - evaluator_thresholds（discretize_thresholds_raw）とは別系統の閾値が accept rules に存在するため、
      thresholds_for_acceptance は「accept rules 用のキー体系」であることが前提。
      未指定(None)なら accept rules のデフォルトで判定する。
    """
    if df_results_with_new_scores is None or df_results_with_new_scores.empty:
        return float("nan")

    decide_accept = None
    err_msgs: List[str] = []

    # 1) canonical (current "truth"): evaluators accept rules
    try:
        from photo_insight.evaluators.portrait_accept_rules import decide_accept as _decide_accept  # type: ignore

        decide_accept = _decide_accept
    except Exception as e:
        err_msgs.append(f"portrait_accept_rules import failed: {e}")

    # 2) legacy fallbacks (keep for safety)
    if decide_accept is None:
        for mod_path in [
            "photo_insight.batch_processor.evaluation_rank.acceptance",
            "batch_processor.evaluation_rank.acceptance",
            "evaluation_rank.acceptance",
        ]:
            try:
                acc_mod = __import__(mod_path, fromlist=["*"])
                if hasattr(acc_mod, "decide_accept") and callable(
                    getattr(acc_mod, "decide_accept")
                ):
                    decide_accept = getattr(acc_mod, "decide_accept")
                    break
            except Exception as e:
                err_msgs.append(f"{mod_path} import failed: {e}")

    if decide_accept is None:
        warn("new accepted: cannot import decide_accept. " + " | ".join(err_msgs))
        return float("nan")

    ok = 0
    total = 0
    first_err: Optional[str] = None

    for _, r in df_results_with_new_scores.iterrows():
        d = r.to_dict()
        try:
            # thresholds 引数を受ける/受けない両対応
            if thresholds_for_acceptance is not None:
                try:
                    out = decide_accept(d, thresholds=thresholds_for_acceptance)
                except TypeError:
                    out = decide_accept(d)
            else:
                out = decide_accept(d)

            acc = (
                bool(out[0])
                if isinstance(out, (tuple, list)) and len(out) >= 1
                else bool(out)
            )
            total += 1
            if acc:
                ok += 1
        except Exception as e:
            if first_err is None:
                first_err = repr(e)
            continue

    if total == 0:
        warn(
            f"new accepted: decide_accept failed for all rows: {first_err or 'unknown'}"
        )
        return float("nan")

    return float(ok / total)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", default="temp/evaluation_results_*.csv")
    ap.add_argument("--n-files", type=int, default=5)
    ap.add_argument("--out-dir", default="temp/score_dist_tune_out")

    ap.add_argument(
        "--params-json",
        default="",
        help="新パラメータをJSONで指定する場合のパス（未指定なら自動提案）",
    )

    # A確定仕様
    ap.add_argument("--min-samples", type=int, default=50)
    ap.add_argument("--min-discrete-ratio", type=float, default=0.95)

    # face系はデフォルト自動提案をしない（極端寄り抑制は“監視”）
    ap.add_argument(
        "--apply-face-auto",
        action="store_true",
        help="face系にも自動閾値提案を適用する（デフォルトはOFF推奨）",
    )
    ap.add_argument(
        "--face-saturation-warn",
        type=float,
        default=0.80,
        help="face系で(0+1)飽和率がこの値を超えたら警告",
    )

    ap.add_argument(
        "--emit-config-yaml",
        default="",
        help="evaluator_thresholds.yaml 形式で出力するパス（例: temp/evaluator_thresholds.yaml）",
    )
    ap.add_argument(
        "--emit-config-json", default="", help="同内容を JSON でも出力するパス（任意）"
    )

    # accepted ratio (from evaluation_ranking_*.csv)
    ap.add_argument("--ranking-glob", default="output/**/evaluation_ranking_*.csv")
    ap.add_argument("--ranking-n-files", type=int, default=5)
    ap.add_argument("--accepted-col", default="accepted_flag")
    ap.add_argument("--secondary-accepted-col", default="secondary_accept_flag")
    ap.add_argument(
        "--accepted-mode",
        choices=["primary", "primary_or_secondary"],
        default="primary",
        help="How to compute accepted ratio from ranking CSV",
    )

    # #701-5 validation thresholds (SSOT)
    ap.add_argument(
        "--validate-sat-max",
        type=float,
        default=0.40,
        help="max allowed saturation(0+1) for NEW",
    )
    ap.add_argument(
        "--validate-accepted-delta-max",
        type=float,
        default=0.10,
        help="max allowed abs delta accepted_ratio",
    )
    ap.add_argument(
        "--validate-discrete-min",
        type=float,
        default=0.95,
        help="min allowed score_discrete_ratio for NEW",
    )
    ap.add_argument(
        "--validate-in-range-min",
        type=float,
        default=0.98,
        help="min allowed in_range_ratio for NEW",
    )
    ap.add_argument(
        "--validate-require-target-l1-improve",
        action="store_true",
        help="if set, require new_tech_target_l1 <= current_tech_target_l1 for tech metrics",
    )
    ap.add_argument(
        "--validate-direction-conflict-fail",
        action="store_true",
        help="if set, direction_final=conflict causes FAIL (recommended for CI)",
    )

    args = ap.parse_args()

    latest_files = find_latest_files(args.input_glob, args.n_files)
    info("using files (latest first):")
    for p in latest_files:
        info(f"  - {p}")

    df = read_concat_csv(latest_files)

    # ---- CSV contract validation ----
    if not validate_dataframe_contract(df):
        warn("CSV contract validation failed. Stop.")
        return 2

    # -----------------------------
    # ranking (accepted ratio)
    # -----------------------------
    ranking_df: Optional[pd.DataFrame] = None
    try:
        r_files = find_latest_files(args.ranking_glob, args.ranking_n_files)
        info("using ranking files (latest first):")
        for p in r_files:
            info(f"  - {p}")
        ranking_df = read_concat_csv(r_files)
    except Exception as e:
        warn(f"ranking CSV not available: {e}")

    current_accepted_ratio = accepted_ratio_from_ranking(
        ranking_df,
        accepted_col=args.accepted_col,
        secondary_col=args.secondary_accepted_col,
        mode=args.accepted_mode,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # JSON params 読み込み（あれば優先）
    json_params: Dict[str, Thresholds] = {}
    if args.params_json:
        pj = Path(args.params_json)
        with pj.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for metric, spec in data.items():
            if "thresholds" not in spec:
                raise ValueError(f"params_json metric '{metric}' missing 'thresholds'")
            json_params[metric] = Thresholds.from_list(spec["thresholds"])

    comparison_rows: List[Dict[str, Any]] = []
    metric_summary: List[Dict[str, Any]] = []
    chosen_params_out: Dict[str, Dict[str, Any]] = {}

    # new accepted 用に new score を results df に反映していく
    df_new_scores = df.copy()

    for score_col in TARGET_SCORE_COLS:
        metric = score_col.replace("_score", "")

        raw_col, raw_source, raw_direction, raw_transform, direction_meta = (
            resolve_raw_col(df, score_col)
        )
        higher_is_better = raw_direction == "higher_is_better"
        raw_note = raw_source  # 互換（必要なら別文字列でもOK）

        if raw_col is None:
            warn(f"Skip '{metric}': missing required raw column (raw resolve failed)")
            continue

        # 必須カラム存在チェック（raw併存必須）
        ok, reason = validate_score_column(
            df,
            score_col,
            raw_col,
            min_samples=args.min_samples,
            min_discrete_ratio=args.min_discrete_ratio,
        )
        if not ok:
            warn(f"Skip '{metric}': {reason}")
            continue

        is_tech = score_col in TECH_SCORE_COLS
        is_face = score_col in FACE_SCORE_COLS

        raw = df[raw_col]
        cur_score = df[score_col]
        raw_num = coerce_numeric(raw)

        # -----------------------------
        # 現状分布
        # -----------------------------
        cur_ratios = compute_score_ratios(cur_score)
        cur_counts = compute_score_counts(cur_score)
        cur_sat = saturation_ratio(cur_ratios)
        cur_l1 = tech_target_l1(cur_ratios) if is_tech else None
        cur_flags = tech_target_flags(cur_ratios) if is_tech else None

        # -----------------------------
        # 閾値決定
        # -----------------------------
        thr_source = "none"
        thr: Optional[Thresholds] = None

        if metric in json_params:
            thr = json_params[metric]
            thr_source = "json"
        else:
            if is_tech:
                thr = propose_thresholds_for_target_distribution(raw_num)
                thr_source = "auto_target_quantiles_12.5_32.5_62.5_87.5"
            elif is_face and args.apply_face_auto:
                thr = propose_thresholds_for_target_distribution(raw_num)
                thr_source = "auto_target_quantiles_12.5_32.5_62.5_87.5"
            else:
                thr = None
                thr_source = "skip_auto"

        # -----------------------------
        # direction sanity check (detect only)
        # -----------------------------
        inferred_hib, corr, n_corr, inferred = infer_direction_by_score(
            raw_num, cur_score
        )

        if inferred and (inferred_hib != higher_is_better):
            tag = "CONTRACT" if metric == "blurriness" else "WARN"
            warn(
                f"[{tag}] direction mismatch: metric={metric} "
                f"expected_higher_is_better={higher_is_better} inferred_higher_is_better={inferred_hib} "
                f"corr={corr} n={n_corr}"
            )

        direction_meta = (
            dict(direction_meta) if isinstance(direction_meta, dict) else {}
        )
        direction_meta.update(
            {
                "direction_inferred": bool(inferred),
                "corr": corr,
                "n_for_corr": int(n_corr),
                "direction_note": (
                    (
                        "corr_check_mismatch"
                        if (inferred and (inferred_hib != higher_is_better))
                        else "corr_check"
                    )
                    if inferred
                    else "corr_check_skipped"
                ),
            }
        )

        raw_spec = build_raw_transform_spec(
            metric,
            raw_col,
            higher_is_better=higher_is_better,
            raw_source=raw_source,
            direction_meta=direction_meta,
        )

        # -----------------------------
        # 新スコア計算
        # -----------------------------
        if thr is not None:
            new_score = raw_num.map(
                lambda x: score_from_raw(x, thr, higher_is_better=higher_is_better)
            )
            thresholds_out: Optional[List[float]] = thr.to_list()
        else:
            new_score = coerce_numeric(cur_score)
            thresholds_out = None

        # new accepted のために results df の score 列を new_score に置換
        if score_col in df_new_scores.columns:
            df_new_scores[score_col] = pd.to_numeric(new_score, errors="coerce")

        note = ""
        if metric == "noise":
            note = "noise thresholds here are suggestions only; apply tool converts them to good_sigma/warn_sigma"

        chosen_params_out[metric] = {
            "raw_col": raw_col,
            "score_col": score_col,
            "thresholds": thresholds_out,
            "source": thr_source,
            "raw_source": raw_source,
            "higher_is_better": higher_is_better,
            "raw_spec": raw_spec,
            **({"note": note} if note else {}),
        }

        # -----------------------------
        # 新分布
        # -----------------------------
        new_ratios = compute_score_ratios(new_score)
        new_counts = compute_score_counts(new_score)
        new_sat = saturation_ratio(new_ratios)
        new_l1 = tech_target_l1(new_ratios) if is_tech else None
        new_flags = tech_target_flags(new_ratios) if is_tech else None

        # -----------------------------
        # face 極端寄り監視
        # -----------------------------
        if is_face and (
            cur_sat >= args.face_saturation_warn or new_sat >= args.face_saturation_warn
        ):
            warn(
                f"Face metric '{metric}' extreme? "
                f"saturation(0+1): current={cur_sat:.3f}, new={new_sat:.3f} "
                f"(warn>= {args.face_saturation_warn})"
            )

        # -----------------------------
        # long 形式
        # -----------------------------
        for sv in DISCRETE_SCORES:
            comparison_rows.append(
                {
                    "metric": metric,
                    "which": "current",
                    "score": sv,
                    "count": cur_counts.get(sv, 0),
                    "ratio": cur_ratios.get(sv, 0.0),
                }
            )
            comparison_rows.append(
                {
                    "metric": metric,
                    "which": "new",
                    "score": sv,
                    "count": new_counts.get(sv, 0),
                    "ratio": new_ratios.get(sv, 0.0),
                }
            )

        # -----------------------------
        # summary (#701-5)
        # -----------------------------
        direction_final = decide_direction_final(
            higher_is_better=higher_is_better,
            direction_inferred=bool(raw_spec.get("direction_inferred", False)),
            corr=raw_spec.get("corr", ""),
            n_for_corr=int(raw_spec.get("n_for_corr", 0)),
            direction_note=str(raw_spec.get("direction_note", "")),
        )

        row: Dict[str, Any] = {
            "metric": metric,
            "type": "tech" if is_tech else ("face" if is_face else "other"),
            "score_col": score_col,
            "raw_col": raw_col,
            "raw_source": raw_source,
            "higher_is_better": higher_is_better,
            "threshold_source": thr_source,
            "thresholds": thr.to_list() if thr is not None else "",
            "current_saturation_0plus1": float(cur_sat),
            "new_saturation_0plus1": float(new_sat),
            "delta_saturation_0plus1": float(new_sat - cur_sat),
            "current_discrete_ratio": float(discrete_ratio(cur_score)),
            "new_discrete_ratio": float(discrete_ratio(new_score)),
            "current_in_range_ratio": float(in_range_ratio(cur_score)),
            "new_in_range_ratio": float(in_range_ratio(new_score)),
            "raw_resolve": raw_note,
            "current_accepted_ratio": _nan_to_empty(current_accepted_ratio),
            "new_accepted_ratio": "",  # 後でまとめて計算して埋める
            "direction_final": direction_final,
            "direction_inferred": raw_spec.get("direction_inferred", False),
            "direction_corr": raw_spec.get("corr", ""),
            "direction_n_for_corr": raw_spec.get("n_for_corr", 0),
            "direction_note": raw_spec.get("direction_note", ""),
            "raw_transform": raw_spec.get("raw_transform", ""),
            "effective_raw_col": raw_spec.get("effective_raw_col", ""),
            "source_raw_col": raw_spec.get("source_raw_col", ""),
        }

        if is_tech:
            row["current_tech_target_l1"] = float(cur_l1)
            row["new_tech_target_l1"] = float(new_l1)
            row["delta_tech_target_l1"] = float(new_l1 - cur_l1)

            row["current_target_match_ratio"] = _nan_to_empty(
                target_match_ratio(cur_flags)
            )
            row["new_target_match_ratio"] = _nan_to_empty(target_match_ratio(new_flags))

            for sv in DISCRETE_SCORES:
                row[f"current_target_flag_{sv}"] = cur_flags[sv]
                row[f"new_target_flag_{sv}"] = new_flags[sv]
        else:
            row["current_tech_target_l1"] = ""
            row["new_tech_target_l1"] = ""
            row["delta_tech_target_l1"] = ""
            row["current_target_match_ratio"] = ""
            row["new_target_match_ratio"] = ""
            for sv in DISCRETE_SCORES:
                row[f"current_target_flag_{sv}"] = ""
                row[f"new_target_flag_{sv}"] = ""

        metric_summary.append(row)

        # -----------------------------
        # plots
        # -----------------------------
        make_plots(plots_dir, metric, raw_num, cur_score, new_score)

        # -----------------------------
        # console
        # -----------------------------
        info("-" * 72)
        info(f"[METRIC] {metric}")
        info(
            f"  raw={raw_col} raw_source={raw_source} higher_is_better={higher_is_better} "
            f"thr_source={thr_source} thresholds={thr.to_list() if thr else None}"
        )

        info("  score : current_ratio -> new_ratio (delta)")
        for sv in DISCRETE_SCORES:
            c = cur_ratios.get(sv, 0.0)
            n = new_ratios.get(sv, 0.0)
            d = n - c
            info(f"  {sv:>4} : {c:6.3f} -> {n:6.3f} ({d:+6.3f})")

        if is_tech:
            info(f"  tech_target_l1: {cur_l1:.6f} -> {new_l1:.6f}")
            info(
                f"  target_match_ratio: {_nan_to_empty(target_match_ratio(cur_flags))} -> {_nan_to_empty(target_match_ratio(new_flags))}"
            )

        if metric == "noise":
            info(
                f"  noise_direction: inferred={raw_spec.get('direction_inferred')} "
                f"corr={raw_spec.get('corr')} n={raw_spec.get('n_for_corr')} "
                f"note={raw_spec.get('direction_note')}"
            )

    if not comparison_rows:
        warn("No valid metrics produced outputs (all skipped by checks).")
        return 2

    # -----------------------------
    # new accepted ratio (best-effort)
    # -----------------------------
    thresholds_for_acceptance = build_thresholds_dict_for_acceptance(chosen_params_out)
    new_accepted_ratio = compute_new_accepted_ratio_via_acceptance_from_results(
        df_new_scores,
        thresholds_for_acceptance=thresholds_for_acceptance,
    )

    # -----------------------------
    # #701-5 validation (SSOT)
    # -----------------------------
    criteria = Criteria7015(
        sat_max=args.validate_sat_max,
        accepted_delta_max=args.validate_accepted_delta_max,
        discrete_min=args.validate_discrete_min,
        in_range_min=args.validate_in_range_min,
        require_target_l1_improve=bool(args.validate_require_target_l1_improve),
        direction_conflict_fail=bool(args.validate_direction_conflict_fail),
    )

    for row in metric_summary:
        row["new_accepted_ratio"] = _nan_to_empty(new_accepted_ratio)
        v_pass, v_reasons = validate_row_701_5(row, criteria=criteria)
        row["validation_pass"] = bool(v_pass)
        row["validation_fail_reasons"] = ";".join(v_reasons)

    # save params
    params_path = out_dir / "new_params_used.json"
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(chosen_params_out, f, ensure_ascii=False, indent=2)

    # ---- emit evaluator config (yaml/json) ----
    if args.emit_config_yaml or args.emit_config_json:
        cfg = build_evaluator_config_from_chosen_params(chosen_params_out)

        if args.emit_config_yaml:
            out_yaml = Path(args.emit_config_yaml)
            out_yaml.parent.mkdir(parents=True, exist_ok=True)
            if yaml is not None:
                with out_yaml.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            else:
                with out_yaml.open("w", encoding="utf-8") as f:
                    f.write(dump_yaml_like(cfg))
            info(f"  emit-config-yaml : {out_yaml}")

        if args.emit_config_json:
            out_json = Path(args.emit_config_json)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            info(f"  emit-config-json : {out_json}")

    # save distribution long/wide
    comp_df = pd.DataFrame(comparison_rows)
    long_csv = out_dir / "score_distribution_long.csv"
    comp_df.to_csv(long_csv, index=False)

    cur_df = (
        comp_df[comp_df["which"] == "current"]
        .rename(columns={"count": "current_count", "ratio": "current_ratio"})
        .drop(columns=["which"])
    )
    new_df = (
        comp_df[comp_df["which"] == "new"]
        .rename(columns={"count": "new_count", "ratio": "new_ratio"})
        .drop(columns=["which"])
    )
    wide = pd.merge(cur_df, new_df, on=["metric", "score"], how="outer").fillna(0)
    wide["delta_ratio"] = wide["new_ratio"] - wide["current_ratio"]
    wide_csv = out_dir / "score_distribution_wide.csv"
    wide.to_csv(wide_csv, index=False)

    # save metric summary
    summary_df = pd.DataFrame(metric_summary)
    summary_csv = out_dir / "metric_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    info("=" * 80)
    info("[DONE]")
    info(f"  out_dir : {out_dir}")
    info(f"  params  : {params_path}")
    info(f"  long    : {long_csv}")
    info(f"  wide    : {wide_csv}")
    info(f"  summary : {summary_csv}")
    info(f"  plots   : {plots_dir}")
    info(
        f"  accepted: current={_nan_to_empty(current_accepted_ratio)} new={_nan_to_empty(new_accepted_ratio)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
