# src/batch_processor/evaluation_rank/acceptance.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

Row = Dict[str, Any]


# ==============================
# Utilities
# ==============================

def safe_float(value: Any) -> float:
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def format_score(x: float) -> float:
    return round(float(x), 2)


def percentile(values: List[Any], p: float) -> float:
    """
    numpy なし percentile（線形補間）。
    - values に "" / None / 非数値が混ざっても壊れない
    - p: 0〜100
    """
    xs: List[float] = []
    for v in values:
        if v in ("", None):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv) or math.isinf(fv):
            continue
        xs.append(fv)

    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return xs[0]

    xs.sort()
    p = max(0.0, min(100.0, float(p)))
    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def extract_contrib(prefix: str, r: Row) -> Dict[str, Any]:
    """
    prefix は "contrib_tech_" / "contrib_face_" / "contrib_comp_" を想定。
    prefix を外したキー -> 値 の Dict にして返す。
    """
    out: Dict[str, Any] = {}
    for k, v in r.items():
        if not isinstance(k, str):
            continue
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


def _bool(value: Any) -> bool:
    """
    CSV の "TRUE"/"False"/"1"/"" 対応のための bool 正規化。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "t", "yes", "y"):
            return True
        if v in ("0", "false", "f", "no", "n", ""):
            return False
    try:
        return bool(int(value))
    except Exception:
        return False


def safe_int_flag(value: Any) -> int:
    return 1 if _bool(value) else 0


def reliability_score(row: Row) -> float:
    """
    コントラクトに基づく「測定信頼度」(0..5)。
    score は落とさず主指標として使い、同点付近の tie-break に使う。
    """
    s = 0.0
    for k in ("noise_eval_status", "exposure_eval_status", "sharpness_eval_status", "contrast_eval_status"):
        if str(row.get(k, "")).strip().lower() == "ok":
            s += 1.0

    blur_ok = (
        _bool(row.get("success"))
        or _bool(row.get("blurriness_success"))
        or _bool(row.get("blurriness_ok"))
        or _bool(row.get("blurriness_eval_ok"))
    )
    if blur_ok:
        s += 1.0

    return s


def overall_sort_key(row: Row) -> Tuple[float, float, float, float, float, str]:
    """
    overall_score を軸にしつつ、サブスコア＋信頼度で tie-break するソートキー。
    """
    overall = safe_float(row.get("overall_score"))
    face = safe_float(row.get("score_face"))
    comp = safe_float(row.get("score_composition"))
    tech = safe_float(row.get("score_technical"))
    rel = reliability_score(row)
    fname = str(row.get("file_name") or row.get("filename") or "")
    return (overall, face, comp, tech, rel, fname)


# ==============================
# Eye state policy (half/closed)
# ==============================

def apply_eye_state_policy(
    rows: List[Row],
    *,
    eye_patch_min: int = 70,
    half_min: float = 0.85,
    closed_min: float = 0.98,
) -> None:
    """
    目状態ポリシー（最終上書き）
    - half_min <= eye_closed_prob < closed_min : 半目 → accepted_flag=0 & secondary_accept_flag=0（採用不可）
    - eye_closed_prob >= closed_min            : 完全閉眼 → 注意（採用は落とさない / Yellow推奨はLR側で）
    - eye_patch_size < eye_patch_min           : 信頼性不足 → unknown扱い（何もしない）
    期待する入力:
      row 直下に 'eye_closed_prob_best','eye_patch_size_best' が入っている
      （無ければ何もしない）
    """

    def _to_float(v: Any, default: Optional[float] = 0.0) -> Optional[float]:
        try:
            if v in ("", None):
                return default
            return float(v)
        except (TypeError, ValueError):
            return default

    def _to_int(v: Any, default: Optional[int] = 0) -> Optional[int]:
        try:
            if v in ("", None):
                return default
            return int(float(v))
        except (TypeError, ValueError):
            return default

    for r in rows:
        if str(r.get("category")) != "portrait":
            continue
        if not _bool(r.get("face_detected")):
            continue

        closed_prob = _to_float(r.get("eye_closed_prob_best"), None)
        patch_size = _to_int(r.get("eye_patch_size_best"), None)
        if closed_prob is None or patch_size is None:
            continue

        if patch_size < eye_patch_min:
            r["eye_state"] = "unknown"
            continue

        if half_min <= closed_prob < closed_min:
            r["eye_state"] = "half"
            r["accepted_flag"] = 0
            r["secondary_accept_flag"] = 0

            tag = f"EYE_HALF_NG(p={format_score(closed_prob)},sz={patch_size})"
            r["accepted_reason"] = tag
            r["secondary_accept_reason"] = ""
            continue

        if closed_prob >= closed_min:
            r["eye_state"] = "closed"
            tag = f"EYE_CLOSED_WARN(p={format_score(closed_prob)},sz={patch_size})"
            base = (r.get("accepted_reason") or "").strip()
            r["accepted_reason"] = f"{base} | {tag}" if base else tag
            continue

        r["eye_state"] = "ok"


# ==============================
# accepted_reason (pro tags)
# ==============================

def _shot_type(row: Row) -> str:
    return str(row.get("shot_type") or "").strip().lower()


def _topk_by_score(d: Dict[str, float], *, k: int) -> List[Tuple[str, float]]:
    items = [(kk, vv) for kk, vv in d.items() if vv is not None]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:k]


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def build_reason_pro(
    row: Row,
    *,
    green_rank: int,
    green_total: int,
    max_tags: int = 3,
) -> str:
    """
    プロ目線の accepted_reason（短文）を生成する。
    - 技術は A方式（伸びすぎない）前提なので「強みタグ」に入れない
    - 顔/構図を中心に「何が良いか」をタグ化
    """
    cat = str(row.get("category") or "")
    grp = str(row.get("accept_group") or "")
    st = _shot_type(row)

    overall = safe_float(row.get("overall_score"))
    f = safe_float(row.get("score_face"))
    c = safe_float(row.get("score_composition"))
    t = safe_float(row.get("score_technical"))

    eye = safe_float(row.get("eye_contact_score"))
    face_dir = safe_float(row.get("face_direction_score"))
    framing = safe_float(row.get("framing_score"))
    face_pos = safe_float(row.get("face_position_score"))
    rule = safe_float(row.get("composition_rule_based_score"))
    thirds = safe_float(row.get("rule_of_thirds_score"))
    lead = safe_float(row.get("lead_room_score"))
    body = safe_float(row.get("body_composition_score"))

    expr = safe_float(row.get("debug_expr_effective", row.get("debug_expression", row.get("expression_score", 0.0))))
    half_pen = safe_float(row.get("debug_half_penalty", 0.0))

    tags: List[Tuple[str, float]] = []

    if cat == "portrait":
        if st in ("full_body", "seated"):
            if framing >= 0.85:
                tags.append(("framing", framing))
            if body >= 0.80:
                tags.append(("body_fit", body))
            if lead >= 0.80:
                tags.append(("lead_room", lead))
            if rule >= 0.85:
                tags.append(("composition", rule))
            if thirds >= 0.85:
                tags.append(("thirds", thirds))
            if f >= 78.0:
                tags.append(("face_strong", f / 100.0))
        else:
            if eye >= 0.85:
                tags.append(("eye_contact", eye))
            if face_dir >= 0.85:
                tags.append(("face_direction", face_dir))
            if face_pos >= 0.85:
                tags.append(("face_position", face_pos))
            if framing >= 0.85:
                tags.append(("framing", framing))
            if rule >= 0.85:
                tags.append(("composition", rule))
            if thirds >= 0.85:
                tags.append(("thirds", thirds))

            if expr >= 0.75 and half_pen <= 0.35:
                tags.append(("expression", _clamp01(expr)))
            elif half_pen >= 0.60:
                tags.append(("eye_state_risk", 0.60))
    else:
        if framing >= 0.85:
            tags.append(("framing", framing))
        if rule >= 0.85:
            tags.append(("composition", rule))
        if thirds >= 0.85:
            tags.append(("thirds", thirds))
        if lead >= 0.80:
            tags.append(("lead_room", lead))

    focus = "face" if (cat == "portrait" and st not in ("full_body", "seated") and f >= c) else "comp"
    top = _topk_by_score({k: v for k, v in tags}, k=max_tags)
    top_txt = ", ".join(str(k) for k, _ in top if isinstance(k, str)) if top else ""

    return (
        f"{cat} group={grp} st={st} "
        f"rank={green_rank}/{green_total} "
        f"o={format_score(overall)} f={format_score(f)} c={format_score(c)} t={format_score(t)} "
        f"focus={focus} tags={top_txt}"
    )


# ==============================
# Rules
# ==============================

@dataclass(frozen=True)
class AcceptRules:
    # ---- 必須（バッチ側ログ互換/将来拡張の保険）----
    portrait_percentile: float = 70.0
    non_face_percentile: float = 80.0

    # ---- secondary（Yellow）は percentile ベース ----
    portrait_secondary_percentile: float = 60.0
    non_face_secondary_percentile: float = 70.0

    # ---- Green（最終採用）----
    # 旧：固定比率（互換用に残す）
    green_ratio_total: float = 0.20
    green_min_total: int = 3

    # ★新：枚数別の比率（デフォルトで有効化）
    green_ratio_small: float = 0.30   # n <= green_count_small_max
    green_ratio_mid: float = 0.25     # (small_max < n <= mid_max)
    green_ratio_large: float = 0.20   # n > green_count_mid_max
    green_count_small_max: int = 60
    green_count_mid_max: int = 120

    # ★新：accept_group ごとの配分（偏り防止）
    green_per_group_enabled: bool = True
    green_per_group_min_each: int = 1

    # ---- flag（候補ピック）----
    flag_ratio: float = 0.30
    flag_min_total: int = 10

    # ---- eye policy (half/closed) ----
    eye_patch_min: int = 70
    eye_half_min: float = 0.85
    eye_closed_min: float = 0.98


# ==============================
# Engine
# ==============================

class AcceptanceEngine:
    def __init__(self, rules: Optional[AcceptRules] = None) -> None:
        self.rules = rules or AcceptRules()

    def _green_ratio_by_count(self, n: int) -> float:
        """
        グループ枚数 n に応じて Green 比率を切り替える。
        デフォルト: <=60:30%, <=120:25%, それ以上:20%
        """
        if n <= int(self.rules.green_count_small_max):
            return float(self.rules.green_ratio_small)
        if n <= int(self.rules.green_count_mid_max):
            return float(self.rules.green_ratio_mid)
        return float(self.rules.green_ratio_large)

    def _assign_accept_group(self, r: Row) -> str:
        if not _bool(r.get("face_detected")):
            return "non_face"

        gid = str(r.get("group_id") or "").strip().upper()
        sid = str(r.get("subgroup_id") or "").strip()
        st = _shot_type(r)

        if gid in ("A", "B") and sid:
            return f"{gid}-{sid}"

        if st in ("full_body", "seated"):
            return "B-1"
        return "A-1"

    def assign_category(self, rows: List[Row]) -> None:
        for r in rows:
            r["category"] = "portrait" if _bool(r.get("face_detected")) else "non_face"
            r["accept_group"] = self._assign_accept_group(r)

    # --------------------------
    # Green content gate（全身/上半身を落としにくく）
    # --------------------------
    def _green_content_ok(self, r: Row) -> bool:
        if r.get("category") != "portrait":
            return True

        st = _shot_type(r)
        face = safe_float(r.get("score_face"))
        comp = safe_float(r.get("score_composition"))

        if st in ("full_body", "seated"):
            return (comp >= 65.0 and face >= 60.0) or (comp >= 58.0 and safe_float(r.get("overall_score")) >= 72.0)

        return (face >= 70.0 and comp >= 55.0) or (face >= 78.0 and comp >= 50.0)

    # --------------------------
    # Green relaxed gate（不足時の救済）
    # --------------------------
    def _green_relaxed_ok(self, r: Row) -> bool:
        """
        strict gate で Green が埋まらない時の救済。
        """
        if r.get("category") != "portrait":
            return True

        st = _shot_type(r)
        face = safe_float(r.get("score_face"))
        comp = safe_float(r.get("score_composition"))
        overall = safe_float(r.get("overall_score"))

        if st in ("full_body", "seated"):
            return (comp >= 60.0 and face >= 58.0) or (comp >= 55.0 and overall >= 70.0)

        # テスト側想定: face>=68 & comp>=50
        return (face >= 68.0 and comp >= 50.0) or (face >= 75.0 and comp >= 48.0)

    def _green_eye_half_ng(self, r: Row) -> bool:
        """
        half目NGを「選定前」に回避するための軽量チェック。
        apply_eye_state_policy と同じ閾値を使う（データがなければNGにしない）。
        """
        if str(r.get("category")) != "portrait":
            return False
        if not _bool(r.get("face_detected")):
            return False

        prob = r.get("eye_closed_prob_best")
        sz = r.get("eye_patch_size_best")
        if prob in ("", None) or sz in ("", None):
            return False

        try:
            p = float(prob)
            s = int(float(sz))
        except Exception:
            return False

        if s < int(self.rules.eye_patch_min):
            return False

        half_min = float(self.rules.eye_half_min)
        closed_min = float(self.rules.eye_closed_min)
        return (half_min <= p < closed_min)

    def _green_policy_ok(self, r: Row) -> bool:
        return self._green_content_ok(r) and (not self._green_eye_half_ng(r))

    def _green_policy_relaxed_ok(self, r: Row) -> bool:
        return self._green_relaxed_ok(r) and (not self._green_eye_half_ng(r))

    def _green_policy_forced_ok(self, r: Row) -> bool:
        """
        forced: content gate は基本無視。ただし「何でもGreen」にはしない。
        テスト期待:
          - face=72, comp=49 は forced で拾える
          - face=40, comp=40 は forced でも拾わない（Bでbackfillされるべき）
        """
        if self._green_eye_half_ng(r):
            return False

        if str(r.get("category")) != "portrait":
            return True

        # ★最低限の床（A=40/40を落としつつ、72/49は通す）
        face = safe_float(r.get("score_face"))
        comp = safe_float(r.get("score_composition"))
        overall = safe_float(r.get("overall_score"))

        # まず顔スコアが一定以上
        if face >= 60.0 and comp >= 45.0:
            return True

        # 例外：overallが高くても顔が低すぎるものは拾わない
        # （Aの90点でも face40 は落とす）
        if overall >= 85.0 and face >= 55.0 and comp >= 45.0:
            return True

        return False

    def _group_best_score(self, rows: List[Row]) -> float:
        if not rows:
            return -1e9
        return max(safe_float(r.get("overall_score")) for r in rows)

    def _compute_group_quotas(
        self,
        group_rows: Dict[str, List[Row]],
        *,
        green_total_global: int,
    ) -> Dict[str, int]:
        """
        accept_group ごとの green quota を計算し、合計が green_total_global になるように調整する。
        - 原則: quota_g = max(min_each, ceil(n_g * ratio_by_count(n_g)))
        - 合計が超える場合: quota>min_each のグループから削って調整
        - green_total_global < group数*min_each の場合:
            各グループの best overall が高い順に min_each を配り、それ以外は0
        """
        groups = list(group_rows.keys())
        if not groups or green_total_global <= 0:
            return {g: 0 for g in groups}

        min_each = max(0, int(self.rules.green_per_group_min_each))

        quotas: Dict[str, int] = {}
        for g, rs in group_rows.items():
            n_g = len(rs)
            ratio_g = self._green_ratio_by_count(n_g)
            q = int(math.ceil(n_g * float(ratio_g)))
            q = max(q, min_each)
            q = min(q, n_g)
            quotas[g] = q

        if green_total_global < (len(groups) * max(1, min_each)):
            ranked = sorted(
                groups,
                key=lambda gg: (self._group_best_score(group_rows.get(gg, [])), gg),
                reverse=True,
            )
            out = {g: 0 for g in groups}
            remain = green_total_global
            for g in ranked:
                if remain <= 0:
                    break
                give = min_each if min_each > 0 else 1
                out[g] = min(give, len(group_rows[g]))
                remain -= out[g]

            i = 0
            while remain > 0 and i < len(ranked):
                g = ranked[i]
                if out[g] < len(group_rows[g]):
                    out[g] += 1
                    remain -= 1
                else:
                    i += 1
            return out

        total = sum(quotas.values())
        if total > green_total_global:
            while total > green_total_global:
                candidates = [g for g in groups if quotas[g] > max(min_each, 1)]
                if not candidates:
                    candidates = [g for g in groups if quotas[g] > 0]
                    if not candidates:
                        break

                g_pick = max(
                    candidates,
                    key=lambda g: (quotas[g], len(group_rows[g]), self._group_best_score(group_rows[g])),
                )
                quotas[g_pick] -= 1
                total -= 1

        return quotas

    # --------------------------
    # accepted_flag（Green）
    # --------------------------
    def apply_accepted_flags(self, rows: List[Row]) -> Dict[str, float]:
        self.assign_category(rows)

        for r in rows:
            r["accepted_flag"] = 0
            r["secondary_accept_flag"] = 0
            r["accepted_reason"] = ""
            r["secondary_accept_reason"] = ""

        total_n = len(rows)
        green_ratio_global = self._green_ratio_by_count(total_n)

        green_total = max(
            int(math.ceil(total_n * float(green_ratio_global))),
            int(self.rules.green_min_total),
        )

        sorted_all = sorted(rows, key=overall_sort_key, reverse=True)

        accepted_count = 0

        def _mark_green(rr: Row, *, fill_tag: str = "") -> None:
            nonlocal accepted_count
            accepted_count += 1
            rr["accepted_flag"] = 1
            base = build_reason_pro(
                rr,
                green_rank=accepted_count,
                green_total=green_total,
                max_tags=3,
            )
            if fill_tag:
                # strictはsuffix無し。埋めは "FILL_*" を付ける（テスト要件）
                rr["accepted_reason"] = f"{base} | {fill_tag}"
            else:
                rr["accepted_reason"] = base

        # --------------------------
        # ★Per-group quota fill（偏り防止）
        #   ここでは「quotaを目標に試みる」が、埋まらない分は全体backfillに回す
        # --------------------------
        if bool(self.rules.green_per_group_enabled) and green_total > 0:
            group_rows: Dict[str, List[Row]] = {}
            for r in rows:
                g = str(r.get("accept_group") or "")
                group_rows.setdefault(g, []).append(r)

            quotas = self._compute_group_quotas(group_rows, green_total_global=green_total)

            for g, rs in group_rows.items():
                if accepted_count >= green_total:
                    break

                q = int(quotas.get(g, 0))
                if q <= 0:
                    continue

                rs_sorted = sorted(rs, key=overall_sort_key, reverse=True)

                def _take_from_group(pred, *, fill_tag: str) -> None:
                    nonlocal q
                    for rr in rs_sorted:
                        if accepted_count >= green_total or q <= 0:
                            break
                        if safe_int_flag(rr.get("accepted_flag")) == 1:
                            continue
                        if not pred(rr):
                            continue
                        _mark_green(rr, fill_tag=fill_tag)
                        q -= 1

                # strictはsuffix無し（＝FILLが混ざらない）
                _take_from_group(self._green_policy_ok, fill_tag="")
                # relaxed/forced は “埋め” なので FILL を付与
                _take_from_group(self._green_policy_relaxed_ok, fill_tag="FILL_RELAX")
                _take_from_group(self._green_policy_forced_ok, fill_tag="FILL_FORCED")

        # --------------------------
        # backfill（全体上位で埋める）: strict -> relaxed -> forced
        # --------------------------
        def _fill_remaining(pred, *, fill_tag: str) -> None:
            for rr in sorted_all:
                if accepted_count >= green_total:
                    break
                if safe_int_flag(rr.get("accepted_flag")) == 1:
                    continue
                if not pred(rr):
                    continue
                _mark_green(rr, fill_tag=fill_tag)

        _fill_remaining(self._green_policy_ok, fill_tag="")
        _fill_remaining(self._green_policy_relaxed_ok, fill_tag="FILL_RELAX")
        _fill_remaining(self._green_policy_forced_ok, fill_tag="FILL_FORCED")

        # --------------------------
        # secondary（Yellow）: percentile ベース（運用互換）
        # --------------------------
        portrait_scores = [safe_float(r.get("overall_score")) for r in rows if r.get("category") == "portrait"]
        non_face_scores = [safe_float(r.get("overall_score")) for r in rows if r.get("category") == "non_face"]

        portrait_sec_thr = percentile(portrait_scores, self.rules.portrait_secondary_percentile) if portrait_scores else 0.0
        non_face_sec_thr = percentile(non_face_scores, self.rules.non_face_secondary_percentile) if non_face_scores else 0.0

        for r in rows:
            if safe_int_flag(r.get("accepted_flag")) == 1:
                continue

            overall = safe_float(r.get("overall_score"))

            if r.get("category") == "portrait" and overall >= portrait_sec_thr:
                r["secondary_accept_flag"] = 1
                r["secondary_accept_reason"] = (
                    f"SEC:portrait group={r.get('accept_group')} "
                    f"overall={format_score(overall)} "
                    f"f={format_score(safe_float(r.get('score_face')))} "
                    f"c={format_score(safe_float(r.get('score_composition')))}"
                )
                r["accepted_reason"] = str(r.get("secondary_accept_reason") or "")

            elif r.get("category") == "non_face" and overall >= non_face_sec_thr:
                r["secondary_accept_flag"] = 1
                r["secondary_accept_reason"] = (
                    f"SEC:non_face overall={format_score(overall)} "
                    f"c={format_score(safe_float(r.get('score_composition')))}"
                )
                r["accepted_reason"] = str(r.get("secondary_accept_reason") or "")

        # ---- 目状態ポリシー（half=強制NG / closed=注意）を最後に適用 ----
        apply_eye_state_policy(
            rows,
            eye_patch_min=int(self.rules.eye_patch_min),
            half_min=float(self.rules.eye_half_min),
            closed_min=float(self.rules.eye_closed_min),
        )

        # --------------------------
        # （ログ用）primary percentile thresholds
        # --------------------------
        portrait_scores = [safe_float(r.get("overall_score")) for r in rows if r.get("category") == "portrait"]
        non_face_scores = [safe_float(r.get("overall_score")) for r in rows if r.get("category") == "non_face"]

        portrait_thr = percentile(portrait_scores, self.rules.portrait_percentile) if portrait_scores else 0.0
        non_face_thr = percentile(non_face_scores, self.rules.non_face_percentile) if non_face_scores else 0.0

        portrait_sec_thr = percentile(portrait_scores, self.rules.portrait_secondary_percentile) if portrait_scores else 0.0
        non_face_sec_thr = percentile(non_face_scores, self.rules.non_face_secondary_percentile) if non_face_scores else 0.0

        return {
            "portrait": float(portrait_thr),
            "non_face": float(non_face_thr),
            "portrait_secondary": float(portrait_sec_thr),
            "non_face_secondary": float(non_face_sec_thr),
            "green_total": float(green_total),
            "green_ratio_effective": float(green_ratio_global),
            "total_n": float(total_n),
            "green_per_group_enabled": 1.0 if bool(self.rules.green_per_group_enabled) else 0.0,
        }

    # --------------------------
    # flag（候補ピック）
    # --------------------------
    def apply_top_flag(self, rows: List[Row]) -> None:
        if not rows:
            return

        sorted_rows = sorted(rows, key=overall_sort_key, reverse=True)

        top_n = max(
            int(math.ceil(len(sorted_rows) * float(self.rules.flag_ratio))),
            int(self.rules.flag_min_total),
        )

        for i, r in enumerate(sorted_rows):
            r["flag"] = 1 if i < top_n else 0

    def run(self, rows: List[Row]) -> Dict[str, float]:
        thresholds = self.apply_accepted_flags(rows)
        self.apply_top_flag(rows)
        return thresholds
