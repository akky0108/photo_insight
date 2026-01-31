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
    # 目推定の信頼性ガード（小さすぎる顔は誤爆しやすいので無視）
    eye_patch_min: int = 70,
    # 半目NG帯（要件：半目は採用不可）
    half_min: float = 0.85,
    # 完全閉眼（要件：完全閉眼は注意）
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
            # データが無ければ判定しない（減点もしない）
            continue

        if patch_size < eye_patch_min:
            r["eye_state"] = "unknown"
            continue

        # 半目NG（採用不可）
        if half_min <= closed_prob < closed_min:
            r["eye_state"] = "half"
            r["accepted_flag"] = 0
            r["secondary_accept_flag"] = 0

            tag = f"EYE_HALF_NG(p={format_score(closed_prob)},sz={patch_size})"
            r["accepted_reason"] = tag
            r["secondary_accept_reason"] = ""
            continue

        # 完全閉眼（注意：採用は維持）
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

    # 0..1 系（スコア列）
    eye = safe_float(row.get("eye_contact_score"))
    face_dir = safe_float(row.get("face_direction_score"))
    framing = safe_float(row.get("framing_score"))
    face_pos = safe_float(row.get("face_position_score"))
    rule = safe_float(row.get("composition_rule_based_score"))
    thirds = safe_float(row.get("rule_of_thirds_score"))
    lead = safe_float(row.get("lead_room_score"))
    body = safe_float(row.get("body_composition_score"))

    # 表情（debug_* があるなら優先）
    expr = safe_float(row.get("debug_expr_effective", row.get("debug_expression", row.get("expression_score", 0.0))))
    half_pen = safe_float(row.get("debug_half_penalty", 0.0))

    tags: List[Tuple[str, float]] = []

    if cat == "portrait":
        if st in ("full_body", "seated"):
            # 全身/座り：構図/収まり優先（eyeは重視しない）
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

            # 顔は“良いなら補足”
            if f >= 78.0:
                tags.append(("face_strong", f / 100.0))

        else:
            # face_only / upper_body：顔＋構図の両立
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

            # 表情（半目代理ペナルティが強い場合は控えめ）
            if expr >= 0.75 and half_pen <= 0.35:
                tags.append(("expression", _clamp01(expr)))
            elif half_pen >= 0.60:
                tags.append(("eye_state_risk", 0.60))
    else:
        # non_face：構図中心
        if framing >= 0.85:
            tags.append(("framing", framing))
        if rule >= 0.85:
            tags.append(("composition", rule))
        if thirds >= 0.85:
            tags.append(("thirds", thirds))
        if lead >= 0.80:
            tags.append(("lead_room", lead))

    # 主戦場（最後に1つだけ）
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

    # ---- flag（候補ピック）----
    flag_ratio: float = 0.30
    flag_min_total: int = 10


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
            # 全身/座り：構図主導。顔条件を緩和
            return (comp >= 65.0 and face >= 60.0) or (comp >= 58.0 and safe_float(r.get("overall_score")) >= 72.0)

        # face_only / upper_body：顔と構図の両立（compは少し緩めて落としにくく）
        return (face >= 70.0 and comp >= 55.0) or (face >= 78.0 and comp >= 50.0)

    # --------------------------
    # backfill relaxed gate（埋め用：少し緩める）
    # --------------------------
    def _backfill_relaxed_ok(self, r: Row) -> bool:
        """
        strict gate で green_total に満たない場合の救済ゲート。
        - non_face は True（上位順で埋める）
        - portrait:
          - full_body/seated: comp>=58
          - upper_body/face_only: face>=68 & comp>=50
        """
        if r.get("category") != "portrait":
            return True

        st = _shot_type(r)
        face = safe_float(r.get("score_face"))
        comp = safe_float(r.get("score_composition"))

        if st in ("full_body", "seated"):
            return comp >= 58.0
        return (face >= 68.0 and comp >= 50.0)

    def _mark_green(self, r: Row, *, green_rank: int, green_total: int, suffix: str = "") -> None:
        """
        Green を付与する統一関数。
        - Green にしたら secondary は必ず落とす（運用の見た目が綺麗）
        - accepted_reason は pro 形式を採用し、suffix を付与できる
        """
        r["accepted_flag"] = 1
        r["secondary_accept_flag"] = 0
        r["secondary_accept_reason"] = ""

        base = build_reason_pro(
            r,
            green_rank=green_rank,
            green_total=green_total,
            max_tags=3,
        )
        if suffix:
            r["accepted_reason"] = f"{base} | {suffix}"
        else:
            r["accepted_reason"] = base

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

        # ★可変Green比率（150枚なら 0.20 → 30枚）
        green_ratio = self._green_ratio_by_count(total_n)

        green_total = max(
            int(math.ceil(total_n * green_ratio)),
            int(self.rules.green_min_total),
        )

        sorted_all = sorted(rows, key=overall_sort_key, reverse=True)

        accepted_count = 0

        # --- 1st pass: strict gate ---
        for r in sorted_all:
            if accepted_count >= green_total:
                break
            if not self._green_content_ok(r):
                continue

            accepted_count += 1
            self._mark_green(r, green_rank=accepted_count, green_total=green_total)

        # --- 2nd pass: relaxed backfill ---
        if accepted_count < green_total:
            for r in sorted_all:
                if accepted_count >= green_total:
                    break
                if safe_int_flag(r.get("accepted_flag")) == 1:
                    continue
                if not self._backfill_relaxed_ok(r):
                    continue

                accepted_count += 1
                self._mark_green(r, green_rank=accepted_count, green_total=green_total, suffix="FILL_RELAXED")

        # --- 3rd pass: forced backfill (quota guarantee) ---
        if accepted_count < green_total:
            for r in sorted_all:
                if accepted_count >= green_total:
                    break
                if safe_int_flag(r.get("accepted_flag")) == 1:
                    continue

                accepted_count += 1
                self._mark_green(r, green_rank=accepted_count, green_total=green_total, suffix="FILL_FORCED")

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
                # accepted_reason は運用上 “一つにまとめて見たい” を維持
                r["accepted_reason"] = str(r.get("secondary_accept_reason") or "")

            elif r.get("category") == "non_face" and overall >= non_face_sec_thr:
                r["secondary_accept_flag"] = 1
                r["secondary_accept_reason"] = (
                    f"SEC:non_face overall={format_score(overall)} "
                    f"c={format_score(safe_float(r.get('score_composition')))}"
                )
                r["accepted_reason"] = str(r.get("secondary_accept_reason") or "")

        # ---- 目状態ポリシー（half=強制NG / closed=注意）を最後に適用 ----
        apply_eye_state_policy(rows)

        # --------------------------
        # （ログ用）primary percentile thresholds
        #   ※Green判定には使っていないが、運用・可視化のため返す
        # --------------------------
        portrait_scores = [safe_float(r.get("overall_score")) for r in rows if r.get("category") == "portrait"]
        non_face_scores = [safe_float(r.get("overall_score")) for r in rows if r.get("category") == "non_face"]

        portrait_thr = percentile(portrait_scores, self.rules.portrait_percentile) if portrait_scores else 0.0
        non_face_thr = percentile(non_face_scores, self.rules.non_face_percentile) if non_face_scores else 0.0

        # secondary thresholds（既存）
        portrait_sec_thr = percentile(portrait_scores, self.rules.portrait_secondary_percentile) if portrait_scores else 0.0
        non_face_sec_thr = percentile(non_face_scores, self.rules.non_face_secondary_percentile) if non_face_scores else 0.0

        # ---- thresholds返り値：旧キー互換（ログ/他依存の保険）----
        # primary percentile はこのエンジンでは使っていないが、参照されても壊れないように返す
        return {
            "portrait": float(portrait_thr),
            "non_face": float(non_face_thr),
            "portrait_secondary": float(portrait_sec_thr),
            "non_face_secondary": float(non_face_sec_thr),
            "green_total": float(green_total),
            # ★追加（互換を壊さない）
            "green_ratio_effective": float(green_ratio),
            "total_n": float(total_n),
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
