from typing import Dict, Any, Tuple, Optional

from evaluators.brightness_compensation import (
    adjust_noise_by_brightness,
    adjust_blur_by_brightness,
)


def decide_accept(
    results: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[bool, str]:
    """
    ポートレート受け入れ判定ロジック。
    ノイズスコアが 0〜1 の 5段階離散 (1.0, 0.75, 0.5, 0.25, 0.0) を前提とした判定。
    """

    def _f(key: str, default: float = 0.0) -> float:
        v = results.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    def _pose_to_100(v: float) -> float:
        """
        pose_score が 0〜1 / 0〜100 どちらでも来てもよいように正規化する。
        """
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        if 0.0 <= v <= 1.0:
            return v * 100.0
        return v

    def _grade_to_score_like(grade: str) -> float | None:
        """
        grade 文字列から 0〜1 のスコア相当値にマッピング。
        未知のラベルは None（無視）とする。
        """
        if not grade:
            return None
        g = grade.strip().lower()

        if g in ("excellent", "very_good", "best"):
            return 1.0
        if g in ("good",):
            return 0.75
        if g in ("fair", "ok", "warn"):
            return 0.5
        if g in ("poor", "weak"):
            return 0.25
        if g in ("bad", "ng"):
            return 0.0
        return None

    def _ok_from(score: float, grade_score: float | None, thr: float) -> bool:
        """
        数値スコア or グレード由来スコアのどちらかが閾値を超えれば OK。
        """
        if score >= thr:
            return True
        if grade_score is not None and grade_score >= thr:
            return True
        return False

    def _thr(name: str, default: float) -> float:
        """
        閾値テーブルから取り出し。なければデフォルトを使う。
        """
        if thresholds is None:
            return default
        try:
            return float(thresholds.get(name, default))
        except (TypeError, ValueError):
            return default

    # ==============================
    # 共通スカラー値の抽出
    # ==============================
    noise_score = _f("noise_score", 0.0)
    face_noise_score = _f("face_noise_score", 0.0)
    blurriness_score = _f("blurriness_score", 0.0)
    face_blurriness_score = _f("face_blurriness_score", blurriness_score)
    exposure_score = _f("exposure_score", 0.0)
    pose_score_raw = _f("pose_score", 0.0)
    pose_score_100 = _pose_to_100(pose_score_raw)
    yaw = abs(_f("yaw", 0.0))
    pitch = abs(_f("pitch", 0.0))  # ★ 俯き／あおり用に追加
    contrast_score = _f("contrast_score", 0.0)

    # 構図系
    raw_comp_score = results.get("composition_score")
    if raw_comp_score is not None:
        try:
            composition_score = float(raw_comp_score)
        except (TypeError, ValueError):
            composition_score = _f("composition_rule_based_score", 0.0)
    else:
        composition_score = _f("composition_rule_based_score", 0.0)

    framing_score = _f("framing_score", 0.0)
    lead_room_score = _f("lead_room_score", 0.0)

    delta_face_sharpness = _f("delta_face_sharpness", -999.0)
    face_sharpness_score = _f("face_sharpness_score", 0.0)

    full_body_detected = bool(results.get("full_body_detected"))
    face_detected = bool(results.get("face_detected"))
    full_body_cut_risk = _f("full_body_cut_risk", 1.0)

    # フレーミング関連（上半身／椅子判定用）
    headroom_ratio = _f("headroom_ratio", 0.0)
    footroom_ratio = _f("footroom_ratio", 0.0)
    side_margin_min_ratio = _f("side_margin_min_ratio", 0.0)

    # 顔ボックスの高さ比
    face_box_height_ratio = _f("face_box_height_ratio", 0.0)

    # ポーズ BBox の縦サイズ比と重心位置（headroom/footroom から逆算）
    body_height_ratio = max(0.0, min(1.0, 1.0 - headroom_ratio - footroom_ratio))
    body_center_y_ratio = headroom_ratio + body_height_ratio / 2.0

    # ==============================
    # 簡易ショット種別推定（チューニング版）
    # ==============================
    def _estimate_shot_type() -> str:
        """
        簡易ショット分類:
          - "full_body"
          - "seated"      : 椅子に座り or しゃがみ系 full body に近い
          - "upper_body"  : 胸〜腰くらいまでの上半身（立ち/座り両方ありうる）
          - "face_only"   : 顔アップ寄り
          - "no_face"     : 顔検出なし
        """

        if not face_detected:
            return "no_face"

        # ---- 閾値群（YAML から上書き可能）----

        # full body を「否定」するための最低高さ・切れリスク
        fb_height_min = _thr("full_body_body_height_min", 0.30)
        fb_cut_risk_max_for_class = _thr("full_body_cut_risk_max_for_shot_type", 0.90)
        fb_footroom_min_for_class = _thr("full_body_footroom_min_for_shot_type", 0.00)

        # seated（座り）候補
        seated_center_min = _thr("seated_center_y_min", 0.50)
        seated_center_max = _thr("seated_center_y_max", 0.75)
        seated_foot_max = _thr("seated_footroom_max", 0.22)
        seated_height_min = _thr("seated_body_height_min", 0.30)

        # upper body 候補
        upper_body_head_min = _thr("upper_body_headroom_min", 0.15)
        upper_body_foot_max = _thr("upper_body_footroom_max", 0.15)

        # 顔アップ寄り判定
        face_only_height_max = _thr("face_only_body_height_max", 0.35)
        center_side_margin_min = _thr("center_side_margin_min", 0.02)

        # ---- full body 検出ありの場合 ----
        if full_body_detected:
            # まず「さすがに full body とは言いづらい」ケースだけ full_body から外す
            if (
                body_height_ratio < fb_height_min
                or full_body_cut_risk > fb_cut_risk_max_for_class
                or footroom_ratio < fb_footroom_min_for_class
            ):
                # 足元ほぼ無し＆頭上に余白 → 上半身寄り
                if (
                    headroom_ratio >= upper_body_head_min
                    and footroom_ratio <= upper_body_foot_max
                ):
                    return "upper_body"

                # 高さも小さく、中央寄せ → 顔アップ寄り
                if (
                    body_height_ratio <= face_only_height_max
                    and side_margin_min_ratio <= center_side_margin_min
                ):
                    return "face_only"

                # それ以外はとりあえず upper_body に寄せる
                return "upper_body"

            # ここまで来たら full_body 系として扱ってよい

            # ★ 先に「座り full body（seated）」判定を行う
            if (
                body_center_y_ratio >= seated_center_min
                and body_center_y_ratio <= seated_center_max
                and footroom_ratio <= seated_foot_max
                and body_height_ratio >= seated_height_min
            ):
                return "seated"

            # それ以外は立ち full body
            return "full_body"

        # ---- full body 検出なし（顔あり）の場合 ----
        # 上半身：頭上に余白あり＆足元余白少なめ
        if (
            headroom_ratio >= upper_body_head_min
            and footroom_ratio <= upper_body_foot_max
        ):
            return "upper_body"

        # 顔アップ：高さ控えめ & サイド余白少なめ（中央寄せ）
        if (
            body_height_ratio <= face_only_height_max
            and side_margin_min_ratio <= center_side_margin_min
        ):
            return "face_only"

        # デフォルトは face_only に寄せる
        return "face_only"


    shot_type = _estimate_shot_type()
    results["shot_type"] = shot_type  # ★ CSV にも残して分析できるようにしておく

    # ==============================
    # grade 系
    # ==============================
    noise_grade = str(results.get("noise_grade", "") or "")
    face_noise_grade = str(results.get("face_noise_grade", "") or "")

    noise_grade_score = _grade_to_score_like(noise_grade)
    face_noise_grade_score = _grade_to_score_like(face_noise_grade)

    contrast_grade = str(results.get("contrast_grade", "") or "")
    contrast_grade_score = _grade_to_score_like(contrast_grade)

    face_contrast_score = _f("face_contrast_score", contrast_score)
    face_contrast_grade = str(results.get("face_contrast_grade", "") or "")
    face_contrast_grade_score = _grade_to_score_like(face_contrast_grade)

    # expression 系
    expression_score = _f("expression_score", 0.5)
    expression_grade = str(results.get("expression_grade", "") or "")
    expression_grade_score = _grade_to_score_like(expression_grade)

    # expression 閾値
    expression_min = _thr("expression_min", 0.5)
    expression_ok = _ok_from(expression_score, expression_grade_score, expression_min)

    # ==============================
    # ★ 明るさ依存補正（A-2 / B-2）
    # ==============================
    noise_score_adj = adjust_noise_by_brightness(noise_score, exposure_score)
    blurriness_score_adj = adjust_blur_by_brightness(blurriness_score, exposure_score)

    # CSV で後から分析できるように結果に残す
    results["noise_score_brightness_adjusted"] = noise_score_adj
    results["blurriness_score_brightness_adjusted"] = blurriness_score_adj

    # 判定に使う値は原則こちら
    noise_score_for_decision = noise_score_adj
    blurriness_score_for_decision = blurriness_score_adj

    # ==============================
    # ノイズ関連の閾値・判定
    # ==============================
    noise_ok_thr = _thr("noise_ok", 0.5)
    noise_good_thr = _thr("noise_good", 0.75)
    face_noise_good_thr = _thr("face_noise_good", 0.75)
    fb_face_noise_min = _thr("full_body_face_noise_min", 0.5)

    noise_ok = _ok_from(noise_score_for_decision, noise_grade_score, noise_ok_thr)
    noise_good = _ok_from(noise_score_for_decision, noise_grade_score, noise_good_thr)
    face_noise_good = _ok_from(face_noise_score, face_noise_grade_score, face_noise_good_thr)

    # full body 用ノイズ判定
    noise_ok_for_full_body = (
        _ok_from(face_noise_score, face_noise_grade_score, fb_face_noise_min)
        or noise_ok
    )

    # ==============================
    # コントラスト関連（full body 用）
    # ==============================
    fb_contrast_min = _thr("full_body_contrast_min", 0.40)
    fb_face_contrast_min = _thr("full_body_face_contrast_min", 0.50)

    contrast_ok_global_fb = _ok_from(
        contrast_score, contrast_grade_score, fb_contrast_min
    )
    contrast_ok_face_fb = _ok_from(
        face_contrast_score, face_contrast_grade_score, fb_face_contrast_min
    )

    contrast_ok_for_full_body = contrast_ok_global_fb or contrast_ok_face_fb

    # ==============================
    # full body 共通条件
    # ==============================
    pose_min_100 = _thr("full_body_pose_min_100", 55.0)
    cut_risk_max = _thr("full_body_cut_risk_max", 0.6)
    blurriness_min_full = _thr("blurriness_min_full_body", 0.45)
    exposure_min_common = _thr("exposure_min_common", 0.5)
    fb_footroom_min = _thr("full_body_footroom_min", 0.0)  # ★ 足元があまりにも切れている full body を弾くためのオプション

    full_body_ok = (
        full_body_detected
        and pose_score_100 >= pose_min_100
        and full_body_cut_risk <= cut_risk_max
        and footroom_ratio >= fb_footroom_min  # ★ 必要ならここを上げて「足ナシ full body」を除外できる
        and noise_ok_for_full_body
        and contrast_ok_for_full_body
        and blurriness_score_for_decision >= blurriness_min_full
        and exposure_score >= exposure_min_common
    )

    # ==============================
    # 顔なしルート（Full body route）
    # ==============================
    if not face_detected:
        if full_body_detected:
            cut_risk = full_body_cut_risk

            if (
                pose_score_100 >= pose_min_100
                and cut_risk <= cut_risk_max
                and footroom_ratio >= fb_footroom_min  # ★ 顔なしでも同様に足元チェック
                and noise_ok_for_full_body
                and contrast_ok_for_full_body
                and blurriness_score_for_decision >= blurriness_min_full
                and exposure_score >= exposure_min_common
            ):
                return True, "full_body"

            return False, "full_body_rejected"

        return False, "no_face"

    # ==============================
    # 顔ありルート
    # ==============================
    # face_quality route 用の閾値
    fq_exposure_min = _thr("face_quality_exposure_min", 0.5)
    fq_face_sharpness_min = _thr("face_quality_face_sharpness_min", 0.75)
    fq_contrast_min = _thr("face_quality_contrast_min", 0.55)
    fq_blur_min = _thr("face_quality_blur_min", 0.55)
    fq_delta_sharpness_min = _thr("face_quality_delta_face_sharpness_min", -10.0)
    fq_yaw_max = _thr("face_quality_yaw_max_abs_deg", 30.0)
    fq_pitch_max = _thr("face_quality_pitch_max_abs_deg", 90.0)  # ★ デフォルト 90° なので既存挙動は変えない

    contrast_ok_for_face_quality = _ok_from(
        face_contrast_score, face_contrast_grade_score, fq_contrast_min
    )

    # Route C: face quality（最優先）
    if (
        exposure_score >= fq_exposure_min
        and face_sharpness_score >= fq_face_sharpness_min
        and face_noise_good
        and contrast_ok_for_face_quality
        and face_blurriness_score >= fq_blur_min
        and delta_face_sharpness >= fq_delta_sharpness_min
        and yaw <= fq_yaw_max
        and pitch <= fq_pitch_max  # ★ 俯き過ぎなどを将来ここで制御可能
        and expression_ok
    ):
        return True, "face_quality"

    # Route D: full body（顔が弱くても拾う）
    if full_body_ok:
        return True, "full_body"

    # Route A: composition
    composition_score_min = _thr("composition_score_min", 0.75)
    framing_score_min = _thr("framing_score_min", 0.5)
    lead_room_min = _thr("lead_room_score_min", 0.10)

    if (
        composition_score >= composition_score_min
        and framing_score >= framing_score_min
        and lead_room_score >= lead_room_min
        and noise_ok
        and blurriness_score_for_decision >= blurriness_min_full
        and exposure_score >= exposure_min_common
    ):
        return True, "composition"

    # Route B: technical
    tech_contrast_min = _thr("technical_contrast_min", 0.60)
    tech_blur_min = _thr("technical_blur_min", 0.60)
    tech_delta_sharpness_min = _thr("technical_delta_face_sharpness_min", -15.0)
    tech_exposure_min = _thr("technical_exposure_min", 1.0)
    tech_face_blur_min = _thr("technical_face_blur_min", 0.50)
    tech_face_sharpness_min = _thr("technical_face_sharpness_min", 0.5)

    contrast_ok_global = _ok_from(
        contrast_score, contrast_grade_score, tech_contrast_min
    )
    contrast_ok_face = _ok_from(
        face_contrast_score, face_contrast_grade_score, tech_contrast_min
    )

    contrast_ok_for_tech = contrast_ok_global or contrast_ok_face

    if (
        noise_good
        and contrast_ok_for_tech
        and blurriness_score_for_decision >= tech_blur_min
        and face_blurriness_score >= tech_face_blur_min
        and face_sharpness_score >= tech_face_sharpness_min
        and delta_face_sharpness >= tech_delta_sharpness_min
        and exposure_score >= tech_exposure_min
    ):
        return True, "technical"

    return False, "rejected"

