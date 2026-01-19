from typing import Dict, Any, Optional, List, Tuple

import numpy as np

from evaluators.rule_based_composition_evaluator import RuleBasedCompositionEvaluator
from evaluators.fullbody_composition_evaluator import FullBodyCompositionEvaluator
from evaluators.base_composition_evaluator import BaseCompositionEvaluator
from evaluators.rule_of_thirds_evaluator import RuleOfThirdsEvaluator  # ★ 追加
from utils.app_logger import Logger


class CompositeCompositionEvaluator(BaseCompositionEvaluator):
    """
    顔構図評価と全身構図評価、およびルール・オブ・サード構図評価を統合するファサードクラス。

    - RuleBasedCompositionEvaluator: 顔中心の構図評価
        - face_composition_raw / face_composition_score を返す
    - FullBodyCompositionEvaluator: 全身の構図評価
        - body_composition_raw / body_composition_score を返す
    - RuleOfThirdsEvaluator: 3分割法に基づく構図評価
        - rule_of_thirds_raw / rule_of_thirds_score を返す

    本クラスではそれらを統合して:
        - composition_raw  : 顔＋全身＋ルール・オブ・サードの連続スコア (0〜1)
        - composition_score: 離散スコア (0 / 0.25 / 0.5 / 0.75 / 1.0)
        - composition_status: face_only / body_only / face_and_body / rule_of_thirds_only / face_and_body_and_rule_of_thirds ...
        を返す。
    """

    def __init__(
        self,
        logger: Optional[Logger] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            logger: 任意のロガーインスタンス。指定がない場合は AppLogger を使用。
            config: 閾値・重み・離散化ルールなどの設定。
                    例:
                    composition:
                      face_weight: 1.0
                      body_weight: 1.0
                      rule_of_thirds_weight: 0.5
                      discretize_thresholds:
                        excellent: 0.85
                        good: 0.70
                        fair: 0.55
                        poor: 0.40
        """
        self.logger = logger or Logger(logger_name="CompositeCompositionEvaluator")

        self.config = config or {}
        comp_conf = self.config.get("composition", {})

        # 顔・全身・ルールオブサード スコアの統合重み
        self.face_weight: float = float(comp_conf.get("face_weight", 1.0))
        self.body_weight: float = float(comp_conf.get("body_weight", 1.0))
        self.rule_of_thirds_weight: float = float(
            comp_conf.get("rule_of_thirds_weight", 0.5)
        )

        # 離散化用しきい値（0〜1）
        disc_conf = comp_conf.get("discretize_thresholds", {})
        self.threshold_excellent: float = float(disc_conf.get("excellent", 0.85))
        self.threshold_good: float = float(disc_conf.get("good", 0.70))
        self.threshold_fair: float = float(disc_conf.get("fair", 0.55))
        self.threshold_poor: float = float(disc_conf.get("poor", 0.40))

        self.face_evaluator = RuleBasedCompositionEvaluator(
            logger=self.logger,
            config=self.config,
        )
        self.body_evaluator = FullBodyCompositionEvaluator(
            logger=self.logger,
            config=self.config,
        )

    def evaluate(
        self,
        image: np.ndarray,
        face_boxes: List[Dict[str, Any]],
        body_keypoints: List[Optional[List[float]]],
    ) -> Dict[str, Any]:
        """
        顔と全身の構図評価、およびルール・オブ・サード評価を統合して実行。

        Args:
            image: 評価対象画像 (H, W, C)
            face_boxes: 顔検出結果リスト
            body_keypoints: 全身キーポイントリスト

        Returns:
            dict: 統合された評価結果。
                  - face_composition_raw / face_composition_score
                  - body_composition_raw / body_composition_score
                  - rule_of_thirds_raw / rule_of_thirds_score
                  - main_subject_center_source / main_subject_center_x / main_subject_center_y
                  - contrib_comp_rule_of_thirds_score
                  - composition_raw / composition_score
                  - composition_status
                  加えて、各 Evaluator が返す他のキーもそのまま含まれる。
        """
        self.logger.info("Starting composite composition evaluation.")

        face_results: Dict[str, Any] = self.face_evaluator.evaluate(image, face_boxes) or {}
        body_results: Dict[str, Any] = self.body_evaluator.evaluate(image, body_keypoints) or {}

        # 個別スコアを取得（なければ None）
        face_raw = face_results.get("face_composition_raw")
        body_raw = body_results.get("body_composition_raw")

        # ルール・オブ・サードの連続スコア（0〜1） + 中心情報を計算
        (
            rule_of_thirds_raw,
            main_subject_center,
            main_subject_center_source,
        ) = self._compute_rule_of_thirds_raw(
            image=image,
            face_results=face_results,
            body_results=body_results,
            face_boxes=face_boxes,
            body_keypoints=body_keypoints,
        )

        # 統合生値を計算（顔＋全身＋ルール・オブ・サードを重み付きでまとめる）
        composition_raw, status = self._merge_raw_scores(
            face_raw=face_raw,
            body_raw=body_raw,
            rule_of_thirds_raw=rule_of_thirds_raw,
        )

        # 離散スコア化
        if composition_raw is None:
            # 「分からないのでニュートラル」として 0.5 を付与
            composition_score = 0.5
            if status == "not_computed":
                status = "not_computed_with_default"
        else:
            composition_score = self._to_discrete_score(composition_raw)

        # 顔・全身の離散スコアも、もし raw のみならここで付与しておく
        face_score = face_results.get("face_composition_score")
        if face_score is None and face_raw is not None:
            face_score = self._to_discrete_score(face_raw)

        body_score = body_results.get("body_composition_score")
        if body_score is None and body_raw is not None:
            body_score = self._to_discrete_score(body_raw)

        # ルール・オブ・サードの離散スコア
        if rule_of_thirds_raw is not None:
            rule_of_thirds_score = self._to_discrete_score(rule_of_thirds_raw)
        else:
            rule_of_thirds_score = None

        # contrib: RoT（ランキング・CSV用）
        contrib_rule_of_thirds = None
        if rule_of_thirds_raw is not None:
            contrib_rule_of_thirds = self.rule_of_thirds_weight * float(rule_of_thirds_raw)

        # ★ 追加: body_composition の寄与（全体構図スコアへの寄与）
        contrib_body_composition = None
        if body_raw is not None and self.body_weight > 0.0:
            contrib_body_composition = self.body_weight * float(body_raw)

        # 中心座標の展開（なければ None）
        if main_subject_center is not None:
            try:
                center_x, center_y = main_subject_center
                main_subject_center_x = float(center_x)
                main_subject_center_y = float(center_y)
            except Exception:
                main_subject_center_x = None
                main_subject_center_y = None
        else:
            main_subject_center_x = None
            main_subject_center_y = None

        combined_results: Dict[str, Any] = {
            **face_results,
            **body_results,  # 注意: group_id / subgroup_id は全身側が上書きする
            "face_composition_raw": face_raw,
            "face_composition_score": face_score,
            "body_composition_raw": body_raw,
            "body_composition_score": body_score,
            "rule_of_thirds_raw": rule_of_thirds_raw,
            "rule_of_thirds_score": rule_of_thirds_score,
            "contrib_comp_rule_of_thirds_score": contrib_rule_of_thirds,
            # ★ ここで body の contrib もセット
            "contrib_comp_body_composition_score": contrib_body_composition,
            "main_subject_center_source": main_subject_center_source,
            "main_subject_center_x": main_subject_center_x,
            "main_subject_center_y": main_subject_center_y,
            "composition_raw": composition_raw,
            "composition_score": composition_score,
            "composition_status": status,
        }

        self.logger.info(
            f"Composite composition evaluation completed. "
            f"raw={composition_raw}, score={composition_score}, status={status}, "
            f"rule_of_thirds_raw={rule_of_thirds_raw}, "
            f"center={main_subject_center}, source={main_subject_center_source}"
        )
        return combined_results

    # -----------------------
    # 内部ヘルパー
    # -----------------------
    def _merge_raw_scores(
        self,
        face_raw: Optional[float],
        body_raw: Optional[float],
        rule_of_thirds_raw: Optional[float] = None,
    ) -> Tuple[Optional[float], str]:
        """
        顔構図・全身構図・ルールオブサード構図の生値を統合して 0〜1 の連続スコアを作る。

        戻り値:
            (composition_raw, status)
            status: "not_computed" / "face_only" / "body_only" / "face_and_body"
                    / "rule_of_thirds_only" / "face_and_body_and_rule_of_thirds" など
        """

        # まずは元の「顔＋全身」だけのパターンを優先的に扱う
        # ※ rule_of_thirds_raw が None のときは完全互換の挙動
        if rule_of_thirds_raw is None or self.rule_of_thirds_weight <= 0.0:
            # どちらも None → 計算不能
            if face_raw is None and body_raw is None:
                return None, "not_computed"

            if face_raw is not None and (body_raw is None or body_raw <= 0.0):
                return float(face_raw), "face_only"

            if body_raw is not None and (face_raw is None or face_raw <= 0.0):
                return float(body_raw), "body_only"

            # 両方ある場合 → 重み付き平均
            w_face = self.face_weight
            w_body = self.body_weight
            if w_face <= 0 and w_body <= 0:
                raw = (float(face_raw) + float(body_raw)) / 2.0
                return raw, "both_zero_weight"

            raw = (float(face_raw) * w_face + float(body_raw) * w_body) / (w_face + w_body)
            return raw, "face_and_body"

        # ここから下は「ルールオブサードも活かす」パス
        has_face = face_raw is not None and face_raw > 0.0
        has_body = body_raw is not None and body_raw > 0.0
        has_rot = rule_of_thirds_raw is not None and rule_of_thirds_raw > 0.0

        if not has_face and not has_body and not has_rot:
            return None, "not_computed"

        total = 0.0
        wsum = 0.0
        parts: List[str] = []

        if has_face and self.face_weight > 0.0:
            total += float(face_raw) * self.face_weight
            wsum += self.face_weight
            parts.append("face")

        if has_body and self.body_weight > 0.0:
            total += float(body_raw) * self.body_weight
            wsum += self.body_weight
            parts.append("body")

        if has_rot and self.rule_of_thirds_weight > 0.0:
            total += float(rule_of_thirds_raw) * self.rule_of_thirds_weight
            wsum += self.rule_of_thirds_weight
            parts.append("rule_of_thirds")

        if wsum <= 0.0:
            # 極端ケース: 全ての weight <= 0 の場合
            # 利用可能な raw の単純平均にフォールバック
            vals = [
                float(v)
                for v in [face_raw, body_raw, rule_of_thirds_raw]
                if v is not None
            ]
            if not vals:
                return None, "not_computed"
            return float(sum(vals) / len(vals)), "fallback_average"

        raw = total / wsum

        # status 表現
        if parts == ["face"]:
            status = "face_only"
        elif parts == ["body"]:
            status = "body_only"
        elif parts == ["rule_of_thirds"]:
            status = "rule_of_thirds_only"
        else:
            status = "_and_".join(parts)  # 例: "face_and_body_and_rule_of_thirds"

        return raw, status

    def _to_discrete_score(self, value: float) -> float:
        """
        0〜1 の連続スコアを 0 / 0.25 / 0.5 / 0.75 / 1.0 に離散化する。

        閾値は config から取得。指定がなければデフォルト値を使用。
        """
        if value is None:
            # 呼び出し側で None チェックする設計だが、一応保険としてニュートラルを返す
            return 0.5

        v = float(value)

        if v >= self.threshold_excellent:
            return 1.0
        if v >= self.threshold_good:
            return 0.75
        if v >= self.threshold_fair:
            return 0.5
        if v >= self.threshold_poor:
            return 0.25
        return 0.0

    # -----------------------
    # ルール・オブ・サード用ヘルパー
    # -----------------------
    def _compute_rule_of_thirds_raw(
        self,
        image: np.ndarray,
        face_results: Dict[str, Any],
        body_results: Dict[str, Any],
        face_boxes: List[Dict[str, Any]],
        body_keypoints: List[Optional[List[float]]],
    ) -> Tuple[Optional[float], Optional[Tuple[float, float]], Optional[str]]:
        """
        既存の顔・全身情報から「メイン被写体の中心」を推定し、
        RuleOfThirdsEvaluator で 0〜1 のスコアを算出する。

        戻り値:
            (rule_of_thirds_raw, main_subject_center, main_subject_center_source)

        取得順の優先度:
            1. face_results["main_subject_center"]
            2. face_results["face_center"]
            3. body_results["full_body_center"] など
            4. face_boxes の先頭 bbox
            5. body_keypoints の重心
            6. 何も取れなければ (None, None, None) を返す（→ RoT は構図に混ぜない）
        """
        evaluator: Optional[RuleOfThirdsEvaluator] = None

        def _eval_with_center(center: Tuple[float, float], source: str):
            nonlocal evaluator
            if not self._is_valid_point(center, image):
                return None, None, None
            if evaluator is None:
                evaluator = RuleOfThirdsEvaluator(image)
            score = evaluator.evaluate_from_point(center)
            return score, center, source

        # 1. 既に main_subject_center があればそれを優先
        if "main_subject_center" in face_results:
            center = face_results["main_subject_center"]
            score, c, s = _eval_with_center(center, "main_subject_center")
            if score is not None:
                return score, c, s

        # 2. 顔中心
        if "face_center" in face_results:
            center = face_results["face_center"]
            score, c, s = _eval_with_center(center, "face_center")
            if score is not None:
                return score, c, s

        # 3. 全身中心（もし body_evaluator 側が出していれば）
        if "full_body_center" in body_results:
            center = body_results["full_body_center"]
            score, c, s = _eval_with_center(center, "full_body_center")
            if score is not None:
                return score, c, s

        # 4. face_boxes から推定
        if face_boxes:
            cx, cy = self._center_from_face_box(face_boxes[0], image)
            if cx is not None and cy is not None:
                score, c, s = _eval_with_center((cx, cy), "face_box")
                if score is not None:
                    return score, c, s

        # 5. body_keypoints の重心から推定
        if body_keypoints:
            cx, cy = self._center_from_body_keypoints(body_keypoints[0], image)
            if cx is not None and cy is not None:
                score, c, s = _eval_with_center((cx, cy), "body_keypoints")
                if score is not None:
                    return score, c, s

        # 6. どうしても決まらない場合は None → RoT を構図に混ぜない
        self.logger.debug(
            "Rule-of-thirds center could not be determined; skipping RoT contribution."
        )
        return None, None, None

    @staticmethod
    def _is_valid_point(point: Any, image: np.ndarray) -> bool:
        """point が (x, y) 形式で画像内に収まっているかの簡易チェック。"""
        try:
            x, y = point
            h, w = image.shape[:2]
            return 0 <= float(x) <= w and 0 <= float(y) <= h
        except Exception:
            return False

    @staticmethod
    def _center_from_face_box(
        box: Dict[str, Any],
        image: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        顔検出 bbox から中心を推定する。

        想定するフォーマットの例:
            - {"x": x, "y": y, "w": w, "h": h}
            - {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2}
            - {"bbox": [x_min, y_min, x_max, y_max]}
            - {"box": [x1, y1, x2, y2]}  # ★ insightface 形式
        """
        h, w = image.shape[:2]

        # パターン1: x, y, w, h
        if all(k in box for k in ("x", "y", "w", "h")):
            x = float(box["x"])
            y = float(box["y"])
            bw = float(box["w"])
            bh = float(box["h"])
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            return (
                float(np.clip(cx, 0.0, float(w))),
                float(np.clip(cy, 0.0, float(h))),
            )

        # パターン2: x_min, y_min, x_max, y_max
        if all(k in box for k in ("x_min", "y_min", "x_max", "y_max")):
            x_min = float(box["x_min"])
            y_min = float(box["y_min"])
            x_max = float(box["x_max"])
            y_max = float(box["y_max"])
            if x_max <= x_min or y_max <= y_min:
                return None, None
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            return (
                float(np.clip(cx, 0.0, float(w))),
                float(np.clip(cy, 0.0, float(h))),
            )

        # パターン3: "bbox": [x_min, y_min, x_max, y_max]
        if "bbox" in box and isinstance(box["bbox"], (list, tuple)) and len(box["bbox"]) == 4:
            x_min, y_min, x_max, y_max = map(float, box["bbox"])
            if x_max <= x_min or y_max <= y_min:
                return None, None
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            return (
                float(np.clip(cx, 0.0, float(w))),
                float(np.clip(cy, 0.0, float(h))),
            )

        # パターン4: "box": [x_min, y_min, x_max, y_max]（insightface 形式）
        if "box" in box and isinstance(box["box"], (list, tuple)) and len(box["box"]) == 4:
            x_min, y_min, x_max, y_max = map(float, box["box"])
            if x_max <= x_min or y_max <= y_min:
                return None, None
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            return (
                float(np.clip(cx, 0.0, float(w))),
                float(np.clip(cy, 0.0, float(h))),
            )

        return None, None

    @staticmethod
    def _center_from_body_keypoints(
        keypoints: Optional[List[float]],
        image: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        全身キーポイントから重心を推定する。

        想定するフォーマットの例:
            [x0, y0, x1, y1, x2, y2, ...]
        """
        if not keypoints:
            return None, None

        try:
            arr = np.asarray(keypoints, dtype=float)
            if arr.ndim != 1 or arr.size < 2:
                return None, None

            xs = arr[0::2]
            ys = arr[1::2]
            if xs.size == 0 or ys.size == 0:
                return None, None

            cx = float(xs.mean())
            cy = float(ys.mean())

            h, w = image.shape[:2]
            cx = float(np.clip(cx, 0.0, float(w)))
            cy = float(np.clip(cy, 0.0, float(h)))
            return cx, cy
        except Exception:
            return None, None
