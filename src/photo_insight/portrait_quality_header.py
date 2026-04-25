class PortraitQualityHeaderGenerator:
    def __init__(self):
        self.face_evaluation_items = [
            "face_detected",
            "portrait_category",  # ← ★追加
            "faces",
            "face_sharpness_score",
            "face_sharpness_raw",
            "face_sharpness_eval_status",
            "face_contrast_score",
            "face_contrast_raw",
            "face_contrast_eval_status",
            "face_contrast_grade",
            "face_noise_score",
            "face_noise_raw",
            "face_noise_grade",
            "face_noise_sigma_midtone",
            "face_noise_sigma_used",
            "face_noise_mask_ratio",
            "face_local_sharpness_score",
            "face_local_sharpness_std",
            "face_local_contrast_score",
            "face_local_contrast_std",
            "face_exposure_score",
            "face_mean_brightness",
            "face_exposure_grade",
            "face_exposure_eval_status",
            "face_exposure_fallback_reason",
            "yaw",
            "pitch",
            "roll",
            "gaze",
            "delta_face_sharpness",
            "delta_face_contrast",
            # ★ 追加: 顔構図評価（最低限）
            "face_composition_raw",
            "face_composition_score",
            "face_composition_status",
            # ★ 追加: 顔ぼやけ具合評価
            "face_blurriness_raw",
            "face_blurriness_score",
            "face_blurriness_grade",
            "face_blurriness_eval_status",
            "face_blurriness_fallback_reason",
            # ★ 追加: 顔ぼやけ具合評価の明るさ補正後スコア
            "face_blurriness_score_brightness_adjusted",
        ]

        self.image_evaluation_items = [
            "sharpness_score",
            "sharpness_raw",
            "sharpness_eval_status",
            # ----------------------------
            # Blurriness
            # ----------------------------
            "blurriness_score",
            "blurriness_raw",
            "blurriness_grade",
            "blurriness_eval_status",
            "blurriness_fallback_reason",
            "contrast_score",
            "contrast_raw",
            "contrast_eval_status",
            "contrast_grade",
            "noise_score",
            "noise_raw",
            "noise_grade",
            "noise_sigma_midtone",
            "noise_sigma_used",
            "noise_mask_ratio",
            "noise_eval_status",
            "noise_fallback_reason",
            # ----------------------------
            # Local Sharpness
            # ----------------------------
            "local_sharpness_score",
            "local_sharpness_raw",
            "local_sharpness_std",
            "local_sharpness_eval_status",
            "local_sharpness_fallback_reason",
            # ----------------------------
            # Local Contrast
            # ----------------------------
            "local_contrast_score",
            "local_contrast_raw",
            "local_contrast_std",
            "local_contrast_eval_status",
            "local_contrast_fallback_reason",
            # ----------------------------
            # Exposure
            # ----------------------------
            "exposure_score",
            "mean_brightness",
            "exposure_grade",
            "exposure_eval_status",
            "exposure_fallback_reason",
            # ----------------------------
            # Brightness adjusted
            # ----------------------------
            "blurriness_score_brightness_adjusted",
            "noise_score_brightness_adjusted",
        ]

        self.composition_evaluation_items = [
            "composition_rule_based_score",
            "face_position_score",
            "framing_score",
            "face_direction_score",
            "eye_contact_score",
            "lead_room_score",
            "body_composition_raw",
            "body_composition_score",
            "composition_raw",
            "composition_score",
            "composition_status",
            "main_subject_center_source",
            "main_subject_center_x",
            "main_subject_center_y",
            "rule_of_thirds_raw",
            "rule_of_thirds_score",
            "contrib_comp_composition_rule_based_score",
            "contrib_comp_face_position_score",
            "contrib_comp_framing_score",
            "contrib_comp_lead_room_score",
            "contrib_comp_body_composition_score",
            "contrib_comp_rule_of_thirds_score",
        ]

        self.body_evaluation_items = [
            "full_body_detected",
            "pose_score",
            "headroom_ratio",
            "footroom_ratio",
            "side_margin_min_ratio",
            "full_body_cut_risk",
            "body_height_ratio",
            "body_center_y_ratio",
        ]

        self.expression_items = [
            "expression_score",
            "expression_grade",
        ]

        self.group_evaluation_items = ["group_id", "subgroup_id", "shot_type"]

        self.result_meta_items = ["accepted_flag", "accepted_reason"]

    def get_all_headers(self) -> list:
        return (
            ["file_name"]
            + self.image_evaluation_items
            + self.body_evaluation_items
            + self.face_evaluation_items
            + self.expression_items
            + self.composition_evaluation_items
            + self.group_evaluation_items
            + self.result_meta_items
        )
