class PortraitQualityHeaderGenerator:
    def __init__(self):
        self.face_evaluation_items = [
            "face_detected",
            "faces",
            "face_sharpness_score",
            "face_sharpness_raw",
            "face_sharpness_eval_status",
            "face_contrast_score",
            "face_contrast_raw",
            "face_contrast_eval_status",
            "face_contrast_grade",
            "face_noise_score",
            "face_noise_grade",
            "face_noise_sigma_midtone",
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
            "face_blurriness_status",
            "face_blurriness_grade",
            #★ 追加: 顔ぼやけ具合評価の明るさ補正後スコア
            "face_blurriness_score_brightness_adjusted",
        ]

        self.image_evaluation_items = [
            "sharpness_score",
            "sharpness_raw",           # ←追加（必要なら）
            "sharpness_eval_status",   # ←追加（任意：デバッグ用）
            "blurriness_score",
            "blurriness_raw",          # ←追加（必要なら）
            "blurriness_grade",
            "blurriness_eval_status",  # ←追加（任意：デバッグ用）
            "contrast_score",
            "contrast_raw",
            "contrast_eval_status",
            "contrast_grade",
            "noise_score",
            "noise_grade",
            "noise_sigma_midtone",
            "noise_sigma_used",
            "noise_mask_ratio",
            "noise_eval_status",
            "noise_fallback_reason",
            "local_sharpness_score",
            "local_sharpness_std",
            "local_contrast_score",
            "local_contrast_std",
            "exposure_score",
            "mean_brightness",
            "exposure_grade",
            "exposure_eval_status",
            "exposure_fallback_reason",
            #★ 追加: 画像全体のぼやけ具合評価 （明るさ補正後スコア含む）
            "blurriness_score_brightness_adjusted",
            #★ 追加: 画像全体のノイズ評価（明るさ補正後スコア含む）
            "noise_score_brightness_adjusted",
        ]

        self.composition_evaluation_items = [
            # 従来（顔周りの構図基礎指標）
            "composition_rule_based_score",
            "face_position_score",
            "framing_score",
            "face_direction_score",
            "eye_contact_score",
            "lead_room_score",
            # ★ 新構造: 最低限の意味スコア + 生値
            "body_composition_raw",
            "body_composition_score",
            "composition_raw",
            "composition_score",
            "composition_status",
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

        self.body_evaluation_items = [
            "full_body_detected",
            "pose_score",
            "headroom_ratio",
            "footroom_ratio",
            "side_margin_min_ratio",
            "full_body_cut_risk",
        ]

        self.expression_items = [
            "expression_score",
            "expression_grade",
        ]

        self.group_evaluation_items = ["group_id", "subgroup_id", "shot_type"]

        self.result_meta_items = ["accepted_flag", "accepted_reason"]

    def get_all_headers(self) -> list:
        """
        すべての評価項目を一つのリストにまとめて返す
        """
        return (
            [
                "file_name",  # ファイル名は必ず最初に追加
            ]
            + self.image_evaluation_items
            + self.body_evaluation_items 
            + self.face_evaluation_items
            + self.expression_items
            + self.composition_evaluation_items
            + self.group_evaluation_items
            + self.result_meta_items
        )

    def get_face_headers(self) -> list:
        """
        顔に関連する評価項目のリストを返す
        """
        return ["file_name"] + self.face_evaluation_items

    def get_image_headers(self) -> list:
        """
        画像全体に関連する評価項目のリストを返す
        """
        return ["file_name"] + self.image_evaluation_items

    def get_expression_headers(self) -> list:
        """
        表情に関連する評価項目のリストを返す
        """
        return ["file_name"] + self.expression_items

    def get_composition_headers(self) -> list:
        """
        構図に関連する評価項目のリストを返す
        """
        return ["file_name"] + self.composition_evaluation_items

    def get_group_headers(self) -> list:
        """
        グループに関連する評価項目のリストを返す
        """
        return ["file_name"] + self.group_evaluation_items

    def get_result_meta_headers(self) -> list:
        """
        結果メタデータに関連する評価項目のリストを返す
        """
        return ["file_name"] + self.result_meta_items