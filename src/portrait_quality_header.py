class PortraitQualityHeaderGenerator:
    def __init__(self):
        self.face_evaluation_items = [
            "face_detected", 
            "face_sharpness_score", 
            "face_contrast_score", 
            "face_noise_score", 
            "face_local_sharpness_score", 
            "face_local_sharpness_std", 
            "face_local_contrast_score", 
            "face_local_contrast_std", 
            "face_exposure_score", 
            "face_mean_brightness", 
            "yaw", 
            "pitch", 
            "roll", 
            "gaze"
        ]
        
        self.image_evaluation_items = [
            "sharpness_score", 
            "blurriness_score", 
            "contrast_score", 
            "noise_score", 
            "local_sharpness_score", 
            "local_sharpness_std", 
            "local_contrast_score", 
            "local_contrast_std", 
            "exposure_score",
            "mean_brightness"
        ]
        
        self.composition_evaluation_items = [
            "composition_rule_based_score", 
            "face_position_score", 
            "framing_score", 
            "face_direction_score",
            "eye_contact_score"
        ]
        
        self.group_evaluation_items = [
            "group_id", 
            "subgroup_id"
        ]

    def get_all_headers(self) -> list:
        """
        すべての評価項目を一つのリストにまとめて返す
        """
        return [
            "file_name",  # ファイル名は必ず最初に追加
        ] + self.image_evaluation_items + self.face_evaluation_items + self.composition_evaluation_items + self.group_evaluation_items

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
