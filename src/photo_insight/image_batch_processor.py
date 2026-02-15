import csv
import cv2
import datetime
import json
import numpy as np
import os
from typing import List, Dict, Optional
from photo_insight.batch_framework.base_batch import BaseBatchProcessor
from photo_insight.utils.app_logger import Logger
from image_loader import ImageLoader  # 画像読み込みクラスをインポート

DEFAULT_IMAGE_DIR = "/mnt/l/picture/2025"
DEFAULT_OUTPUT_DIR = "temp"


class ImageBatchProcessor(BaseBatchProcessor):
    """画像データを処理するためのバッチプロセッサ"""

    def __init__(
        self, target_date: Optional[str] = None, logger: Logger = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.date = self._parse_date(target_date)

        # ディレクトリやファイルパスを設定
        self._set_directories_and_files(self.date)

        self.logger = logger if logger else Logger(logger_name="ImageBatchProcessor")
        self.loader = ImageLoader(self.logger)  # ImageLoaderインスタンスを生成

    def _parse_date(self, date_str: Optional[str]) -> str:
        """ターゲット日付を解析。フォーマットが不正の場合は現在日付を使用"""
        if date_str:
            try:
                datetime.datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
            except ValueError:
                self.logger.error(
                    "Invalid date format. Use 'YYYY-MM-DD'. Using current date as fallback."
                )
        return datetime.datetime.now().strftime("%Y-%m-%d")

    def _set_directories_and_files(self, date: str):
        """処理するディレクトリとCSVファイルのパスを設定"""
        self.image_dir = os.path.join(
            self.config.get("base_directory_root", DEFAULT_IMAGE_DIR), date
        )
        self.output_directory = self.config.get("output_directory", DEFAULT_OUTPUT_DIR)
        self.result_csv_file = os.path.join(
            self.output_directory, "evaluation_results.csv"
        )

    def setup(self) -> None:
        """評価結果を読み込むためのセットアップフェーズ"""
        super().setup()  # 親クラスの共通セットアップ処理
        self._load_config()

        # 日付に基づいたファイルパスを設定
        file_path = os.path.join(
            self.paths.get("evaluation_data_dir", "./temp"),
            f"evaluation_results_{self.date}.csv",
        )
        if not os.path.exists(file_path):
            self.logger.error(f"Evaluation data file not found: {file_path}")
            raise FileNotFoundError(f"Evaluation data file not found: {file_path}")

        self.data = self._load_evaluation_data(file_path)  # 評価データをロード

    def _load_config(self) -> None:
        """設定ファイルを読み込む"""
        self.paths = self.config.get("paths", {})

    def _load_evaluation_data(self, file_path: str) -> List[Dict[str, str]]:
        """評価データをCSVファイルからロードする"""
        self.logger.info(f"Loading evaluation data from {file_path}.")
        data = []
        try:
            with open(file_path, mode="r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # `face_evaluation` をパース
                    row["face_evaluation"] = self._parse_face_evaluation(
                        row.get("face_evaluation", "")
                    )
                    data.append(row)
        except Exception as e:
            self.logger.error(f"Error reading evaluation data: {e}", exc_info=True)
            raise
        self.logger.info("Evaluation data loaded successfully.")
        return data

    def _parse_face_evaluation(self, face_evaluation: str) -> Optional[Dict]:
        """`face_evaluation` を辞書形式に変換"""
        if isinstance(face_evaluation, dict):
            return face_evaluation
        if isinstance(face_evaluation, str):
            try:
                face_evaluation = eval(face_evaluation)
                self.logger.debug(f"Decoded face_evaluation: {face_evaluation}")
                return face_evaluation
            except json.JSONDecodeError as e:
                self.logger.warning(
                    f"Failed to decode face_evaluation JSON: {face_evaluation}, error: {e}"
                )
                return None
        self.logger.warning(f"Invalid face_evaluation format: {face_evaluation}")
        return None

    def _process_batch(self, batch: List[Dict[str, str]]) -> None:
        """バッチ単位で評価データを処理"""
        for entry in batch:
            image_file = entry.get("file_name")
            image_path = os.path.join(self.image_dir, image_file)

            try:
                # 画像をロード
                image = self.loader.load_image(image_path)
                face_bbox = self._extract_face_bbox(entry)

                # 顔データがなければスキップ
                if not face_bbox:
                    self.logger.warning(
                        f"No valid face data for {image_file}. Skipping."
                    )
                    continue

                # 各評価を実施
                thirds_score = self.evaluate_rule_of_thirds(image, face_bbox)
                margin_score = self.evaluate_margin(image)

                # 結果をログに記録
                self.logger.info(
                    f"Processed {image_file}: Rule of Thirds={thirds_score:.2f}, Margin={margin_score:.2f}"
                )

            except Exception as e:
                self.logger.error(f"Error processing {image_file}: {e}", exc_info=True)

    def _extract_face_bbox(self, entry: Dict[str, str]) -> Optional[tuple]:
        """顔領域のバウンディングボックスを抽出"""
        face_eval = entry.get("face_evaluation", {})
        if isinstance(face_eval, dict):
            faces = face_eval.get("faces", [])
            if faces and isinstance(faces[0], dict):
                self.logger.debug(f"Valid face data found: {faces[0].get('box')}")
                return faces[0].get("box")
            else:
                self.logger.warning(f"No faces data in entry: {entry}")
        else:
            self.logger.warning(f"Invalid face_evaluation format: {face_eval}")
        return None

    def evaluate_rule_of_thirds(
        self, image: np.ndarray, face_bbox: tuple = None
    ) -> float:
        """三分割法の評価"""
        height, width = image.shape[:2]
        thirds_points = [
            (width // 3, height // 3),
            (2 * width // 3, height // 3),
            (width // 3, 2 * height // 3),
            (2 * width // 3, 2 * height // 3),
        ]

        # 顔の中心点を計算
        if face_bbox:
            x, y, w, h = face_bbox
            center_x, center_y = x + w // 2, y + h // 2
        else:
            self.logger.warning("No face detected, returning score 0.")
            return 0.0

        # 中心点が全ての三分割交点に近いほどスコアが高い
        min_distance = min(
            np.sqrt((center_x - x) ** 2 + (center_y - y) ** 2) for x, y in thirds_points
        )

        normalized_score = 1 - (
            min_distance / np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
        )
        return max(0.0, normalized_score)

    def evaluate_margin(self, image: np.ndarray) -> float:
        """余白の評価 (画像内のコンテンツの集中度)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return 0.0

        # 最外周の輪郭領域を取得
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        image_area = image.shape[0] * image.shape[1]
        content_area = w * h
        margin_score = 1 - (content_area / image_area)
        return max(0.0, margin_score)


def main(target_date: Optional[str] = None):
    """バッチ処理を実行するメイン関数"""
    # ロガーのインスタンスを生成
    logger = Logger(logger_name="ImageBatchProcessor")

    # ImageBatchProcessorのインスタンスを作成
    processor = ImageBatchProcessor(target_date=target_date, logger=logger)

    try:
        # セットアップを実行
        processor.setup()

        # 評価データをバッチ処理で処理
        processor._process_batch(processor.data)
    except Exception as e:
        logger.error(f"Error during batch processing: {e}", exc_info=True)


if __name__ == "__main__":
    # 実行時に指定した日付を引数として渡すことが可能
    target_date = "2025-03-02"  # 例: 日付を指定
    main(target_date)
