import os
import pandas as pd
import yaml
import numpy as np
from typing import List, Dict, Union
from photo_insight.batch_framework.base_batch import BaseBatchProcessor


class ThresholdCalculator(BaseBatchProcessor):
    """
    評価スコアに基づいて閾値を計算するクラス。

    Attributes:
        input_folder (str): 評価データを格納したフォルダのパス。
        output_path (str): 閾値YAMLファイルを保存するパス。
        evaluation_data (pd.DataFrame): 読み込んだ評価データを格納するDataFrame。
        thresholds (Dict[str, Dict[str, float]]): 各スコア列に対して計算された閾値。
    """

    def __init__(self, config_path: str, input_folder: str, output_path: str):
        super().__init__(config_path=config_path)
        self.input_folder = input_folder
        self.output_path = output_path
        self.evaluation_data = pd.DataFrame()
        self.thresholds = {}

    def setup(self) -> None:
        """初期化処理を行う。評価データの読み込みを実施する。"""
        super().setup()
        self.logger.info("ThresholdCalculatorのセットアップを開始します。")
        self.evaluation_data = self.load_evaluation_data()
        self.logger.info(f"{len(self.evaluation_data)}件の評価データを読み込みました。")

    def process(self) -> None:
        """閾値の計算を行い、結果を保存する。"""
        self.logger.info("閾値の計算を開始します。")
        self._process_batch(self.evaluation_data)
        self.save_thresholds()

    def cleanup(self) -> None:
        """処理終了後のクリーンアップ処理を行う。"""
        super().cleanup()
        self.logger.info("ThresholdCalculatorのクリーンアップを完了しました。")

    def load_evaluation_data(self) -> pd.DataFrame:
        """
        入力フォルダ内のCSVファイルから評価データを読み込む。

        Returns:
            pd.DataFrame: 結合された評価データ。
        """
        data_frames = []
        for filename in os.listdir(self.input_folder):
            if filename.endswith(".csv") and "evaluation_results_" in filename:
                file_path = os.path.join(self.input_folder, filename)
                self.logger.info(f"評価データを{file_path}から読み込んでいます。")
                try:
                    df = pd.read_csv(file_path)
                    data_frames.append(df)
                except Exception as e:
                    self.logger.error(
                        f"{file_path}の読み込み中にエラーが発生しました: {e}"
                    )

        if data_frames:
            return pd.concat(data_frames, ignore_index=True)
        else:
            self.logger.warning("評価データが見つかりませんでした。")
            return pd.DataFrame()

    def _process_batch(self, batch_data: pd.DataFrame) -> None:
        """
        評価データのバッチを処理し、閾値を計算する。

        Args:
            batch_data (pd.DataFrame): 処理対象の評価データ。
        """
        score_columns = [
            "sharpness_score",
            "blurriness_score",
            "contrast_score",
            "face_sharpness_score",
            "face_contrast_score",
            "face_noise_score",
            "noise_score",
        ]

        for score_column in score_columns:
            scores = self.extract_scores(score_column, batch_data)
            if scores:
                self.logger.debug(
                    f"{score_column}のスコア{len(scores)}件に対して閾値を計算します。"
                )
                self.thresholds[score_column] = self.calculate_thresholds_for_scores(
                    scores, levels=5
                )
            else:
                self.logger.warning(
                    f"{score_column}に有効なデータがありません。ゼロ閾値を設定します。"
                )
                self.thresholds[score_column] = self.get_zero_thresholds()

    def extract_scores(self, key: str, batch_data: pd.DataFrame) -> List[float]:
        """
        指定された列から有効なスコアを抽出する。

        Args:
            key (str): 抽出対象の列名。
            batch_data (pd.DataFrame): 評価データを含むDataFrame。

        Returns:
            List[float]: 抽出されたスコアのリスト。
        """
        if key in batch_data.columns:
            valid_data = batch_data[key].dropna()
            return [self.convert_to_float(value) for value in valid_data]
        return []

    @staticmethod
    def convert_to_float(value: Union[float, np.generic]) -> float:
        """
        値をfloat型に変換する。

        Args:
            value (Union[float, np.generic]): 変換対象の値。

        Returns:
            float: 変換されたfloat型の値。
        """
        if isinstance(value, np.generic):
            return float(value)
        return float(value)

    @staticmethod
    def calculate_thresholds_for_scores(
        scores: List[float], levels: int = 5
    ) -> Dict[str, float]:
        """
        スコアに基づいて閾値を計算する。

        Args:
            scores (List[float]): 閾値計算対象のスコア。
            levels (int): 設定する閾値のレベル数（デフォルト: 5）。

        Returns:
            Dict[str, float]: 計算された閾値。
        """
        quantiles = np.linspace(0, 100, levels + 1)
        return {
            f"level_{i+1}": np.percentile(scores, quantiles[i]) for i in range(levels)
        }

    @staticmethod
    def get_zero_thresholds(levels: int = 5) -> Dict[str, float]:
        """
        ゼロの閾値を生成する。

        Args:
            levels (int): 設定する閾値のレベル数（デフォルト: 5）。

        Returns:
            Dict[str, float]: ゼロで構成された閾値。
        """
        return {f"level_{i+1}": 0.0 for i in range(levels)}

    def save_thresholds(self) -> None:
        """
        計算された閾値をYAMLファイルに保存する。
        """
        self.logger.info(f"閾値を{self.output_path}に保存しています。")
        try:
            thresholds_for_yaml = {
                key: {k: float(v) for k, v in value.items()}
                for key, value in self.thresholds.items()
            }
            with open(self.output_path, mode="w", encoding="utf-8") as yamlfile:
                yaml.dump(
                    thresholds_for_yaml,
                    yamlfile,
                    default_flow_style=False,
                    allow_unicode=True,
                )
            self.logger.info("閾値を正常に保存しました。")
        except Exception as e:
            self.logger.error(f"閾値を{self.output_path}に保存できませんでした: {e}")


# メイン処理
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="評価データに基づいて閾値を計算します。"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config/config.yaml",
        help="設定ファイルのパス。",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="./temp/",
        help="評価データを含むフォルダのパス。",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./config/thresholds.yaml",
        help="閾値YAMLファイルを保存するパス。",
    )
    args = parser.parse_args()

    processor = ThresholdCalculator(
        config_path=args.config_path,
        input_folder=args.input_folder,
        output_path=args.output_path,
    )
    processor.execute()
