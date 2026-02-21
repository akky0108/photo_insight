import os
import pandas as pd
import shutil
from photo_insight.batch_framework.base_batch import (
    BaseBatchProcessor,
)  # バッチ処理の基底クラスをインポート


class CsvToTxtProcessor(BaseBatchProcessor):
    def __init__(self, date_str: str, *args, **kwargs):
        """
        初期化メソッド
        :param date_str: 日付パラメータ（yyyy-mm-dd形式）
        """
        super().__init__(*args, **kwargs)
        self.date_str = date_str  # 日付パラメータを受け取る
        self.txt_path = None  # テキストファイルのパスを保存する変数

    def setup(self):
        """
        セットアップフェーズ：ディレクトリの準備
        - 入力と出力ディレクトリのパスを設定
        - 出力ディレクトリが存在しない場合は作成
        """
        self.logger.info("Setup phase started.")
        self.input_dir = os.path.join(self.project_root, "output")  # 入力ディレクトリ
        self.output_dir = os.path.join(
            self.project_root, "processed_txt"
        )  # 出力ディレクトリ

        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Processing files from {self.input_dir}")

    def process(self):
        """
        処理フェーズ：指定日付のCSVを読み込み、2段階の条件を順番に適用してTXTに変換
        """
        self.logger.info("Processing phase started.")
        csv_filename = f"evaluation_ranking_{self.date_str}.csv"
        csv_path = os.path.join(self.input_dir, csv_filename)

        if not os.path.exists(csv_path):
            self.logger.error(f"CSV file {csv_filename} not found.")
            return

        self.txt_filename = f"evaluation_ranking_{self.date_str}.txt"
        self.txt_path = os.path.join(self.output_dir, self.txt_filename)

        try:
            # CSVを読み込む
            df = pd.read_csv(csv_path)

            # ステップ1: face_sharpness_evaluation >= 4のデータを抽出
            df_face_sharpness = df[df["face_sharpness_evaluation"] >= 4]
            self.logger.info(
                f"Step 1: face_sharpness_evaluation >= 4 records: "
                f"{len(df_face_sharpness)}"
            )

            # ステップ2: ①で除外されたデータから overall_evaluation >= 3 を抽出
            df_remaining = df[
                ~df.index.isin(df_face_sharpness.index)
            ]  # ①で抽出されたデータを除外
            df_overall_evaluation = df_remaining[
                df_remaining["overall_evaluation"] >= 3
            ]
            self.logger.info(
                f"Step 2: overall_evaluation >= 3 records: {len(df_overall_evaluation)}"
            )

            # 2つのデータセットを結合
            df_combined = pd.concat(
                [df_face_sharpness, df_overall_evaluation], ignore_index=True
            )
            self.logger.info(
                f"Total records after combining both conditions: {len(df_combined)}"
            )

            # 結合後にデータが空の場合は処理を終了
            if df_combined.empty:
                self.logger.warning(
                    "No records matching the conditions. No output generated."
                )
                return

            # 最終データをTXTとして保存
            df_combined.to_csv(
                self.txt_path, index=False, header=True, sep=",", encoding="utf-8-sig"
            )

            self.logger.info(
                f"Successfully processed {csv_filename} to {self.txt_filename}"
            )

            # ファイル名を出力
            print(self.txt_filename)
        except Exception as e:
            self.logger.error(f"Error processing file: {e}")
            raise

    def cleanup(self):
        """
        クリーンアップフェーズ：ファイルコピーを実行
        - 処理後のテキストファイルを指定のディレクトリにコピー
        """
        self.logger.info("Cleanup phase started.")

        if self.txt_path:
            # WSLパスに変換されたWindowsのパス
            target_dir = "/mnt/d/Users/Akihiro/Creative Cloud Files/rating"
            # コピー先のディレクトリ（WSLパス）
            os.makedirs(
                target_dir, exist_ok=True
            )  # コピー先のディレクトリが存在しない場合は作成
            try:
                shutil.copy(
                    self.txt_path, os.path.join(target_dir, self.txt_filename)
                )  # ファイルをコピー
                self.logger.info(f"Copied {self.txt_filename} to {target_dir}")
            except Exception as e:
                self.logger.error(f"Error copying file: {e}")

        self.logger.info("Cleanup phase completed.")


if __name__ == "__main__":
    import argparse  # コマンドライン引数を処理

    # コマンドライン引数で日付を指定
    parser = argparse.ArgumentParser(description="Process CSV file to TXT format.")
    parser.add_argument("--date", required=True, help="Date in yyyy-mm-dd format")
    args = parser.parse_args()

    # 指定された日付をProcessorに渡す
    processor = CsvToTxtProcessor(date_str=args.date)
    processor.execute()
