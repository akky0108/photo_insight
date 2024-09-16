import logging
import logging.config
import os
import yaml
import threading
import atexit

class Logger:
    _instance = None  # シングルトンインスタンス
    _lock = threading.Lock()  # スレッドセーフのためのロック

    def __new__(cls, *args, **kwargs):
        """ シングルトンインスタンスを作成、または既存インスタンスを返す """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_file=None, logger_name='MyAppLogger'):
        if getattr(self, '_initialized', False):
            return  # 既に初期化済みの場合は何もしない

        # デフォルトのconfig_fileパスは実行クラス（スクリプト）の場所とする
        if config_file is None:
            config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging_config.yaml')

        # YAMLファイルからロギング設定を読み込む
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    logging.config.dictConfig(config)
                print(f"Logging configured from {config_file}.")
            except (yaml.YAMLError, FileNotFoundError) as e:
                print(f"Error loading logging config: {e}. Using default logging settings.")
                logging.basicConfig(level=logging.DEBUG)
        else:
            print(f"Config file {config_file} not found. Using default logging settings.")
            logging.basicConfig(level=logging.DEBUG)

        # ロガーを設定
        self.logger = logging.getLogger(logger_name)

        # プログラム終了時にクリーンアップを登録
        atexit.register(self.cleanup)

        # 初期化フラグを立てる
        self._initialized = True

    def info(self, message):
        """INFOレベルのログメッセージを出力"""
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(message)

    def error(self, message):
        """ERRORレベルのログメッセージを出力"""
        if self.logger.isEnabledFor(logging.ERROR):
            self.logger.error(message)

    def debug(self, message):
        """DEBUGレベルのログメッセージを出力"""
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(message)

    def warning(self, message):
        """WARNINGレベルのログメッセージを出力"""
        if self.logger.isEnabledFor(logging.WARNING):
            self.logger.warning(message)

    def critical(self, message):
        """CRITICALレベルのログメッセージを出力"""
        if self.logger.isEnabledFor(logging.CRITICAL):
            self.logger.critical(message)

    def log_metric(self, metric_name: str, score: float) -> None:
        """メトリック名とスコアをINFOレベルでログに出力"""
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"{metric_name.capitalize()} score: {score}")

    def change_logger_name(self, new_name):
        """ロガーネームを変更"""
        # 現在のロガーを削除
        logging.Logger.manager.loggerDict.pop(self.logger.name, None)

        # 新しいロガーネームでロガーを再取得
        self.logger = logging.getLogger(new_name)

    def get_logger(self):
        """現在のロガーを返す"""
        return self.logger

    def cleanup(self):
        """プログラム終了時にハンドラをクリーンアップ"""
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

    def isEnabledFor(self, level):
        """特定のログレベルが有効かどうかを確認"""
        return self.logger.isEnabledFor(level)

# 使用例
if __name__ == "__main__":
    logger = Logger(logger_name='MyAppLogger')
    logger.info("This is an info message with the original logger name.")
    
    logger.change_logger_name('NewLoggerName')
    logger.info("This is an info message with the new logger name.")
    
    logger.log_metric('sharpness', 0.85)
    logger.log_metric('contrast', 0.75)
