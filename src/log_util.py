import logging
from logging.handlers import TimedRotatingFileHandler
import os

class Logger:
    def __init__(self, log_file='app.log', log_dir='logs', level=logging.INFO, log_format=None, handlers=None, logger_name=None):
        if logger_name is None:
            logger_name = __name__
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)

        if log_format is None:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)

        if handlers is None:
            # Ensure the log directory exists
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Construct the full path for the log file
            log_file_path = os.path.join(log_dir, log_file)
            
            file_handler = TimedRotatingFileHandler(
                log_file_path, when='midnight', interval=1, backupCount=7
            )
            file_handler.suffix = "%Y-%m-%d"
            
            console_handler = logging.StreamHandler()
            
            handlers = [file_handler, console_handler]

        for handler in handlers:
            if handler not in self.logger.handlers:
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def critical(self, message):
        self.logger.critical(message)

    @property
    def get_logger(self):
        return self.logger

    def __del__(self):
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

if __name__ == "__main__":
    logger = Logger(log_file='myapp.log', log_dir='my_logs', logger_name='MyAppLogger')
    logger.info("This is an info message.")
    logger.error("This is an error message.")
    logger.debug("This is a debug message.")
