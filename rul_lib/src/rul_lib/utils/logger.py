import atexit
import logging
import logging.config
import json
from pathlib import Path
from importlib.resources import files


class LoggerSetup:
  ''' Class to setup logging configuration from a json file.'''

  def __init__(self, logger_name: str = 'logger', config_path: str | Path = None):
    if config_path is None:
      config_path = files('rul_lib.config').joinpath('logger_config.json')
    self.config_path = Path(config_path)
    self.logger = None
    self.logger_name = logger_name

  def setup_logging(self):
    with open(self.config_path) as f_in:
      config = json.load(f_in)
    logging.config.dictConfig(config)
    self.logger = logging.getLogger(self.logger_name)
    queue_handler = logging.getHandlerByName('queue_handler')
    if queue_handler is not None:
      queue_handler.listener.start()
      atexit.register(queue_handler.listener.stop)

  def get_logger(self):
    if self.logger is None:
      self.setup_logging()
    return self.logger
