import logging
import os
from model import Base1

def select_model(model_name, config):
    models = {
        "Base1": Base1,
    }

    model = models.get(model_name)

    if model:
        return model(config)
    else:
        raise ValueError(f"Model '{model_name}' not found.")


def create_logger(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(
        filename=os.path.join(args.logger_path + ".log"))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger
