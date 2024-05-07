import logging
from logging.handlers import RotatingFileHandler

def setup_logger():
    # Create a logger object
    logger = logging.getLogger("ShoppingList")
    logger.setLevel(logging.DEBUG)  # Set the minimum level of logging

    # Create a file handler that logs even debug messages
    handler = RotatingFileHandler('./log/shopping_list.log', maxBytes=10000, backupCount=5)
    handler.setLevel(logging.DEBUG)  # You can set different levels if needed

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)

    return logger

if __name__ == '__main__':
    logger = setup_logger()

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
