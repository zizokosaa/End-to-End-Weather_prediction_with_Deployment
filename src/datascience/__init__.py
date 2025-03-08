import os
import sys
import logging

## Define the format for log messages as asctime: Timestamp, levelname: Logging level (e.g., INFO, WARNING, ERROR)., message: the actual log message
logging_str="[%(asctime)s: %(levelname)s: %(module)s: %(message)s]" # to create the message you have to do logger.info(" lol ")

log_dir="logs"  # the directory that will contain all logging files
log_filepath=os.path.join(log_dir,"logging.log")
os.makedirs(log_dir,exist_ok=True) # creating logging file if it doesn't exist 

logging.basicConfig(

    level=logging.INFO,
    format=logging_str,

    handlers=[
        logging.FileHandler(log_filepath), # write the log messages to the file logs/logging.log
        logging.StreamHandler(sys.stdout) # print log message to the console
    ]
)

logger = logging.getLogger("datasciencelogger")