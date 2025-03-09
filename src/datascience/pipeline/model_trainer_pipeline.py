from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.model_trainer import ModelTrainer
from src.datascience import logger


STAGE_NAME="Model Trainer Stage"

class ModelTrainingrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
            config =  ConfigurationManager()
            model_trainer_config=config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train()


if __name__ == "__main__":
    try:
          logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
          obj = ModelTrainingrainingPipeline()
          obj.initiate_model_training()
          logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx============x")
    except Exception as e:
        logger.exception(e)
        raise e