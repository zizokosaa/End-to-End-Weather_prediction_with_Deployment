from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.model_trainer import ModelTrainer
from src.datascience.components.model_evaluation import ModelEvaluation
from src.datascience import logger


STAGE_NAME1="Model Trainer Stage"
STAGE_NAME2 = "Model evaluation stage"
class ModelTrainingrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config = model_trainer_config)
        train_x ,train_y,test_x,test_y = model_trainer_config.create_sequence_and_training()
        model_trainer_config.Creating_model(train_x,train_y)
        return test_x,test_y
        # model evaluation
    def initiate_model_evaluating(self,test_x,test_y):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.log_into_mlflow(test_x,test_y)


if __name__ == "__main__":
    try:
          logger.info(f">>>>>> stage {STAGE_NAME1} started <<<<<<")
          obj = ModelTrainingrainingPipeline()
          test_x,test_y=obj.initiate_model_training()
          logger.info(f">>>>>> stage {STAGE_NAME1} completed <<<<<<\n\nx============x")

          logger.info(f">>>>>> stage {STAGE_NAME2} started <<<<<<")
          obj2 = ModelTrainingrainingPipeline()
          obj2.initiate_model_evaluating(test_x,test_y)
          logger.info(f">>>>>> stage {STAGE_NAME2} completed <<<<<<\n\nx============x")
    except Exception as e:
        logger.exception(e)
        raise e