import os
import sys
import shutil

from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of model evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   Copies the best trained model to the local 'saved_models' directory,
                        replacing the previous production model.

        Output      :   Returns ModelPusherArtifact with the destination path
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            logging.info("=" * 80)

            src_path = self.model_evaluation_artifact.trained_model_path
            dest_path = self.model_pusher_config.local_model_path   # e.g. "saved_models/model.pkl"

            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            logging.info(f"Copying trained model from '{src_path}' → '{dest_path}'")
            shutil.copy2(src_path, dest_path)
            logging.info("Model successfully saved to local storage.")

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name="local",           # no S3; kept for artifact-entity compatibility
                s3_model_path=dest_path        # now a local file path
            )

            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")
            return model_pusher_artifact

        except Exception as e:
            raise MyException(e, sys) from e