import os
import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[object]:
        """
        Loads the best existing model from local storage if it exists.
        Returns None if no saved model is found.
        """
        try:
            model_path = self.model_eval_config.local_model_path   # e.g. "saved_models/model.pkl"
            if os.path.exists(model_path):
                logging.info(f"Found existing model at {model_path}")
                return load_object(file_path=model_path)
            logging.info("No existing production model found locally.")
            return None
        except Exception as e:
            raise MyException(e, sys)

    # ------------------------------------------------------------------ #
    #  Pre-processing helpers (mirror DataTransformation for consistency)  #
    # ------------------------------------------------------------------ #
    def _map_gender_column(self, df):
        logging.info("Mapping 'Gender' column to binary values")
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        return df

    def _create_dummy_columns(self, df):
        logging.info("Creating dummy variables for categorical features")
        df = pd.get_dummies(df, drop_first=True)
        return df

    def _rename_columns(self, df):
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df

    def _drop_id_column(self, df):
        logging.info("Dropping '_id' column if present")
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        return df

    # ------------------------------------------------------------------ #
    #  Core evaluation logic                                               #
    # ------------------------------------------------------------------ #
    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Compares the freshly trained model against the best locally-saved
        production model (if any) using F1-score on the held-out test set.
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # FIX: axis=1 (drop column), not axis=2
            x = test_df.drop(TARGET_COLUMN, axis=1)
            y = test_df[TARGET_COLUMN]

            logging.info("Test data loaded; applying pre-processing transforms …")
            x = self._map_gender_column(x)
            x = self._drop_id_column(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)

            # Score the newly trained model
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"Trained model F1-score (from trainer artifact): {trained_model_f1_score}")

            # Score the current production model (if one exists)
            best_model_f1_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info("Computing F1-score for the existing production model …")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(
                    f"Production model F1: {best_model_f1_score} | "
                    f"New model F1: {trained_model_f1_score}"
                )

            tmp_best_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=trained_model_f1_score > tmp_best_score,
                difference=trained_model_f1_score - tmp_best_score,
            )
            logging.info(f"Evaluation result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Runs the full model-evaluation step and returns a
        ModelEvaluationArtifact (no S3 references).
        """
        try:
            logging.info("=" * 80)
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()

            # local_model_path comes from ModelEvaluationConfig (replaces s3_model_key_path)
            local_model_path = self.model_eval_config.local_model_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=local_model_path,          # field name kept for compat; now a local path
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference,
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys) from e