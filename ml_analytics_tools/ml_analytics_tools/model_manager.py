"""
A module for managing MLflow model lifecycle including registration, logging, and deletion.
"""
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import logging
from typing import Any, Dict, Optional, Union, List

# Setup logging
def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with the specified log level."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


logger = setup_logging()

class ModelManager:
    """
    A class to manage MLflow model lifecycle including registration, logging, and deletion.

    Attributes
    ----------
    client : MlflowClient
        The MLflow client instance for interacting with the MLflow API.
    model_name : str
        The name of the model to be registered.
    task : str
        The task the model is designed for (classification, regression, etc.).
    project : Optional[str]
        The project associated with the model. Default is None.
    description : Optional[str]
        A description of the model. Default is None.
    team : Optional[str]
        The team responsible for the model. Default is None.

    user : Optional[str]
        The user ID associated with the model. Default is None.
    tracking_uri : Optional[str]
        The MLflow tracking URI. Default is None.
    """

    def __init__(
        self,
        *,
        model_name: str,
        task: Optional[str] = None,
        project: Optional[str] = None,
        description: Optional[str] = None,
        team: Optional[str] = None,
        user: Optional[str] = None,
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize a ModelManager instance.

        Parameters
        ----------
        model_name : str
            The name of the model to be registered.
        task : str
            The task the model is designed for (classification, regression, etc.).
        model_description : Optional[str]
            A description of the model.
        team : Optional[str]
            The team responsible for the model.
        project : Optional[str]
            The project associated with the model.
        user : Optional[str]
            The user ID associated with the model.
        tracking_uri : Optional[str]
        """

        self.logger = logging.getLogger(f"{__name__}.ModelManager")
        # Set MLflow tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Check tags
        if not all([team, user, project, task]):
            logger.error("All tags must be provided (team, user, project, task).")
            raise

        self.client = MlflowClient()
        self.model_name = model_name
        self.team = team
        self.user = user
        self.project = project
        self.task = task
        self.description = description
        self.run_id = None
        self.model_uri = None

        # Create tags dictionary, filtering out None values
        tags = {k: v for k, v in {
            "team": self.team,
            "user": self.user,
            "project": self.project,
            "task": self.task
        }.items() if v is not None}

        self._create_or_update_registered_model(tags)
        self._setup_experiment()

    def _create_or_update_registered_model(self, tags: Dict[str, str]) -> None:
        """
        Create a registered model or update its tags if it already exists.

        Parameters
        ----------
        tags : Dict[str, str]
            Tags to associate with the model.
        """

        # add user id to tags
        if self.user:
            tags['user'] = self.user

        try:
            self.client.create_registered_model(
                name=self.model_name,
                tags=tags,
                description=self.description
            )
            self.logger.info(f"Model '{self.model_name}' created successfully.")
        except Exception as e:
            if "already exists" in str(e):
                self.logger.info(f"Model '{self.model_name}' already exists. Updating tags and description.")
                # update tags for existing model
                for key, value in tags.items():
                    self.client.set_registered_model_tag(self.model_name, key, value)
                # update description
                if self.description:
                    self.client.update_registered_model(self.model_name, self.description)
            else:
                self.logger.error(f"Error creating model '{self.model_name}': {e}")
                raise

    def _setup_experiment(self) -> None:
        """Set up an MLflow experiment for this model."""
        try:
            # Try to create the experiment
            experiment_id = self.client.create_experiment(self.model_name)
            self.logger.info(f"Created new experiment '{self.model_name}' with ID: {experiment_id}")
        except Exception as e:
            # If experiment already exists, get its ID
            experiment = mlflow.get_experiment_by_name(self.model_name)
            if experiment:
                self.logger.info(f"Using existing experiment '{self.model_name}' with ID: {experiment.experiment_id}")
            else:
                self.logger.error(f"Error setting up experiment '{self.model_name}': {e}")
                raise

        # Set the active experiment
        mlflow.set_experiment(self.model_name)

    def log_model(
        self, *,
        model,
        input_data,
        predictions,
        flavor="sklearn",
        register_model=True,
        description=None,
        tags=None,
        **kwargs  # any additional parameters required by specific flavors
    ):
        """
        Log the model to MLflow using a dynamic flavor.

        Parameters
        ----------
        model : object
            The trained model to log.
        input_data : pd.DataFrame
            The input data used for inference.
        predictions : np.ndarray
            The predictions made by the model.
        flavor : str, default="sklearn"
            The MLflow flavor to use for model logging.
        register_model : bool, default=True
            Whether to register the model.
        description : str, optional
            Additional model description.
        tags : dict, optional
            Dictionary of tags to add to the model.
        **kwargs :
            Additional keyword arguments to pass to the flavor-specific log_model function.
        """
        try:
            # Infer model signature using MLflow's utility (e.g., from mlflow.models.signature)
            signature = infer_signature(input_data, predictions)

            if mlflow.active_run() is None:
                mlflow.start_run()

            with mlflow.active_run() as active_run:
                self.run_id = active_run.info.run_id
                registered_model_name = self.model_name if register_model else None

                # Dynamically obtain the MLflow flavor module, e.g., mlflow.sklearn, mlflow.tensorflow, etc.
                flavor_module = getattr(mlflow, flavor, None)
                if flavor_module is None:
                    self.logger.error(f"MLflow flavor '{flavor}' is not available.")
                    raise ValueError(f"MLflow flavor '{flavor}' is not supported in this context.")

                # Get the log_model function from the flavor module.
                log_model_func = getattr(flavor_module, "log_model", None)
                if log_model_func is None:
                    self.logger.error(f"'log_model' not found in mlflow.{flavor} module.")
                    raise ValueError(f"log_model function not available for flavor '{flavor}'.")

                # Log the model using the desired flavor.
                log_model_func(
                    model,
                    artifact_path="model",
                    registered_model_name=registered_model_name,
                    signature=signature,
                    input_example=input_data.iloc[0:1],
                    **kwargs  # Pass additional parameters if required
                )

                self.logger.info(f"Model '{self.model_name}' logged successfully using mlflow.{flavor}.")

                # Optionally add version metadata if required.
                if register_model:
                    self._add_model_version_metadata(
                        description=description,
                        tags=tags,
                        user=self.user
                    )

                mlflow.end_run()

        except Exception as e:
            self.logger.error(f"Error logging model '{self.model_name}': {str(e)}")
            raise

    def _add_model_version_metadata(
            self,
            description: Optional[str] = None,
            tags: Optional[Dict[str, str]] = None,
            user: Optional[str] = None) -> Optional[int]:
            """
            Add a description, tags, and a user ID (as a tag) to the model version
            associated with the given run URI (or the last logged run if not provided).

            Parameters
            ----------
            description : str, optional
                The description to add to this model version.
            tags : Dict[str, str], optional
                Key-value pairs to add as tags to the model version.
            user : str, optional
                The user ID to store in the model version's tags.
            Returns
            -------
            Optional[int]
                The integer version number that was modified, or None if not found.
            """
            try:
                # Use run_uri if provided, else use self.run_id
                if self.run_id is None:
                    self.logger.error("No model run_id!")

                versions = self.client.search_model_versions(f"name='{self.model_name}'")
                target_version = None
                for mv in versions:
                    if mv.run_id == self.run_id:
                        target_version = int(mv.version)
                        break
                if target_version is None:
                    self.logger.warning(f"No matching version found for run ID: {run_id}")
                    return None

                # Update description if provided
                if description:
                    self.client.update_model_version(
                        name=self.model_name,
                        version=str(target_version),
                        description=description
                    )
                    # self.logger.info(f"Set description for '{self.model_name}' version {target_version}")

                # Combine any provided tags with user (if present)
                final_tags = tags.copy() if tags else {}
                if user is not None:
                    final_tags["user"] = user

                # Set all tags
                for key, value in final_tags.items():
                    self.client.set_model_version_tag(
                        name=self.model_name,
                        version=str(target_version),
                        key=key,
                        value=value
                    )

                self.logger.info(f"Successfully updated model '{self.model_name}' version {target_version}")
            except Exception as e:
                self.logger.error(f"Error updating model '{self.model_name}': {e}")

    def get_latest_model_version(self) -> Optional[int]:
        """
        Get the latest version of the registered model.

        Returns
        -------
        Optional[int]
            The latest version number or None if no versions exist.
        """
        try:
            all_versions = self.client.search_model_versions(f"name='{self.model_name}'")
            if all_versions:
                # Find the version with the highest version number
                latest_version = max(all_versions, key=lambda v: int(v.version))
                # Return just the version number as an integer
                return int(latest_version.version)
            else:
                self.logger.info(f"No versions found for model '{self.model_name}'.")
                return None
        except Exception as e:
            self.logger.error(f"Error getting latest version for model '{self.model_name}': {e}")
            return None

    def get_model_by_alias(self, alias: str, flavor='sklearn'):
        """
        Get the model associated with a specific alias.

        Parameters
        ----------
        alias : str
            The alias to search for.
        flavor : str, optional
            The flavor of the model to retrieve. Default is 'sklearn'.

        Returns
        -------
        model
            The model associated with the alias, or None if not found.
        """

        model_uri = f"models:/{self.model_name}@{alias}"
        flavor_module = getattr(mlflow, flavor, None)
        load_model_func = getattr(flavor_module, "load_model", None)
        try:
            model = load_model_func(model_uri)
            self.logger.info(f"Successfully retrieved model '{self.model_name}' and alias '{alias}'")
            return model
        except Exception as e:
            self.logger.error(f"Error getting model with alias '{alias}': {e}")
            return None

    def get_latest_model(self, flavor='sklearn'):
        """
        Get registered model with the latest version.

        Parameters
        ----------
        flavor : str, optional
            The flavor of the model to retrieve. Default is 'sklearn'. 

        Returns
        -------
        model
            The model with the lastest version, or None if not found.
        """

        version = self.get_latest_model_version()
        if version is not None:
            try:
                model_uri = f"models:/{self.model_name}/{version}"
                flavor_module = getattr(mlflow, flavor, None)
                load_model_func = getattr(flavor_module, "load_model", None)
                model = load_model_func(model_uri)
                self.logger.info(f"Successfully retrieved model '{self.model_name}' version {version}")
                return model
            except Exception as e:
                self.logger.error(f"Error getting latest version for model '{self.model_name}': {e}")
                return None

    def set_model_alias(self, *, version: Optional[int] = None, alias: str) -> None:
        """
        Set an alias for a specific model version.
        Aliases enable you to refer to a model version by name rather than numeric version.
        Requires MLflow 2.0 or newer.

        Parameters
        ----------
        version : int
            The version of the model to which the alias should be assigned.
        alias : str
            The alias name (e.g., "latest", "candidate", "production-ready").
        """
        try:

            if version is None:
                version = self.get_latest_model_version()

            self.client.set_registered_model_alias(
                name=self.model_name,
                version=str(version),
                alias=alias
            )

            self.logger.info(
                f"Set alias '{alias}' for version {version} of model '{self.model_name}'."
            )
        except Exception as e:
            self.logger.error(
                f"Error setting alias '{alias}' for model '{self.model_name}', "
                f"version {version}: {e}"
            )
            raise

    def delete_model(self, version: Optional[int] = None, alias: Optional[str] = None) -> None:
        """
        Delete a specific version of this model from the MLflow model registry.

        Parameters
        ----------
        version : int
            The version number of the model to delete.

        Raises
        ------
        Exception
            If an error occurs while deleting the model version.
        """
        if version is not None and alias is not None:
            self.logger.error("Cannot delete model version and alias at the same time.")
        try:
            if alias is not None:
                version = self.client.get_model_version_by_alias(name=self.model_name, alias=alias).version
            if version is None:
                self.logger.error(f"Version {version} not found for model '{self.model_name}'.")

            self.client.delete_model_version(name=self.model_name, version=str(version))
            self.logger.info(f"Deleted version {version} of model '{self.model_name}'.")
        except Exception as e:
            self.logger.error(f"Error deleting model '{self.model_name}' version {version}: {e}")
            raise
