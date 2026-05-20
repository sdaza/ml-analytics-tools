"""
A module for managing MLflow model lifecycle including registration, logging, and deletion.
"""

import os
from typing import Literal

import mlflow
from mlflow.exceptions import RestException
from mlflow.models import infer_signature
from mlflow.server.auth.client import AuthServiceClient
from mlflow.tracking import MlflowClient

from .utils import get_credential_value, get_logger, log_and_raise_error

PermissionLevel = Literal["READ", "EDIT", "MANAGE", "NO_PERMISSIONS"]


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
        task: str | None = None,
        project: str | None = None,
        description: str | None = None,
        team: str | None = None,
        user: str | None = None,
        tracking_uri: str | None = None,
        workspace: str | None = None,
        create_registered_model: bool = False,
        start_initial_run: bool = False,
        run_name: str | None = None,
    ):
        """
        Initialize a ModelManager instance.

        Parameters
        ----------
        model_name : str
            The name of the model to be registered.
        task : str
            The task the model is designed for (classification, regression, etc.).
        project : Optional[str]
            The project associated with the model.
        description : Optional[str]
            A description of the model.
        team : Optional[str]
            The team responsible for the model.
        user : Optional[str]
            The user ID associated with the model.
        tracking_uri : Optional[str]
            The MLflow tracking URI. If None, uses MLFLOW_TRACKING_URI env var,
            falling back to MLflow's configured default.
        workspace : Optional[str]
            The MLflow workspace name. If None, uses MLFLOW_WORKSPACE env var.
            If neither is set, workspace is not configured (for servers without workspace support).
        create_registered_model : bool
            Whether to create the registered model (default: True), otherwise only it will create the experiment.
        run_name : str, optional
            The name for the MLflow run. If None, a default name will be used.
        start_initial_run : bool, default=False
            Whether to start an MLflow run upon initialization.
        """

        self._logger = get_logger("Model Manager")

        for env_name in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
            try:
                os.environ[env_name] = get_credential_value(env_name)
            except Exception:
                self._logger.debug("%s is not configured; continuing without it.", env_name)
        os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "true"
        os.environ["MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE"] = "15728640"
        os.environ["MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE"] = "15728640"
        os.environ["MLFLOW_MULTIPART_DOWNLOAD_MINIMUM_FILE_SIZE"] = "524288000"
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "120"
        os.environ["MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT"] = "300"

        # Set MLflow tracking URI: explicit param > env var > MLflow default.
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(self.tracking_uri)
        self._logger.debug(f"MLflow tracking URI set to: {self.tracking_uri}")

        # Set MLflow workspace: explicit param > env var; skip if not set (server may not support workspaces)
        self.workspace = workspace or os.environ.get("MLFLOW_WORKSPACE")
        if self.workspace:
            mlflow.set_workspace(self.workspace)
            self._logger.debug(f"MLflow workspace set to: {self.workspace}")
        else:
            self._logger.debug("No workspace configured, using server default.")

        try:
            self.auth_client = AuthServiceClient(self.tracking_uri)
        except Exception as e:
            self._logger.error(f"Failed to initialize AuthServiceClient: {e}. Permission management will not work.")
            self.auth_client = None  # Set to None if initialization fails

        # Initialize MLflow client
        self.client = MlflowClient()
        self.model_name = model_name
        self.team = team
        self.user = user
        self.project = project
        self.task = task
        self.description = description
        self.run_id = None
        self.experiment_id = None
        self.model_uri = None
        self.start_initial_run = start_initial_run

        tags = {
            k: v
            for k, v in {"team": self.team, "user": self.user, "project": self.project, "task": self.task}.items()
            if v is not None
        }

        if create_registered_model:
            self._create_registered_model(tags)

        self._setup_experiment()

        if self.start_initial_run:
            self.start_run(run_name)
        else:
            self.run_id = None

    def start_run(self, run_name: str | None = None) -> None:
        """
        Starts a new MLflow run if no run is currently managed by this instance (self.run_id is None).
        Associates the run with self.experiment_id.
        """

        # checking there is no active run managed by this instance
        self.run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

        if self.run_id is not None:
            self._logger.info(
                f"Run {self.run_id} is already considered active by this ModelManager instance. "
                "To start a new run, you can use start_new_run()."
            )
            current_mlflow_run = mlflow.active_run()
            if not current_mlflow_run or current_mlflow_run.info.run_id != self.run_id:
                self._logger.warning(
                    f"Mismatch: ModelManager's run_id is {self.run_id}, "
                    f"but MLflow's active run is {current_mlflow_run.info.run_id if current_mlflow_run else 'None'}. "
                    "This might indicate an inconsistent state."
                )
        else:
            try:
                mlflow.set_experiment(experiment_id=self.experiment_id)
                active_run_obj = mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id)
                self.run_id = active_run_obj.info.run_id
                log_message = f"Started new MLflow run with ID: {self.run_id} for experiment ID: {self.experiment_id}"
                if run_name:
                    log_message += f" with name: {run_name}"
                self._logger.info(log_message + ".")
            except Exception as e:
                log_and_raise_error(self._logger, f"Cannot start a new MLflow run: {e}")
                self.run_id = None

    def end_run(self, status: str = "FINISHED") -> None:
        """Ends the active MLflow run managed by this instance and clears self.run_id."""
        if self.run_id:
            current_active_run = mlflow.active_run()
            if current_active_run and current_active_run.info.run_id == self.run_id:
                try:
                    mlflow.end_run(status=status)
                    self._logger.info(f"MLflow run {self.run_id} ended with status: {status}.")
                except Exception as e:
                    self._logger.error(f"Error ending MLflow run {self.run_id}: {e}")
            elif current_active_run:
                self._logger.warning(
                    f"Attempted to end ModelManager's run {self.run_id}, but the current "
                    f"MLflow active run is {current_active_run.info.run_id}. "
                    f"ModelManager's run {self.run_id} was not ended by this call."
                )
            else:
                self._logger.info(
                    f"Attempted to end ModelManager's run {self.run_id}, but there is no "
                    "active MLflow run. Assuming it was already ended."
                )
        else:
            self._logger.info("No run ID associated with this ModelManager instance to end.")

        self.run_id = None

    def start_new_run(self, run_name: str | None = None) -> None:
        """
        Ensures any previous run managed by this instance is ended, then starts a new MLflow run.

        Parameters
        ----------
        run_name : str, optional
            The name for the new MLflow run. If None, a default name might be used by MLflow.
        """
        if self.run_id is not None:
            self.end_run()

        # After end_run, self.run_id is None, so start_run will proceed to create a new run.
        self.start_run(run_name)

    def _create_registered_model(self, tags: dict[str, str]) -> None:
        """
        Create a registered model or update its tags if it already exists.

        Parameters
        ----------
        tags : Dict[str, str]
            Tags to associate with the model.
        """

        # add user id to tags
        if self.user is None:
            log_and_raise_error(self._logger, "Define user to create instance of ModelManager.")
        else:
            tags["user"] = self.user

        try:
            model_instance = self.client.get_registered_model(self.model_name)
            if model_instance:
                self._logger.info(f"Model '{self.model_name}' already exists.")
                if tags:
                    for key, value in tags.items():
                        self.client.set_registered_model_tag(name=self.model_name, key=key, value=value)
        except Exception:
            if not all([self.team, self.user, self.project, self.task]):
                log_and_raise_error(self._logger, "All tags must be provided (team, user, project, task).")
            try:
                self.client.create_registered_model(name=self.model_name, tags=tags, description=self.description)
                self._logger.info(f"Model '{self.model_name}' created successfully.")
            except Exception as e:
                log_and_raise_error(self._logger, f"Error creating model '{self.model_name}': {e}")

    def _setup_experiment(self) -> None:
        """Set up an MLflow experiment."""
        try:
            # Try to create the experiment
            experiment_id = self.client.create_experiment(self.model_name)
            self._logger.info(f"Created new experiment '{self.model_name}' with ID: {experiment_id}")
            self.experiment_id = experiment_id

        except Exception as e:
            # If experiment already exists, get its ID
            experiment = mlflow.get_experiment_by_name(self.model_name)
            self.experiment_id = experiment.experiment_id
            if experiment:
                self._logger.info(f"Using existing experiment '{self.model_name}' with ID: {experiment.experiment_id}")
            else:
                log_and_raise_error(self._logger, f"Error setting up experiment '{self.model_name}': {e}")

        try:
            if self.project or self.team or self.user or self.task:
                experiment_tags = {"project": self.project, "team": self.team, "user": self.user, "task": self.task}

                experiment_tags = {k: v for k, v in experiment_tags.items() if v is not None}
                for key, value in experiment_tags.items():
                    self.client.set_experiment_tag(experiment_id=self.experiment_id, key=key, value=value)
        except Exception as e:
            self._logger.warning(f"Error setting experiment tags: {e}. Check if you are the owner of this experiment.")

        # Set the active experiment
        mlflow.set_experiment(self.model_name)

    def log_model(
        self,
        *,
        model=None,
        input_data=None,
        predictions=None,
        flavor="sklearn",
        register_model=True,
        description=None,
        tags=None,
        python_model=None,
        name="model",
        **kwargs,
    ):
        """
        Log the model to MLflow using a dynamic flavor, including support for the 'pyfunc' flavor.

        Parameters
        ----------
        model : object, optional
            The trained model to log (used for most flavors).
        input_data : pd.DataFrame, optional
            The input data used for inference (used for signature and input_example).
        predictions : np.ndarray, optional
            The predictions made by the model (used for signature).
        flavor : str, default="sklearn"
            The MLflow flavor to use for model logging. Supports 'sklearn', 'pyfunc', etc.
        register_model : bool, default=True
            Whether to register the model.
        description : str, optional
            Additional model description.
        tags : dict, optional
            Dictionary of tags to add to the model.
        python_model : mlflow.pyfunc.PythonModel, optional
            The PythonModel instance to log (required for 'pyfunc' flavor).
        name : str, default='model'
            The name under which the model will be logged.

        **kwargs :
            Additional keyword arguments to pass to the flavor-specific log_model function.
        """
        active_mlflow_run = mlflow.active_run()
        if not self.run_id:
            log_and_raise_error(
                self._logger,
                "No MLflow run is currently managed by this ModelManager instance. "
                "Call start_new_run() or ensure __init__ completed a run start.",
            )
        if not active_mlflow_run or active_mlflow_run.info.run_id != self.run_id:
            log_and_raise_error(
                self._logger,
                f"The MLflow run managed by this instance ({self.run_id}) is not the "
                f"currently active MLflow run ({active_mlflow_run.info.run_id if active_mlflow_run else 'None'}). "
                "Please ensure the correct run is active or start a new run.",
            )

        try:
            signature = None
            input_example = None
            if input_data is not None and predictions is not None:
                signature_input = input_data.copy()
                int_cols = signature_input.select_dtypes(include=["int32", "int64"]).columns
                if len(int_cols):
                    signature_input[int_cols] = signature_input[int_cols].astype("float64")
                signature = infer_signature(signature_input, predictions)
                input_example = signature_input
            registered_model_name = self.model_name if register_model else None

            if tags:
                mlflow.set_tags(tags)
            if description:
                mlflow.set_tag("mlflow.note.content", self.description)

            if flavor == "pyfunc":
                from mlflow import pyfunc

                pyfunc.log_model(
                    python_model=python_model,
                    name=name,
                    registered_model_name=registered_model_name,
                    signature=signature,
                    input_example=input_example,
                    **kwargs,
                )
                self._logger.info(f"Model '{self.model_name}' logged successfully using mlflow.pyfunc.")
            else:
                flavor_module = getattr(mlflow, flavor, None)
                if flavor_module is None:
                    log_and_raise_error(self._logger, f"MLflow flavor '{flavor}' is not available.")

                log_model_func = getattr(flavor_module, "log_model", None)
                if log_model_func is None:
                    log_and_raise_error(self._logger, f"'log_model' not found in mlflow.{flavor} module.")

                log_model_func(
                    model,
                    name=name,
                    registered_model_name=registered_model_name,
                    signature=signature,
                    input_example=input_example,
                    **kwargs,
                )
                self._logger.info(f"Model '{self.model_name}' logged successfully using mlflow.{flavor}.")

            if register_model:
                self._add_model_version_metadata(description=description, tags=tags, user=self.user)
        except Exception as e:
            log_and_raise_error(self._logger, f"Error logging model '{self.model_name}': {str(e)}")

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """
        Logs a local file or directory as an artifact to the active MLflow run.

        Parameters
        ----------
        local_path : str
            Path to the local file or directory to log.
        artifact_path : Optional[str]
            If provided, the artifact will be logged to this path within the run's artifact URI.
            If None, the artifact is logged to the root of the run's artifact URI.
        """
        active_mlflow_run = mlflow.active_run()
        if not self.run_id:
            log_and_raise_error(
                self._logger,
                "No MLflow run is currently managed by this ModelManager instance. "
                "Call start_new_run() or ensure __init__ completed a run start.",
            )
        if not active_mlflow_run or active_mlflow_run.info.run_id != self.run_id:
            log_and_raise_error(
                self._logger,
                f"The MLflow run managed by this instance ({self.run_id}) is not the "
                f"currently active MLflow run ({active_mlflow_run.info.run_id if active_mlflow_run else 'None'}). "
                "Please ensure the correct run is active or start a new run.",
            )

        try:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
            self._logger.debug(f"Logged artifact '{local_path}'.")
        except Exception as e:
            log_and_raise_error(self._logger, f"Error logging artifact '{local_path}' to run '{self.run_id}': {e}")

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        """
        Logs all files in a local directory as artifacts to the active MLflow run.

        Parameters
        ----------
        local_dir : str
            Path to the local directory containing files to log.
        artifact_path : Optional[str]
            If provided, artifacts will be logged to this path within the run's artifact URI.
            If None, artifacts are logged to the root of the run's artifact URI.
        """
        active_mlflow_run = mlflow.active_run()
        if not self.run_id:
            log_and_raise_error(
                self._logger,
                "No MLflow run is currently managed by this ModelManager instance. "
                "Call start_new_run() or ensure __init__ completed a run start.",
            )
        if not active_mlflow_run or active_mlflow_run.info.run_id != self.run_id:
            log_and_raise_error(
                self._logger,
                f"The MLflow run managed by this instance ({self.run_id}) is not the "
                f"currently active MLflow run ({active_mlflow_run.info.run_id if active_mlflow_run else 'None'}). "
                "Please ensure the correct run is active or start a new run.",
            )

        try:
            mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
            self._logger.debug(f"Logged artifacts from '{local_dir}'.")
        except Exception as e:
            log_and_raise_error(self._logger, f"Error logging artifacts from '{local_dir}' to run '{self.run_id}': {e}")

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """
        Logs a single metric to the active MLflow run.

        Parameters
        ----------
        key : str
            Metric key.
        value : float
            Metric value.
        step : Optional[int]
            Metric step.
        """
        active_mlflow_run = mlflow.active_run()
        if not self.run_id:
            log_and_raise_error(
                self._logger,
                "No MLflow run is currently managed by this ModelManager instance. "
                "Call start_new_run() or ensure __init__ completed a run start.",
            )
        if not active_mlflow_run or active_mlflow_run.info.run_id != self.run_id:
            log_and_raise_error(
                self._logger,
                f"The MLflow run managed by this instance ({self.run_id}) is not the "
                f"currently active MLflow run ({active_mlflow_run.info.run_id if active_mlflow_run else 'None'}). "
                "Please ensure the correct run is active or start a new run.",
            )
        try:
            mlflow.log_metric(key, value, step=step)
            self._logger.debug(f"Logged metric '{key}': {value}.")
        except Exception as e:
            log_and_raise_error(self._logger, f"Error logging metric '{key}' to run '{self.run_id}': {e}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """
        Logs multiple metrics to the active MLflow run.

        Parameters
        ----------
        metrics : Dict[str, float]
            Dictionary of metric keys and values.
        step : Optional[int]
            Metric step.
        """
        active_mlflow_run = mlflow.active_run()
        if not self.run_id:
            log_and_raise_error(
                self._logger,
                "No MLflow run is currently managed by this ModelManager instance. "
                "Call start_new_run() or ensure __init__ completed a run start.",
            )
        if not active_mlflow_run or active_mlflow_run.info.run_id != self.run_id:
            log_and_raise_error(
                self._logger,
                f"The MLflow run managed by this instance ({self.run_id}) is not the "
                f"currently active MLflow run ({active_mlflow_run.info.run_id if active_mlflow_run else 'None'}). "
                "Please ensure the correct run is active or start a new run.",
            )
        try:
            mlflow.log_metrics(metrics, step=step)
            self._logger.debug(f"Logged {len(metrics)} metrics.")
        except Exception as e:
            log_and_raise_error(self._logger, f"Error logging metrics to run '{self.run_id}': {e}")

    def log_param(self, key: str, value: any) -> None:
        """
        Logs a single parameter to the active MLflow run.

        Parameters
        ----------
        key : str
            Parameter key.
        value : Any
            Parameter value.
        """
        active_mlflow_run = mlflow.active_run()
        if not self.run_id:
            log_and_raise_error(
                self._logger,
                "No MLflow run is currently managed by this ModelManager instance. "
                "Call start_new_run() or ensure __init__ completed a run start.",
            )
        if not active_mlflow_run or active_mlflow_run.info.run_id != self.run_id:
            log_and_raise_error(
                self._logger,
                f"The MLflow run managed by this instance ({self.run_id}) is not the "
                f"currently active MLflow run ({active_mlflow_run.info.run_id if active_mlflow_run else 'None'}). "
                "Please ensure the correct run is active or start a new run.",
            )
        try:
            mlflow.log_param(key, value)
            self._logger.debug(f"Logged param '{key}'.")
        except Exception as e:
            log_and_raise_error(self._logger, f"Error logging parameter '{key}' to run '{self.run_id}': {e}")

    def log_params(self, params: dict[str, any]) -> None:
        """
        Logs multiple parameters to the active MLflow run.

        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameter keys and values.
        """
        active_mlflow_run = mlflow.active_run()
        if not self.run_id:
            log_and_raise_error(
                self._logger,
                "No MLflow run is currently managed by this ModelManager instance. "
                "Call start_new_run() or ensure __init__ completed a run start.",
            )
        if not active_mlflow_run or active_mlflow_run.info.run_id != self.run_id:
            log_and_raise_error(
                self._logger,
                f"The MLflow run managed by this instance ({self.run_id}) is not the "
                f"currently active MLflow run ({active_mlflow_run.info.run_id if active_mlflow_run else 'None'}). "
                "Please ensure the correct run is active or start a new run.",
            )
        try:
            mlflow.log_params(params)
            self._logger.debug(f"Logged {len(params)} params.")
        except Exception as e:
            log_and_raise_error(self._logger, f"Error logging parameters to run '{self.run_id}': {e}")

    def set_tag(self, key: str, value: any) -> None:
        """
        Sets a single tag on the active MLflow run.

        Parameters
        ----------
        key : str
            Tag key.
        value : Any
            Tag value.
        """
        active_mlflow_run = mlflow.active_run()
        if not self.run_id:
            log_and_raise_error(
                self._logger,
                "No MLflow run is currently managed by this ModelManager instance. "
                "Call start_new_run() or ensure __init__ completed a run start.",
            )
        if not active_mlflow_run or active_mlflow_run.info.run_id != self.run_id:
            log_and_raise_error(
                self._logger,
                f"The MLflow run managed by this instance ({self.run_id}) is not the "
                f"currently active MLflow run ({active_mlflow_run.info.run_id if active_mlflow_run else 'None'}). "
                "Please ensure the correct run is active or start a new run.",
            )
        try:
            mlflow.set_tag(key, value)
            self._logger.debug(f"Set tag '{key}'.")
        except Exception as e:
            log_and_raise_error(self._logger, f"Error setting tag '{key}' on run '{self.run_id}': {e}")

    def set_tags(self, tags: dict[str, any]) -> None:
        """
        Sets multiple tags on the active MLflow run.

        Parameters
        ----------
        tags : Dict[str, Any]
            Dictionary of tag keys and values.
        """
        active_mlflow_run = mlflow.active_run()
        if not self.run_id:
            log_and_raise_error(
                self._logger,
                "No MLflow run is currently managed by this ModelManager instance. "
                "Call start_new_run() or ensure __init__ completed a run start.",
            )
        if not active_mlflow_run or active_mlflow_run.info.run_id != self.run_id:
            log_and_raise_error(
                self._logger,
                f"The MLflow run managed by this instance ({self.run_id}) is not the "
                f"currently active MLflow run ({active_mlflow_run.info.run_id if active_mlflow_run else 'None'}). "
                "Please ensure the correct run is active or start a new run.",
            )
        try:
            mlflow.set_tags(tags)
            self._logger.debug(f"Set {len(tags)} tags.")
        except Exception as e:
            log_and_raise_error(self._logger, f"Error setting tags on run '{self.run_id}': {e}")

    def _add_model_version_metadata(
        self, description: str | None = None, tags: dict[str, str] | None = None, user: str | None = None
    ) -> int | None:
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
                log_and_raise_error(self._logger, "No model run_id!")

            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            target_version = None
            for mv in versions:
                if mv.run_id == self.run_id:
                    target_version = int(mv.version)
                    break
            if target_version is None:
                self._logger.warning(f"No matching version found for run ID: {self.run_id}")
                return None

            # Update description if provided
            if description:
                self.client.update_model_version(
                    name=self.model_name, version=str(target_version), description=description
                )

            # Combine any provided tags with user (if present)
            final_tags = tags.copy() if tags else {}
            if user is not None:
                final_tags["user"] = user

            # Set all tags
            for key, value in final_tags.items():
                self.client.set_model_version_tag(
                    name=self.model_name, version=str(target_version), key=key, value=value
                )

            self._logger.debug(f"Updated model '{self.model_name}' version {target_version}.")
        except Exception as e:
            log_and_raise_error(self._logger, f"Error updating model '{self.model_name}': {e}")

    def get_latest_model_version(self) -> int | None:
        """
        Get the latest version of the registered model.

        Returns
        -------
        Optional[int]
            The latest version number.
        """
        try:
            all_versions = self.client.search_model_versions(f"name='{self.model_name}'")
            if all_versions:
                # Find the version with the highest version number
                latest_version = max(all_versions, key=lambda v: int(v.version))
                # Return just the version number as an integer
                return int(latest_version.version)
            else:
                self._logger.info(f"No versions found for model '{self.model_name}'.")
        except Exception as e:
            log_and_raise_error(self._logger, f"Error getting latest version for model '{self.model_name}': {e}")

    def get_model_uri(self, version: int | None = None, alias: str | None = None) -> str:
        """
        Get the URI of a registered model by version or alias.

        Parameters
        ----------
        version : int, optional
            The version of the model to retrieve the URI for.
        alias : str, optional
            The alias of the model to retrieve the URI for.

        Returns
        -------
        str
            The URI of the registered model.

        Raises
        ------
        ValueError
            If neither version nor alias is provided.
        """
        if version is None and alias is None:
            log_and_raise_error(self._logger, "Either 'version' or 'alias' must be provided to get the model URI.")

        try:
            if alias is not None:
                version = self.client.get_model_version_by_alias(name=self.model_name, alias=alias).version

            model_uri = f"models:/{self.model_name}/{version}"
            return model_uri
        except Exception as e:
            log_and_raise_error(
                self._logger,
                f"Error retrieving model URI for model '{self.model_name}', version '{version}', alias '{alias}': {e}",
            )

    def set_model_alias(self, *, version: int | None = None, alias: str) -> None:
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

            self.client.set_registered_model_alias(name=self.model_name, version=str(version), alias=alias)

            self._logger.info(f"Set alias '{alias}' for version {version} of model '{self.model_name}'.")
        except Exception as e:
            log_and_raise_error(
                self._logger, f"Error setting alias '{alias}' for model '{self.model_name}', version {version}: {e}"
            )

    def register_model(
        self, run_name: str, experiment_id: str = None, description: str = None, tags: dict = None
    ) -> None:
        """
        Move a run model to the model registry.

        Parameters
        ----------
        run_name : str
            The name of the model run to be moved.
        description : str, optional
            A description of the model.
        tags : dict, optional
            A dictionary of tags to associate with the model.
        experiment_id : str, optional
            Experiment ID to search for the run. If None, uses the current experiment ID.
        """

        if self.user is None:
            log_and_raise_error(self._logger, "Define user to register a model.")
        if tags is None:
            tags = {}
        tags["user"] = self.user

        if experiment_id is None:
            experiment_id = self.experiment_id

        runs = self.client.search_runs(
            experiment_ids=[experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'"
        )

        # Check if runs were found and return the run ID
        if not runs:
            log_and_raise_error(
                self._logger, f"No runs found with name '{run_name}' in experiment ID '{experiment_id}'."
            )

        run_id = runs[0].info.run_id

        try:
            mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name=self.model_name,
                tags=tags,
            )
            self.client.update_model_version(
                name=self.model_name, version=self.get_latest_model_version(), description=description
            )
            self._logger.info(f"Model '{self.model_name}' registered successfully from run '{run_name}'.")
        except Exception as e:
            log_and_raise_error(self._logger, f"Error registering model '{self.model_name}' from run '{run_name}': {e}")

    def load_model(
        self,
        model_uri: str | None = None,
        version: int | None = None,
        alias: str | None = None,
        flavor: str = "sklearn",
        **kwargs,
    ):
        """
        Load a model using model URI, version, or alias, supporting all MLflow flavors including 'pyfunc'.

        Parameters
        ----------
        model_uri : str, optional
            The URI of the model to load. If provided, it takes precedence over version and alias.
        version : int, optional
            The version of the model to load.
        alias : str, optional
            The alias of the model to load.
        flavor : str, optional
            The flavor of the model to retrieve. Default is 'sklearn'. Supports 'pyfunc', etc.
        **kwargs :
            Additional keyword arguments to pass to the flavor-specific load_model function.

        Returns
        -------
        model
            The loaded model.

        Raises
        ------
        ValueError
            If none of model_uri, version, or alias is provided.
        """
        if model_uri is None:
            if version is None and alias is None:
                log_and_raise_error(
                    self._logger, "Either 'model_uri', 'version', or 'alias' must be provided to load the model."
                )

            # Get model URI from version or alias
            model_uri = self.get_model_uri(version=version, alias=alias)

        try:
            if flavor == "pyfunc":
                from mlflow import pyfunc

                model = pyfunc.load_model(model_uri, **kwargs)
                self._logger.info(f"Successfully loaded model from URI '{model_uri}' using flavor 'pyfunc'.")
                return model
            else:
                flavor_module = getattr(mlflow, flavor, None)
                if flavor_module is None:
                    log_and_raise_error(self._logger, f"MLflow flavor '{flavor}' is not available.")

                load_model_func = getattr(flavor_module, "load_model", None)
                if load_model_func is None:
                    log_and_raise_error(self._logger, f"'load_model' not found in mlflow.{flavor} module.")

                model = load_model_func(model_uri, **kwargs)
                self._logger.info(f"Successfully loaded model from URI '{model_uri}' using flavor '{flavor}'.")
                return model
        except Exception as e:
            log_and_raise_error(self._logger, f"Error loading model from URI '{model_uri}': {e}")

    def _resolve_logged_model_uri(self, run_id: str, experiment_id: str) -> str:
        """
        Resolve the model artifact URI for a run. For MLflow 3.x runs, returns the LoggedModel's
        artifact_location directly to avoid a proxy hang in RunsArtifactRepository. Falls back to
        runs:/{run_id}/model for MLflow 2.x runs.
        """
        try:
            page_token = None
            while True:
                page = self.client.search_logged_models(
                    experiment_ids=[experiment_id],
                    filter_string="name = 'model'",
                    page_token=page_token,
                )
                for lm in page:
                    if lm.source_run_id == run_id:
                        self._logger.info(f"Loading model from LoggedModel URI (run: {run_id[:8]}...).")
                        return lm.artifact_location
                if not page.token:
                    break
                page_token = page.token
        except Exception as e:
            self._logger.warning(f"Could not resolve LoggedModel URI for run '{run_id}': {e}. Falling back.")

        return f"runs:/{run_id}/model"

    def load_model_from_experiment(self, run_name: str, experiment_name: str = None, flavor: str = "sklearn", **kwargs):
        """
        Load a model from a specific experiment and run, supporting all MLflow flavors including 'pyfunc'.

        Parameters
        ----------
        experiment_name : str, optional
            The name of the experiment to search for the run. If None, uses the current experiment name.
        run_name : str
            The name of the run to load the model from.
        flavor : str, optional
            The flavor of the model to retrieve. Default is 'sklearn'.
        **kwargs :
            Additional keyword arguments to pass to the flavor-specific load_model function.

        Returns
        -------
        model
            The model loaded from the specified experiment and run.
        """
        if experiment_name is None:
            experiment_name = self.model_name

        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if not experiment:
                log_and_raise_error(self._logger, f"Experiment '{experiment_name}' not found.")

            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'"
            )
            if not runs:
                log_and_raise_error(self._logger, f"Run '{run_name}' not found in experiment '{experiment_name}'.")

            run_id = runs[0].info.run_id
            model_uri = self._resolve_logged_model_uri(run_id=run_id, experiment_id=experiment.experiment_id)
            return self.load_model(model_uri=model_uri, flavor=flavor, **kwargs)
        except Exception as e:
            log_and_raise_error(
                self._logger, f"Error loading model from experiment '{experiment_name}', run '{run_name}': {e}"
            )

    def load_latest_model(self, flavor="sklearn", **kwargs):
        """
        Get registered model with the latest version, supporting all MLflow flavors including 'pyfunc'.

        Parameters
        ----------
        flavor : str, optional
            The flavor of the model to retrieve. Default is 'sklearn'.
        **kwargs :
            Additional keyword arguments to pass to the flavor-specific load_model function.

        Returns
        -------
        model
            The model with the lastest version, or None if not found.
        """
        version = self.get_latest_model_version()
        if version is not None:
            try:
                model_uri = f"models:/{self.model_name}/{version}"
                return self.load_model(model_uri=model_uri, flavor=flavor, **kwargs)
            except Exception as e:
                log_and_raise_error(self._logger, f"Error getting latest version for model '{self.model_name}': {e}")

    def delete_model(self, model_uri: str | None = None, version: int | None = None, alias: str | None = None) -> None:
        """
        Delete a specific version of this model from the MLflow model registry.

        Parameters
        ----------
        model_uri : str, optional
            The URI of the model to delete.
        version : int, optional
            The version number of the model to delete.
        alias : str, optional
            The alias of the model to delete.

        Raises
        ------
        Exception
            If an error occurs while deleting the model version.
        """
        if model_uri is None:
            if version is None and alias is None:
                log_and_raise_error(
                    self._logger, "Either 'model_uri', 'version', or 'alias' must be provided to delete the model."
                )

            # Get model URI from version or alias
            model_uri = self.get_model_uri(version=version, alias=alias)
        if version is not None and alias is not None:
            log_and_raise_error(self._logger, "Cannot delete model version and alias at the same time.")
        try:
            if alias is not None:
                version = self.client.get_model_version_by_alias(name=self.model_name, alias=alias).version
            if version is None:
                log_and_raise_error(self._logger, f"Version {version} not found for model '{self.model_name}'.")

            self.client.delete_model_version(name=self.model_name, version=str(version))
            self._logger.info(f"Deleted version {version} of model '{self.model_name}'.")
        except Exception as e:
            log_and_raise_error(self._logger, f"Error deleting model '{self.model_name}' version {version}: {e}")

    def grant_experiment_permission(self, username: str, permission: PermissionLevel) -> None:
        """
        Grants or updates permission for a user on the current experiment.
        """
        if not self.auth_client:
            log_and_raise_error(self._logger, "AuthServiceClient not initialized. Cannot manage permissions.")
        if not self.experiment_id:
            log_and_raise_error(self._logger, "Experiment ID not set. Cannot manage permissions.")

        valid_permissions = ["READ", "EDIT", "MANAGE", "NO_PERMISSIONS"]

        if permission not in valid_permissions:
            log_and_raise_error(
                self._logger, f"Invalid permission level '{permission}'. Must be one of {valid_permissions}"
            )

        try:
            self.auth_client.update_experiment_permission(self.experiment_id, username, permission)
            self._logger.info(
                f"Successfully created '{permission}' permission for user '{username}' "
                f"on experiment '{self.experiment_id}'."
            )
        except RestException:
            self.auth_client.create_experiment_permission(self.experiment_id, username, permission)
            self._logger.info(
                f"Successfully created '{permission}' permission for user '{username}' "
                f"on experiment '{self.experiment_id}'."
            )
        except Exception as e_unexpected:
            log_and_raise_error(
                self._logger, f"An unexpected error occurred while setting experiment permission: {e_unexpected}"
            )

    def grant_registered_model_permission(self, username: str, permission: PermissionLevel) -> None:
        """
        Grants or updates permission for a user on the current registered model.
        """
        if not self.auth_client:
            log_and_raise_error(self._logger, "AuthServiceClient not initialized. Cannot manage permissions.")

        valid_permissions = ["READ", "EDIT", "MANAGE", "NO_PERMISSIONS"]
        if permission not in valid_permissions:
            log_and_raise_error(
                self._logger, f"Invalid permission level '{permission}'. Must be one of {valid_permissions}"
            )
        try:
            self.auth_client.update_registered_model_permission(self.model_name, username, permission)
            self._logger.info(
                f"Successfully updated permission for user '{username}' on registered model "
                f"'{self.model_name}' to '{permission}'."
            )
        except RestException:
            self.auth_client.create_registered_model_permission(self.model_name, username, permission)
            self._logger.info(
                f"Successfully created '{permission}' permission for user '{username}' "
                f"on registered model '{self.model_name}'."
            )
        except Exception as e_unexpected:
            log_and_raise_error(
                self._logger, f"An unexpected error occurred while setting registered model permission: {e_unexpected}"
            )

    def load_artifact_from_model_version(self, version: int, artifact_path: str) -> str:
        """
        Download an artifact from a specific version of the registered model.

        Parameters
        ----------
        version : int
            The version of the registered model.
        artifact_path : str
            The path to the artifact within the model version.

        Returns
        -------
        str
            The local path to the downloaded artifact.
        """
        try:
            model_uri = f"models:/{self.model_name}/{version}"
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=f"{model_uri}/{artifact_path}")
            self._logger.info(f"Downloaded artifact '{artifact_path}' from model version {version} to '{local_path}'.")
            return local_path
        except Exception as e:
            log_and_raise_error(
                self._logger, f"Error downloading artifact '{artifact_path}' from model version {version}: {e}"
            )

    def load_artifact_from_run(self, run_name: str, artifact_path: str, experiment_name: str = None) -> str:
        """
        Download an artifact from a run by run name.

        Parameters
        ----------
        run_name : str
            The name of the run.
        artifact_path : str
            The path to the artifact within the run.
        experiment_name : str, optional
            The name of the experiment. If None, uses the model name.

        Returns
        -------
        str
            The local path to the downloaded artifact.
        """
        if experiment_name is None:
            experiment_name = self.model_name
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if not experiment:
                log_and_raise_error(self._logger, f"Experiment '{experiment_name}' not found.")
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'"
            )
            if not runs:
                log_and_raise_error(self._logger, f"Run '{run_name}' not found in experiment '{experiment_name}'.")
            run_id = runs[0].info.run_id
            run_uri = f"runs:/{run_id}/{artifact_path}"
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=run_uri)
            self._logger.info(f"Downloaded artifact '{artifact_path}' from run '{run_name}' to '{local_path}'.")
            return local_path
        except Exception as e:
            log_and_raise_error(
                self._logger, f"Error downloading artifact '{artifact_path}' from run '{run_name}': {e}"
            )

    def get_run_data(self, version: int | str | None = None, alias: str | None = None):
        """
        Get run data from a model version using either version number or alias.

        Parameters
        ----------
        version : int, str, or None
            The version number of the model. Can be an integer, string, or "latest".
        alias : str, optional
            The alias of the model version (e.g., "production", "staging").

        Returns
        -------
        mlflow.entities.Run
            The MLflow run object containing all run data (params, metrics, tags, etc.).

        Raises
        ------
        ValueError
            If neither version nor alias is provided, or if the version/alias is not found.
        """
        if version is None and alias is None:
            log_and_raise_error(self._logger, "Either 'version' or 'alias' must be provided to get run data.")

        try:
            # Handle version resolution
            if alias is not None:
                model_version = self.client.get_model_version_by_alias(name=self.model_name, alias=alias)
            elif version == "latest":
                latest_version = self.get_latest_model_version()
                if latest_version is None:
                    log_and_raise_error(self._logger, f"No versions found for model '{self.model_name}'.")
                model_version = self.client.get_model_version(name=self.model_name, version=str(latest_version))
            else:
                model_version = self.client.get_model_version(name=self.model_name, version=str(version))

            run = self.client.get_run(model_version.run_id)

            self._logger.info(
                f"Successfully retrieved run data for model '{self.model_name}' version {model_version.version} "
                f"(run_id: {model_version.run_id})."
            )
            return run

        except Exception as e:
            log_and_raise_error(
                self._logger,
                f"Error retrieving run data for model '{self.model_name}' (version: {version}, alias: {alias}): {e}",
            )
