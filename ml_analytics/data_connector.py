"""
Generic utility functions for data processing and database connection.
"""
import os
import logging
import redshift_connector
import boto3

# Setup logging and warning filters
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class DataConnector:
    def __init__(self, *, database=None, user=None, password=None, host=None, port=None,
                 s3_bucket=None):
        """
        Initialize a DataConnector instance.

        Parameters
        ----------
        database : str, optional
            The name of the Redshift database to connect to. Defaults to None.
        user : str, optional
            The username to use when connecting to the database. Defaults to None.
        password : str, optional
            The password to use when connecting to the database. Defaults to None.
        host : str, optional
            The hostname or IP address of the Redshift instance. Defaults to None.
        port : str, optional
            The port number to use when connecting to the database. Defaults to None.

        If any of the above parameters are not provided, the corresponding environment
        variables will be used (REDSHIFT_DATABASE, REDSHIFT_USER, REDSHIFT_PASSWORD,
        REDSHIFT_HOST, REDSHIFT_PORT).

        Attributes
        ----------
        database : str
            The name of the Redshift database being connected to.
        user : str
            The username being used to connect to the database.
        password : str
            The password being used to connect to the database.
        host : str
            The hostname or IP address of the Redshift instance.
        port : str
            The port number being used to connect to the database.
        connection : redshift_connector.Connection
            The active connection to the Redshift database.
        cursor : redshift_connector.Cursor
            A cursor object that can be used to execute SQL queries.
        logger : logging.Logger
            A logger object that can be used to log events.

        Notes
        -----
        The connection is established automatically when the DataConnector instance is
        created. The connection is set to autocommit mode, so any changes made to the
        database will be committed immediately. The cursor is also created automatically
        and can be used to execute SQL queries.
        """

        if database is None:
            database = os.environ.get('REDSHIFT_DATABASE')
        if user is None:
            user = os.environ.get('REDSHIFT_USER')
        if password is None:
            password = os.environ.get('REDSHIFT_PASSWORD')
        if host is None:
            host = os.environ.get('REDSHIFT_HOST')
        if port is None:
            port = os.environ.get('REDSHIFT_PORT', '5439')

        self.__database = database
        self.__user = user
        self.__password = password
        self.__host = host
        self.__port = port

        self.connection = redshift_connector.connect(
            host=self.__host,
            database=self.__database,
            user=self.__user,
            password=self.__password,
            port=self.__port
        )
        self.connection.autocommit = True
        self.cursor = self.connection.cursor()

        self.logger = logging.getLogger("__name__")

        if s3_bucket is None:
            self.s3_bucket = 'docplanner-mlp-projects'

        self.logger.info('DataConnector initialized')

    def sql(self, query):
        """
        Execute a SQL query against the Redshift database and return the result as a pandas DataFrame.

        Parameters
        ----------
        query : str
            The SQL query to be executed.

        Returns
        -------
        pandas.DataFrame
            The result of the query as a pandas DataFrame.

        Raises
        ------
        Exception
            If any error occurs during the execution of the query.
        """
        try:
            self.cursor.execute(query)
            self.logger.info('Data fetched successfully')
            return  self.cursor.fetch_dataframe()
        except Exception as e:
            self.logger.error(f'Error fetching data: {e}')
            raise

    def save_table_redshift(self, df, table_name=None, schema=None, mode="append"):
        """
        Checks if the specified table exists in Redshift, creates it if necessary,
        then writes the DataFrame to that table.

        Parameters
        ----------
        df : pandas.DataFrame
            The data to be written to Redshift.
        table_name : str
            The name of the table in Redshift.
        schema : str
            The schema containing the table.
        mode : str, optional
            The write mode ("append" or "overwrite"). Defaults to "append".

        Raises
        ------
        Exception
            If any error occurs during the table creation or data insertion process.
        """
        try:
            # Build quoted schema.table for Redshift
            fully_qualified_table = f"{schema}.{table_name}"

            # 1) Check if table already exists
            check_sql = f"""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = '{schema}'
            AND table_name = '{table_name}'
            """
            self.cursor.execute(check_sql)
            table_exists = self.cursor.fetchall()[0][0]  # 1 if exists, 0 if not

            if table_exists == 1 and mode.lower() == "overwrite":
                # Drop the existing table
                drop_sql = f"DROP TABLE {fully_qualified_table}"
                self.cursor.execute(drop_sql)
                self.logger.info(f"Dropped existing table {fully_qualified_table} for overwrite.")
                table_exists = 0  # So the code below will re-create the table

            # Create the table if it does not exist
            if table_exists == 0:
                columns_definitions = []
                for col, dtype in df.dtypes.items():
                    if "int" in str(dtype).lower():
                        col_type = "INTEGER"
                    elif "float" in str(dtype).lower():
                        col_type = "DOUBLE PRECISION"
                    else:
                        col_type = "VARCHAR(255)"
                    columns_definitions.append(f'"{col}" {col_type}')

                create_sql = f"""
                CREATE TABLE {fully_qualified_table} (
                    {", ".join(columns_definitions)}
                )
                """
                self.cursor.execute(create_sql)
                self.logger.info(f"Creating table {fully_qualified_table}")

            # Now write the DataFrame to the (re)created/ existing table
            self.cursor.write_dataframe(df, table=fully_qualified_table)
            self.logger.info(f"Data successfully written to {fully_qualified_table}")

        except Exception as e:
            self.logger.error(f"Error saving table: {e}")
            raise

    def save_dataframe_to_s3(self, df, bucket = None, directory = None, filename=None, file_format='csv'):
            """
            Save a pandas DataFrame to an S3 bucket in a new directory as a CSV or Parquet file.

            Parameters
            ----------
            df : pandas.DataFrame
                The DataFrame to save.
            bucket : str
                The S3 bucket name.
            directory : str
                The directory path where the file will be saved (it should end with a '/').
            filename : str
                The name of the file (without extension).
            file_format : str, optional
                File format to save ('csv' or 'parquet'). Default is 'csv'.

            Returns
            -------
            None
            """
            # Ensure the directory ends with a slash
            if bucket is None: 
                bucket = self.s3_bucket

            if filename is None:
                self.logger.error("No filename provided")
            if directory is None:
                directory = ''
            if not directory.endswith('/'):
                directory += '/'

            # Full path to the S3 location
            s3_path = f"s3://{bucket}/{directory}{filename}.{file_format}"

            self.initialize_s3_client()

            # Write DataFrame to S3
            try:
                if file_format == 'csv':
                    df.to_csv(s3_path, index=False)
                elif file_format == 'parquet':
                    df.to_parquet(s3_path, index=False)
                else:
                    raise ValueError("Unsupported file format. Choose 'csv' or 'parquet'.")
                self.logger.info(f"Saved file to {s3_path}")
            except Exception as e:
                self.logger.error(f"Error saving DataFrame to S3: {e}")
                raise

    def list_s3_files(self, bucket=None, prefix=''):
        """
        List files in the specified S3 bucket. Optionally filter by a prefix.

        Parameters
        ----------
        bucket: str
            The S3 bucket name.
        prefix: str, optional
            The prefix to filter files. Defaults to ''.

        Returns
        -------
        list
            A list of file keys in the specified bucket (and prefix).
        """

        self.initialize_s3_client()

        if bucket is None:
            bucket = self.s3_bucket
        try:
            response = self.s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if 'Contents' in response:
                files = [obj['Key'] for obj in response['Contents']]
            else:
                files = []
            self.logger.info(f"Found {len(files)} files in bucket {bucket} with prefix '{prefix}'")
            return files
        except Exception as e:
            self.logger.error(f"Error listing files from S3: {e}")
            raise

    def delete_s3_file(self, bucket=None, key=None):

        self.initialize_s3_client()

        if bucket is None:
            bucket = self.s3_bucket
        try:
            self.s3.delete_object(Bucket=bucket, Key=key)
            self.logger.info(f"Deleted file {key} from bucket {bucket}")
        except Exception as e:
            self.logger.error(f"Error deleting file from S3: {e}")
            raise

    def close_reshdift_connection(self):
        try:
            if self.connection:
                self.connection.close()
                self.logger.info('Connection closed')
        except Exception as e:
            self.logger.error(f'Error closing connection: {e}')

    def initialize_s3_client(self):
        try:
            # Attempt to create an S3 client
            self.s3 = boto3.Session().client('s3')
            self.logger.info("S3 client initialized")
        except Exception as e:
            self.logger.error(f"Error initializing S3 client: {e}")
            raise

    def create_local_file_path(self, relative_path, base_dir="."):
        """
        Generate a local file path by combining a base directory and a relative path.

        Parameters
        ----------
        relative_path : str
            The relative path to be appended to the base directory.
        base_dir : str, optional
            The base directory to use. Defaults to the current directory.

        Returns
        -------
        str
            The full file path as a string.

        Notes
        -----
        If the special variable '__file__' is available, this function will use its
        directory location as the base directory, otherwise it will use the current
        working directory.
        """
        if '__file__' in globals():
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            base_dir = os.path.abspath(os.path.join(os.getcwd(), ''))
        file_path = os.path.join(base_dir, relative_path)
        return file_path

    def create_redshift_table_from_s3(self, s3_file_path, redshift_table):
        """Copy data from S3 into a Redshift table using Parquet"""

        self.initialize_s3_client()
        copy_query = f"""
        COPY {redshift_table}
        FROM 's3://{self.s3_bucket}/{s3_file_path}'
        FORMAT AS PARQUET;
        """

        with self.connection.cursor() as cursor:
            cursor.execute(copy_query)
        self.connection.commit()
        self.logger.info('Table created!')
