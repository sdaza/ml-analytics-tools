"""
Slack connector for sending messages, images, and files to Slack channels and users.

This module provides a simple interface to interact with Slack's API for common tasks
such as sending messages, uploading files, and managing channel communications.
"""

from pathlib import Path

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from .utils import get_credential_value, get_logger, log_and_raise_error


class SlackConnector:
    """
    A connector class for interacting with Slack API.

    This class provides methods to send messages, upload files, and interact
    with Slack channels using a bot token.
    """

    def __init__(
        self,
        token: str = None,
        token_path: str | Path = None,
        log_level: str = "INFO",
        return_response: bool = False,
        scope: str = "ai-data-products",
    ):
        """
        Initialize the Slack connector.

        Parameters
        ----------
        token : str, optional
            Slack Bot Token (starts with 'xoxb-').
            If not provided, will check SLACK_BOT_TOKEN environment variable or mounted secrets,
            then token_path, then default locations.
        token_path : str | Path, optional
            Path to file containing the Slack token.
            If not provided, will look for 'slack_token.txt' in current directory.
        log_level : str, optional
            Logging level. Default is "INFO".
        return_response : bool, optional
            If True, methods return full API response. If False, returns None for cleaner output.
            Default is False.
        scope : str, optional
            Scope for SecretProvider mounted secrets (e.g., '/mnt/{scope}/SLACK_BOT_TOKEN').
            Default is "ai-data-products". Used in production containerized environments.

        Examples
        --------
        >>> # Using token directly
        >>> slack = SlackConnector(token="xoxb-your-token")
        >>>
        >>> # Using token from .env file (recommended)
        >>> # Add to .env: SLACK_BOT_TOKEN=xoxb-your-token
        >>> slack = SlackConnector()
        >>>
        >>> # Using token file
        >>> slack = SlackConnector(token_path="slack_token.txt")
        >>>
        >>> # Using production mounted secrets with custom scope
        >>> slack = SlackConnector(scope="custom-scope")
        """
        self._logger = get_logger("SlackConnector")
        self._logger.setLevel(log_level)
        self.return_response = return_response

        if token is None and token_path is None:
            # Try to get token from environment variable or mounted secret
            try:
                token = get_credential_value("SLACK_BOT_TOKEN", scope=scope)
                self._logger.debug("Using Slack token from environment or mounted secret")
            except Exception:
                # Fall back to checking default file location
                default_path = Path.cwd() / "slack_token.txt"
                if default_path.exists():
                    token_path = default_path
                else:
                    log_and_raise_error(
                        self._logger,
                        "Slack token not found. Please provide one of:\n"
                        "  1. Set SLACK_BOT_TOKEN in .env file\n"
                        "  2. Pass 'token' parameter\n"
                        "  3. Pass 'token_path' parameter\n"
                        "  4. Create 'slack_token.txt' in current directory\n"
                        "  5. Mount secret at /mnt/{scope}/SLACK_BOT_TOKEN in production",
                    )

        if token_path is not None:
            token_path = Path(token_path)
            if not token_path.exists():
                log_and_raise_error(
                    self._logger,
                    f"Token file not found at: {token_path}",
                )
            token = token_path.read_text().strip()

        try:
            self.client = WebClient(token=token)
            response = self.client.auth_test()
            self.bot_user_id = response["user_id"]
            self.team_name = response.get("team", "Unknown")
            self.bot_name = response.get("user", "Bot")
            self._logger.info(f"Successfully connected to Slack workspace: {self.team_name} (bot: {self.bot_name})")
        except SlackApiError as e:
            log_and_raise_error(
                self._logger,
                f"Failed to connect to Slack: {e.response['error']}",
            )
        except Exception as e:
            log_and_raise_error(
                self._logger,
                f"Error initializing Slack client: {e}",
            )

    def send_message(
        self,
        channel: str,
        text: str,
        blocks: list[dict] = None,
        thread_ts: str = None,
        unfurl_links: bool = True,
        unfurl_media: bool = True,
    ) -> dict:
        """
        Send a message to a Slack channel or user.

        Parameters
        ----------
        channel : str
            Channel name (e.g., '#general', 'general') or user ID.
        text : str
            Message text (also used as fallback for notifications).
        blocks : list[dict], optional
            Slack Block Kit blocks for rich formatting.
        thread_ts : str, optional
            Thread timestamp to reply in a thread.
        unfurl_links : bool, optional
            Whether to unfurl links. Default is True.
        unfurl_media : bool, optional
            Whether to unfurl media. Default is True.

        Returns
        -------
        dict
            API response containing message details.

        Examples
        --------
        >>> slack = SlackConnector(token="xoxb-your-token")
        >>>
        >>> # Simple message
        >>> slack.send_message("#general", "Hello from Python!")
        >>>
        >>> # Rich formatting with blocks
        >>> blocks = [
        ...     {
        ...         "type": "section",
        ...         "text": {"type": "mrkdwn", "text": "*Important Alert*"}
        ...     }
        ... ]
        >>> slack.send_message("#alerts", "Alert", blocks=blocks)
        >>>
        >>> # Reply in a thread
        >>> response = slack.send_message("#general", "Main message")
        >>> slack.send_message("#general", "Reply", thread_ts=response["ts"])
        """
        # Resolve channel (accepts name or ID)
        channel_id = self.get_or_resolve_channel_id(channel)

        try:
            response = self.client.chat_postMessage(
                channel=channel_id,
                text=text,
                blocks=blocks,
                thread_ts=thread_ts,
                unfurl_links=unfurl_links,
                unfurl_media=unfurl_media,
            )
            self._logger.info("Message sent successfully")
            return response.data if self.return_response else None

        except SlackApiError as e:
            log_and_raise_error(
                self._logger,
                f"Error sending message to Slack: {e.response['error']}",
            )
        except Exception as e:
            log_and_raise_error(
                self._logger,
                f"Error sending message: {e}",
            )

    def send_message_with_image(
        self,
        channel: str,
        text: str,
        image_url: str,
        image_alt: str = "image",
        title: str = None,
        thread_ts: str = None,
    ) -> dict:
        """
        Send a message with an inline image using Block Kit.

        Note: The image URL must be publicly accessible.

        Parameters
        ----------
        channel : str
            Channel name or ID.
        text : str
            Message text.
        image_url : str
            Publicly accessible URL to the image.
        image_alt : str, optional
            Alt text for the image. Default is "image".
        title : str, optional
            Optional title above the image.
        thread_ts : str, optional
            Thread timestamp to reply in a thread.

        Returns
        -------
        dict
            API response containing message details.

        Examples
        --------
        >>> slack = SlackConnector(token="xoxb-your-token")
        >>> slack.send_message_with_image(
        ...     "#general",
        ...     "Check out our results!",
        ...     "https://example.com/chart.png",
        ...     image_alt="Performance chart",
        ...     title="Model Performance"
        ... )
        """
        blocks = []

        if title:
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*{title}*"}})

        blocks.extend(
            [
                {"type": "section", "text": {"type": "mrkdwn", "text": text}},
                {"type": "image", "image_url": image_url, "alt_text": image_alt},
            ]
        )

        result = self.send_message(channel, text, blocks=blocks, thread_ts=thread_ts)
        self._logger.info("Message with image sent successfully")
        return result

    def send_file(
        self,
        channels: str | list[str],
        file_path: str | Path = None,
        content: str | bytes = None,
        filename: str = None,
        title: str = None,
        initial_comment: str = None,
        thread_ts: str = None,
    ) -> dict:
        """
        Upload a file to Slack channel(s).

        Parameters
        ----------
        channels : str | list[str]
            Channel name(s) to upload to (e.g., '#general', 'general').
        file_path : str | Path, optional
            Path to file to upload.
        content : str | bytes, optional
            File content (alternative to file_path).
        filename : str, optional
            Filename to display in Slack (required if using content).
        title : str, optional
            Title of the file.
        initial_comment : str, optional
            Comment to add with the file.
        thread_ts : str, optional
            Thread timestamp to upload file to a thread.

        Returns
        -------
        dict
            API response containing file details.

        Examples
        --------
        >>> slack = SlackConnector(token="xoxb-your-token")
        >>>
        >>> # Upload a file
        >>> slack.send_file(
        ...     channels="#data-team",
        ...     file_path="report.csv",
        ...     title="Daily Report",
        ...     initial_comment="Here's today's report"
        ... )
        >>>
        >>> # Upload to multiple channels
        >>> slack.send_file(
        ...     channels=["#team-a", "#team-b"],
        ...     file_path="metrics.png",
        ...     title="Metrics Dashboard"
        ... )
        >>>
        >>> # Upload from content
        >>> import pandas as pd
        >>> df = pd.DataFrame({'col1': [1, 2, 3]})
        >>> csv_content = df.to_csv(index=False)
        >>> slack.send_file(
        ...     channels="#data-team",
        ...     content=csv_content,
        ...     filename="data.csv",
        ...     title="Generated Data"
        ... )
        """

        if isinstance(channels, str):
            channels = [channels]

        channel_ids = [self.get_or_resolve_channel_id(ch) for ch in channels]

        try:
            if file_path is not None:
                response = self.client.files_upload_v2(
                    channel=channel_ids[0] if len(channel_ids) == 1 else None,
                    channels=channel_ids if len(channel_ids) > 1 else None,
                    file=str(file_path),
                    title=title,
                    initial_comment=initial_comment,
                    thread_ts=thread_ts,
                )
            elif content is not None:
                if filename is None:
                    log_and_raise_error(
                        self._logger,
                        "Parameter 'filename' is required when using 'content'",
                    )
                response = self.client.files_upload_v2(
                    channel=channel_ids[0] if len(channel_ids) == 1 else None,
                    channels=channel_ids if len(channel_ids) > 1 else None,
                    content=content,
                    filename=filename,
                    title=title,
                    initial_comment=initial_comment,
                    thread_ts=thread_ts,
                )
            else:
                log_and_raise_error(
                    self._logger,
                    "Either 'file_path' or 'content' must be provided",
                )

            channel_list = ", ".join([ch.lstrip("#") for ch in channels])
            self._logger.info(f"Successfully uploaded file to {channel_list}")
            return response.data if self.return_response else None

        except SlackApiError as e:
            log_and_raise_error(
                self._logger,
                f"Error uploading file to Slack: {e.response['error']}",
            )
        except Exception as e:
            log_and_raise_error(
                self._logger,
                f"Error uploading file: {e}",
            )

    def list_channels(
        self,
        types: str = "public_channel",
        limit: int = 1000,
        exclude_archived: bool = True,
    ) -> list[dict]:
        """
        List channels in the workspace.

        Parameters
        ----------
        types : str, optional
            Comma-separated channel types to list.
            Options: 'public_channel', 'private_channel', 'im', 'mpim'.
            Default is 'public_channel'.
        limit : int, optional
            Maximum number of channels to return. Default is 1000.
        exclude_archived : bool, optional
            Whether to exclude archived channels. Default is True.

        Returns
        -------
        list[dict]
            List of channel objects with keys: id, name, is_channel, is_private, etc.

        Examples
        --------
        >>> slack = SlackConnector(token="xoxb-your-token")
        >>>
        >>> # List public channels
        >>> channels = slack.list_channels()
        >>> for ch in channels:
        ...     print(f"#{ch['name']}")
        >>>
        >>> # List private channels (requires appropriate scopes)
        >>> private = slack.list_channels(types="private_channel")
        """
        try:
            response = self.client.conversations_list(
                types=types,
                limit=limit,
                exclude_archived=exclude_archived,
            )

            return response["channels"]

        except SlackApiError as e:
            log_and_raise_error(
                self._logger,
                f"Error listing channels: {e.response['error']}",
            )
        except Exception as e:
            log_and_raise_error(
                self._logger,
                f"Error listing channels: {e}",
            )

    def get_channel_id(self, channel_name: str) -> str | None:
        """
        Get channel ID from channel name.

        Parameters
        ----------
        channel_name : str
            Channel name (with or without #).

        Returns
        -------
        str | None
            Channel ID if found, None otherwise.

        Examples
        --------
        >>> slack = SlackConnector(token="xoxb-your-token")
        >>> channel_id = slack.get_channel_id("#general")
        >>> print(channel_id)  # C01234567
        """
        # Clean channel name
        if channel_name.startswith("#"):
            channel_name = channel_name[1:]

        try:
            # Try private channels first
            try:
                private_channels = self.list_channels(types="private_channel")
                for channel in private_channels:
                    if channel["name"] == channel_name:
                        return channel["id"]
            except Exception:
                pass

            channels = self.list_channels(types="public_channel")
            for channel in channels:
                if channel["name"] == channel_name:
                    return channel["id"]

            return None

        except Exception as e:
            self._logger.error(f"Error getting channel ID: {e}")
            return None

    def get_or_resolve_channel_id(self, channel: str) -> str:
        """
        Get channel ID by name. If not found in conversations.list,
        send a test message to resolve the ID (works for Slack Connect / multi-workspace).
        """

        channel_clean = channel.lstrip("#")
        channel_id = self.get_channel_id(channel_clean)
        if channel_id:
            return channel_id

        try:
            resp = self.client.chat_postMessage(channel=channel_clean, text=".")
            resolved_id = resp["channel"]

            try:
                self.client.chat_delete(channel=resolved_id, ts=resp["ts"])
            except Exception:
                pass

            return resolved_id
        except SlackApiError as e:
            log_and_raise_error(self._logger, f"Could not resolve channel '{channel_clean}': {e.response['error']}")

    def delete_message(self, channel: str, ts: str) -> dict:
        """
        Delete a message.

        Parameters
        ----------
        channel : str
            Channel name or ID where the message was posted.
        ts : str
            Timestamp of the message to delete.

        Returns
        -------
        dict
            API response.

        Examples
        --------
        >>> slack = SlackConnector(token="xoxb-your-token")
        >>> response = slack.send_message("#general", "Temporary message")
        >>> slack.delete_message("#general", response["ts"])
        """

        channel_id = self.get_or_resolve_channel_id(channel)

        try:
            response = self.client.chat_delete(
                channel=channel_id,
                ts=ts,
            )

            self._logger.info(f"Message deleted in channel #{channel}, TS {ts}")
            return response.data

        except SlackApiError as e:
            log_and_raise_error(
                self._logger,
                f"Error deleting message: {e.response['error']}",
            )
        except Exception as e:
            log_and_raise_error(
                self._logger,
                f"Error deleting message: {e}",
            )

    def add_reaction(self, channel: str, ts: str, emoji: str) -> dict:
        """
        Add an emoji reaction to a message.

        Parameters
        ----------
        channel : str
            Channel name or ID where the message was posted.
        ts : str
            Timestamp of the message.
        emoji : str
            Emoji name (without colons, e.g., 'thumbsup', 'white_check_mark').

        Returns
        -------
        dict
            API response.

        Examples
        --------
        >>> slack = SlackConnector(token="xoxb-your-token")
        >>> response = slack.send_message("#general", "Great work!")
        >>> slack.add_reaction("#general", response["ts"], "thumbsup")
        """

        channel_id = self.get_or_resolve_channel_id(channel)

        emoji = emoji.strip(":")

        try:
            response = self.client.reactions_add(
                channel=channel_id,
                timestamp=ts,
                name=emoji,
            )

            self._logger.info(f"Added reaction :{emoji}: to message in #{channel}")
            return response.data

        except SlackApiError as e:
            log_and_raise_error(
                self._logger,
                f"Error adding reaction: {e.response['error']}",
            )
        except Exception as e:
            log_and_raise_error(
                self._logger,
                f"Error adding reaction: {e}",
            )

    def delete_file(self, file_id: str) -> dict:
        """
        Delete a file from Slack.

        Parameters
        ----------
        file_id : str
            The ID of the file to delete (e.g., 'F01234567').

        Returns
        -------
        dict
            API response.

        Examples
        --------
        >>> slack = SlackConnector(token="xoxb-your-token")
        >>> # Upload a file and get its ID
        >>> response = slack.send_file("#general", file_path="test.txt")
        >>> file_id = response['file']['id']
        >>> # Later, delete the file
        >>> slack.delete_file(file_id)
        """
        try:
            response = self.client.files_delete(
                file=file_id,
            )

            self._logger.info("File deleted successfully")
            return response.data

        except SlackApiError as e:
            log_and_raise_error(
                self._logger,
                f"Error deleting file: {e.response['error']}",
            )
        except Exception as e:
            log_and_raise_error(
                self._logger,
                f"Error deleting file: {e}",
            )
