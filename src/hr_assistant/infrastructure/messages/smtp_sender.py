import asyncio
import smtplib
import ssl
from email.message import EmailMessage

from hr_assistant.core.config import Settings


class SmtpMessageSender:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def configured(self) -> bool:
        return self.settings.smtp_configured

    async def send(self, recipient: str, subject: str, body: str) -> None:
        if not self.configured:
            raise RuntimeError("SMTP is not configured.")
        await asyncio.to_thread(self._send_sync, recipient, subject, body)

    def _send_sync(self, recipient: str, subject: str, body: str) -> None:
        message = EmailMessage()
        message["From"] = self.settings.smtp_from
        message["To"] = recipient
        message["Subject"] = subject
        message.set_content(body)

        password = (
            self.settings.smtp_password.get_secret_value()
            if self.settings.smtp_password
            else None
        )
        username = self.settings.smtp_username
        timeout_seconds = 20
        if self.settings.smtp_use_ssl:
            with smtplib.SMTP_SSL(
                self.settings.smtp_host,
                self.settings.smtp_port,
                timeout=timeout_seconds,
                context=ssl.create_default_context(),
            ) as client:
                self._authenticate(client, username, password)
                client.send_message(message)
            return

        with smtplib.SMTP(
            self.settings.smtp_host,
            self.settings.smtp_port,
            timeout=timeout_seconds,
        ) as client:
            client.ehlo()
            if self.settings.smtp_starttls:
                client.starttls(context=ssl.create_default_context())
                client.ehlo()
            self._authenticate(client, username, password)
            client.send_message(message)

    @staticmethod
    def _authenticate(client, username: str | None, password: str | None) -> None:
        if username:
            client.login(username, password or "")

