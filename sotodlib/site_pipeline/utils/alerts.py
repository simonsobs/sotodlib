"""Alert and notification utilities for site_pipeline."""

import datetime as dt
import time
import requests


def send_alert(webhook, alertname='', tag='', error='', timestamp=None):  # make_book
    """Send an alert to the given webhook.

    Parameters
    ----------
    webhook : str
        Webhook URL to send alert.
    alertname : str, optional
        Name of alert.
    tag : str, optional
        Tag to differentiate types of alerts.
    error : str, optional
        Error message to send as the main body of the alert.
    timestamp : str, optional
        The timestamp to attach to alert.
    """
    if not webhook:
        return "No webhook provided"

    if isinstance(timestamp, type(None)):
        timestamp = round(dt.datetime.now(tz=dt.timezone.utc).timestamp())
    elif isinstance(timestamp, str):
        ts = dt.datetime.fromisoformat(timestamp)
        if ts.tzinfo is None:
            timestamp = round(ts.replace(tzinfo=dt.timezone.utc).timestamp())
        else:
            timestamp = round(ts.timestamp())
    elif isinstance(timestamp, (int, float)):
        timestamp = round(timestamp)
    elif isinstance(timestamp, dt.datetime):
        if timestamp.tzinfo is None:
            timestamp = round(timestamp.replace(tzinfo=dt.timezone.utc).timestamp())
        else:
            timestamp = round(timestamp.timestamp())
    else:
        return f"Could not convert timestamp type {type(timestamp)}"

    try:
        # Directly to Slack channel
        if 'slack.com' in webhook:
            from slack_sdk.webhook import WebhookClient

            webhook = WebhookClient(webhook)
            response = webhook.send(
                text="Pipeline alert",
                attachments=[
                    {
                        "mrkdwn_in": ["text"],
                        "color": "danger",
                        "title": f"[FIRING] {tag} ({alertname})",
                        "text": f"error: {error}\ntimestamp: {timestamp}",
                        "footer": "Pipeline",
                        "ts": f"{time.time()}"
                    }
                ]
            )
            if response.status_code != 200:
                raise Exception("Issue with Slack webhook.")

        # Custom webhook (Campana)
        else:
            data = {
                    'status': 'firing',
                    'alerts': [
                        {
                            'status': 'firing',
                            'labels': {'alertname': alertname, 'tag': tag},
                            'annotations': {'error': error, 'groups': 'pipeline', 'timestamp': timestamp},
                        }
                    ],
            }
            response = requests.post(webhook, json=data)
            if response.status_code != requests.codes.ok:
                raise Exception("Issue with custom webhook.")

        return "Alert sent"
    except Exception as e:
        return f"Failed to send alert: {e}"
