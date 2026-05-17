

import os
import platform
import random
import sys
import threading
import time
from pathlib import Path

import requests

from ultralytics.utils import (
    ENVIRONMENT,
    LOGGER,
    ONLINE,
    RANK,
    SETTINGS,
    TESTS_RUNNING,
    TQDM,
    TryExcept,
    __version__,
    colorstr,
    get_git_origin_url,
    is_colab,
    is_git_dir,
    is_pip_package,
)
from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

HUB_API_ROOT = os.environ.get("ULTRALYTICS_HUB_API", "https://api.ultralytics.com")
HUB_WEB_ROOT = os.environ.get("ULTRALYTICS_HUB_WEB", "https://hub.ultralytics.com")

PREFIX = colorstr("Ultralytics HUB: ")
HELP_MSG = "If this issue persists please visit https://github.com/ultralytics/hub/issues for assistance."


def request_with_credentials(url: str) -> any:
    """
    Make an AJAX request with cookies attached in a Google Colab environment.

    Args:
        url (str): The URL to make the request to.

    Returns:
        (any): The response data from the AJAX request.

    Raises:
        OSError: If the function is not run in a Google Colab environment.
    """
    if not is_colab():
        raise OSError("request_with_credentials() must run in a Colab environment")
    from google.colab import output
    from IPython import display

    display.display(
        display.Javascript(
            """
            window._hub_tmp = new Promise((resolve, reject) => {
                const timeout = setTimeout(() => reject("Failed authenticating existing browser session"), 5000)
                fetch("%s", {
                    method: 'POST',
                    credentials: 'include'
                })
                    .then((response) => resolve(response.json()))
                    .then((json) => {
                    clearTimeout(timeout);
                    }).catch((err) => {
                    clearTimeout(timeout);
                    reject(err);
                });
            });
            """
            % url
        )
    )
    return output.eval_js("_hub_tmp")


def requests_with_progress(method, url, **kwargs):
    """
    Make an HTTP request using the specified method and URL, with an optional progress bar.

    Args:
        method (str): The HTTP method to use (e.g. 'GET', 'POST').
        url (str): The URL to send the request to.
        **kwargs (dict): Additional keyword arguments to pass to the underlying `requests.request` function.

    Returns:
        (requests.Response): The response object from the HTTP request.

    Note:
        - If 'progress' is set to True, the progress bar will display the download progress for responses with a known
        content length.
        - If 'progress' is a number then progress bar will display assuming content length = progress.
    """
    progress = kwargs.pop("progress", False)
    if not progress:
        return requests.request(method, url, **kwargs)
    response = requests.request(method, url, stream=True, **kwargs)
    total = int(response.headers.get("content-length", 0) if isinstance(progress, bool) else progress)
    try:
        pbar = TQDM(total=total, unit="B", unit_scale=True, unit_divisor=1024)
        for data in response.iter_content(chunk_size=1024):
            pbar.update(len(data))
        pbar.close()
    except requests.exceptions.ChunkedEncodingError:
        response.close()
    return response


def smart_request(method, url, retry=3, timeout=30, thread=True, code=-1, verbose=True, progress=False, **kwargs):
    """
    Makes an HTTP request using the 'requests' library, with exponential backoff retries up to a specified timeout.

    Args:
        method (str): The HTTP method to use for the request. Choices are 'post' and 'get'.
        url (str): The URL to make the request to.
        retry (int, optional): Number of retries to attempt before giving up. Default is 3.
        timeout (int, optional): Timeout in seconds after which the function will give up retrying. Default is 30.
        thread (bool, optional): Whether to execute the request in a separate daemon thread. Default is True.
        code (int, optional): An identifier for the request, used for logging purposes. Default is -1.
        verbose (bool, optional): A flag to determine whether to print out to console or not. Default is True.
        progress (bool, optional): Whether to show a progress bar during the request. Default is False.
        **kwargs (dict): Keyword arguments to be passed to the requests function specified in method.

    Returns:
        (requests.Response): The HTTP response object. If the request is executed in a separate thread, returns None.
    """
    retry_codes = (408, 500)

    @TryExcept(verbose=verbose)
    def func(func_method, func_url, **func_kwargs):
        """Make HTTP requests with retries and timeouts, with optional progress tracking."""
        r = None
        t0 = time.time()
        for i in range(retry + 1):
            if (time.time() - t0) > timeout:
                break
            r = requests_with_progress(func_method, func_url, **func_kwargs)
            if r.status_code < 300:
                break
            try:
                m = r.json().get("message", "No JSON message.")
            except AttributeError:
                m = "Unable to read JSON."
            if i == 0:
                if r.status_code in retry_codes:
                    m += f" Retrying {retry}x for {timeout}s." if retry else ""
                elif r.status_code == 429:
                    h = r.headers
                    m = (
                        f"Rate limit reached ({h['X-RateLimit-Remaining']}/{h['X-RateLimit-Limit']}). "
                        f"Please retry after {h['Retry-After']}s."
                    )
                if verbose:
                    LOGGER.warning(f"{PREFIX}{m} {HELP_MSG} ({r.status_code})")
                if r.status_code not in retry_codes:
                    return r
            time.sleep(2**i)
        return r

    args = method, url
    kwargs["progress"] = progress
    if thread:
        threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True).start()
    else:
        return func(*args, **kwargs)


class Events:
    """
    A class for collecting anonymous event analytics. Event analytics are enabled when sync=True in settings and
    disabled when sync=False. Run 'yolo settings' to see and update settings YAML file.

    Attributes:
        url (str): The URL to send anonymous events.
        rate_limit (float): The rate limit in seconds for sending events.
        metadata (dict): A dictionary containing metadata about the environment.
        enabled (bool): A flag to enable or disable Events based on certain conditions.
    """

    url = "https://www.google-analytics.com/mp/collect?measurement_id=G-X8NCJYTQXM&api_secret=QLQrATrNSwGRFRLE-cbHJw"

    def __init__(self):
        """Initializes the Events object with default values for events, rate_limit, and metadata."""
        self.events = []
        self.rate_limit = 60.0
        self.t = 0.0
        self.metadata = {
            "cli": Path(sys.argv[0]).name == "yolo",
            "install": "git" if is_git_dir() else "pip" if is_pip_package() else "other",
            "python": ".".join(platform.python_version_tuple()[:2]),
            "version": __version__,
            "env": ENVIRONMENT,
            "session_id": round(random.random() * 1e15),
            "engagement_time_msec": 1000,
        }
        self.enabled = (
            SETTINGS["sync"]
            and RANK in (-1, 0)
            and not TESTS_RUNNING
            and ONLINE
            and (is_pip_package() or get_git_origin_url() == "https://github.com/ultralytics/ultralytics.git")
        )

    def __call__(self, cfg):
        """
        Attempts to add a new event to the events list and send events if the rate limit is reached.

        Args:
            cfg (IterableSimpleNamespace): The configuration object containing mode and task information.
        """
        if not self.enabled:

            return

        if len(self.events) < 25:
            params = {
                **self.metadata,
                "task": cfg.task,
                "model": cfg.model if cfg.model in GITHUB_ASSETS_NAMES else "custom",
            }
            if cfg.mode == "export":
                params["format"] = cfg.format
            self.events.append({"name": cfg.mode, "params": params})

        t = time.time()
        if (t - self.t) < self.rate_limit:

            return

        data = {"client_id": SETTINGS["uuid"], "events": self.events}

        smart_request("post", self.url, json=data, retry=0, verbose=False)

        self.events = []
        self.t = t

events = Events()

