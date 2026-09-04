"""TuiApp behavior: layout, input modes, key bindings, modals, and approval prompts."""

import asyncio

from wizolt.tui import TuiApp


class _StubJob:
    def __init__(self, status):
        self.status = status
















ACTIONS = [("Approve", ""), ("View order", "v"), ("Worker config", "c"), ("Refuse", "n")]


def _approval_app():
    """A TuiApp parked on an approval prompt, with the pending input future installed.

    The future belongs to the loop the test runs on, which is why the callers are async: that is
    also how the real prompt works -- the request is awaited on the loop that runs the app."""
    app = TuiApp()
    app._input_loop = asyncio.get_running_loop()
    app._input_pending = app._input_loop.create_future()
    app.input_mode = "approval"
    assert app.set_approval_form(ACTIONS) is True
    return app


def _answered(app):
    """(answer, resolved) for the app's pending input request."""
    pending = app._input_pending
    if pending is None or not pending.done():
        return None, False
    return pending.result(), True


def _active(app, key):
    return [binding for binding in reversed(app.make_bindings().bindings) if binding.keys == (key,) and binding.filter()]














































def quick_hint_app(hints=("run the tests", "show the diff", "commit")):
    submitted = []
    app = TuiApp(on_chat_submit=submitted.append, quick_hints_fn=lambda: hints)
    app.set_idle()
    return app, submitted














































































