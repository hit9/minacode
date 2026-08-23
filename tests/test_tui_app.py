"""TuiApp behavior: layout, input modes, key bindings, modals, and approval prompts."""

import threading

from minacode.tui import TuiApp


class _StubJob:
    def __init__(self, status):
        self.status = status
















ACTIONS = [("Approve", ""), ("View order", "v"), ("Worker config", "c"), ("Refuse", "n")]


def _approval_app():
    app = TuiApp()
    app._input_pending = threading.Event()
    app.input_mode = "approval"
    assert app.set_approval_form(ACTIONS) is True
    return app


def _active(app, key):
    return [binding for binding in reversed(app.make_bindings().bindings) if binding.keys == (key,) and binding.filter()]














































def quick_hint_app(hints=("run the tests", "show the diff", "commit")):
    submitted = []
    app = TuiApp(on_chat_submit=submitted.append, quick_hints_fn=lambda: hints)
    app.set_idle()
    return app, submitted














































































