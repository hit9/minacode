"""delegate approval (split from tests/test_worker_handoff.py)."""
from agent_harness import session
from test_worker_handoff import FakeModelClient, _delegate_session

from wizolt.tools import TOOL_REGISTRY


def test_delegate_send_is_confirmed_even_under_yolo(tmp_path):
    from wizolt.tools import DelegateTool

    s = session(tmp_path)
    s.settings.yolo = True

    send = DelegateTool(s, [{"action": "send", "order": "do the thing"}])
    assert send.needs_confirmation() is True
    assert send.always_confirms() is True

    for action in ("status", "reset"):
        other = DelegateTool(s, [{"action": action}])
        assert other.always_confirms() is False, action

    # Every other mutating tool keeps yolo's meaning: only Delegate opts out.
    from wizolt.tools import EditTool

    assert EditTool(s, ["a.py", []]).always_confirms() is False


def test_spawned_worker_reuses_the_parent_catalog(tmp_path):
    from wizolt.providers.sync import CatalogRuntime
    from wizolt.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    parent.catalog = CatalogRuntime(parent.config.data_dir)

    worker = DelegateTool(parent, [{"action": "status"}])._spawn_worker(parent)

    assert worker.catalog is parent.catalog

async def test_delegate_send_confirmation_prompt_and_reasons(tmp_path, monkeypatch):
    from wizolt.base import ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner
    from wizolt.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    for answer, expected in [("y", (True, "")), ("", (True, "")), ("n", (False, "")), ("a", (False, "a"))]:
        runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda _prompt, a=answer: a, output_fn=lambda text: None)
        confirmed, reason = await runner.confirm(
            ToolCall("delegate-1", "Delegate", [{"action": "send", "order": "o"}]), DelegateTool(parent, [{"action": "send", "order": "o"}])
        )
        assert (confirmed, reason) == expected, answer

    # A c-prefixed sentence is an ordinary reason, never the config key (whole-line exact match only).
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda prompt: "cost too high", output_fn=lambda text: None)
    confirmed, reason = await runner.confirm(
        ToolCall("delegate-2", "Delegate", [{"action": "send", "order": "o"}]), DelegateTool(parent, [{"action": "send", "order": "o"}])
    )
    assert (confirmed, reason) == (False, "cost too high")

def test_delegate_approval_brief_lists_send_and_worker_details(tmp_path):
    from prompt_toolkit.utils import get_cwidth

    from wizolt.base import LogRole, ToolCall
    from wizolt.config import (
        ProviderConfig,
    )
    from wizolt.tools import EditTool, toolblocks
    from wizolt.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    parent.config.providers["fast"] = ProviderConfig(model="worker-model", reasoning="high", api="responses")
    parent.config.worker_provider = "fast"
    parent.config.worker_model = "override-model"
    parent.config.worker_api = ""

    order_lines = [f"line {i}" for i in range(1, 16)]
    args = {"action": "send", "order": "\n".join(order_lines), "title": "fix things", "language": "Chinese", "max_steps": 7}
    tool = DelegateTool(parent, [args])
    block = toolblocks.approval_display(parent, ToolCall("delegate-1", "Delegate", [args]), tool, "confirm")
    rows = [(item.label, item.text) for item, _ in block.walk()]
    labels = [label for label, _ in rows]
    texts = [text for _, text in rows]
    assert any(label.strip() == "title" for label in labels) and "fix things" in texts
    # The order is a single line: first line plus a (… N more lines) tail, never a 12-line dump.
    order_label = next(label for label, text in rows if label.strip() == "order")
    assert order_label.strip() == "order"
    order_text = next(text for label, text in rows if label.strip() == "order")
    assert order_text.startswith("line 1")
    assert "14 more lines" in order_text  # 15 - 1 = 14 overflow, folded into the one line
    assert "line 2" not in order_text
    assert any(label.strip() == "language" for label in labels) and "Chinese" in texts
    assert any(label.strip() == "max_steps" for label in labels) and "7" in texts
    # The worker config is four rows, one per knob, with inherited values marked explicitly.
    assert all(label.strip() in {"provider", "model", "effort", "api"} for label, _ in rows if "provider" in label)
    assert "worker" not in labels  # no combined single-line row anymore
    assert next(text for label, text in rows if label.strip() == "provider") == "fast"  # explicit override
    assert next(text for label, text in rows if label.strip() == "model") == "override-model"
    assert next(text for label, text in rows if label.strip() == "effort") == "(inherit) high"
    assert next(text for label, text in rows if label.strip() == "api") == "(inherit) responses"  # worker_api empty
    # Cyan FIELD rows for the whole brief except the trailing gray key legend.
    assert all(item.role is LogRole.FIELD for item, _ in list(block.walk())[1:-1])
    legend = list(block.walk())[-1][0]
    assert legend.role is LogRole.META and "approve" in legend.text
    # Every field label is padded to one display width so the values start on one column.
    assert len({get_cwidth(label) for label, _ in rows[1:-1]}) == 1  # root and legend excluded

    # An explicit worker_api override wins over the entry's api.
    parent.config.worker_api = "chat"
    block = toolblocks.approval_display(parent, ToolCall("delegate-2", "Delegate", [args]), tool, "confirm")
    rows = [(item.label, item.text) for item, _ in block.walk()]
    api_row = next(text for label, text in rows if label.strip() == "api")
    assert api_row == "chat"
    assert "(inherit)" not in api_row

    # Non-send Delegate calls keep the plain display; Edit keeps its preview children.
    status_tool = DelegateTool(parent, [{"action": "status"}])
    block = toolblocks.approval_display(parent, ToolCall("delegate-3", "Delegate", [{"action": "status"}]), status_tool, "confirm")
    assert not block.has_children
    edit_tool = EditTool(parent, ["new.py", "", [{"op": "create", "content": "x\n"}]])
    block = toolblocks.approval_display(parent, ToolCall("edit-1", "Edit", ["new.py", "", [{"op": "create", "content": "x\n"}]]), edit_tool, "confirm")
    assert block.has_children

async def test_delegate_config_cycle_changes_worker_knobs_and_refreshes_live_worker(tmp_path):
    from dataclasses import replace

    from wizolt.base import LogBlock, ToolCall
    from wizolt.config import (
        ProviderConfig,
    )
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner
    from wizolt.session import Session
    from wizolt.tools.delegate import DelegateTool, refresh_worker_entry

    parent = _delegate_session(tmp_path)
    parent.config.providers["alt"] = ProviderConfig(model="alt-model", reasoning="low", api="anthropic")
    worker = Session(cwd=str(tmp_path), config=replace(parent.config), settings=parent.settings, uid=parent.uid + ".w", listed=False)
    parent.worker = worker

    # The injected picker loop (CommandLoop.run_worker_config in production) drives the changes and
    # writes the config / refreshes the live worker itself; the runner only triggers it on `c`.
    def picker():
        parent.config.worker_provider = "alt"
        parent.config.worker_model = "worker-m"
        parent.config.worker_reasoning = "off"
        parent.config.worker_api = "responses"
        refresh_worker_entry(parent.config, worker, "alt")

    calls = []
    answers = iter(["c", "y"])
    prompts = []
    outputs = []

    def input_fn(prompt):
        prompts.append(prompt)
        return next(answers)

    runner = ToolRunner(parent, ContextManager(parent), input_fn=input_fn, output_fn=outputs.append)
    runner.worker_config_picker = lambda: calls.append(1) or picker()
    confirmed, reason = await runner.confirm(
        ToolCall("delegate-1", "Delegate", [{"action": "send", "order": "o"}]), DelegateTool(parent, [{"action": "send", "order": "o"}])
    )
    assert (confirmed, reason) == (True, "")
    assert calls == [1]  # the `c` key drove the picker loop exactly once
    assert parent.config.worker_provider == "alt"
    assert parent.config.worker_model == "worker-m"
    assert parent.config.worker_reasoning == "off"
    assert parent.config.worker_api == "responses"
    # The live worker's active entry carries the overrides (copy-on-write, never shared).
    assert worker.config.active_provider == "alt"
    entry = worker.config.providers["alt"]
    assert (entry.model, entry.reasoning, entry.api) == ("worker-m", "off", "responses")
    assert worker.config.providers is not parent.config.providers
    # The prompt re-asked, and the config block printed the values the picker left behind. The
    # brief is NOT redrawn: the first copy is still on screen, and a second is transcript noise.
    assert sum(1 for prompt in prompts if "Approve delegation? [Y/n/c] " in prompt) == 2
    assert any("worker config" in str(out) for out in outputs if isinstance(out, LogBlock))
    assert len([out for out in outputs if isinstance(out, LogBlock) and any(item.label.strip() == "order" for item, _ in out.walk())]) == 1

    # Without an injected picker (headless / non-CommandLoop) the `c` key prints the config block
    # and re-asks without crashing.
    prompts = []
    outputs = []
    answers = iter(["c", "y"])
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda prompt: prompts.append(prompt) or next(answers), output_fn=outputs.append)
    confirmed, reason = await runner.confirm(
        ToolCall("delegate-2", "Delegate", [{"action": "send", "order": "o"}]), DelegateTool(parent, [{"action": "send", "order": "o"}])
    )
    assert (confirmed, reason) == (True, "")
    assert any("worker config" in str(out) for out in outputs if isinstance(out, LogBlock))
    assert parent.config.worker_provider == "alt"  # untouched without a picker
    assert parent.config.worker_api == "responses"  # untouched without a picker

async def test_delegate_view_opens_viewer_then_approves(tmp_path):
    from wizolt.base import ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner
    from wizolt.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    order_lines = [f"line {i}" for i in range(1, 16)]
    order = "\n".join(order_lines)
    args = {"action": "send", "order": order, "title": "fix things", "max_steps": 7}
    seen = []
    answers = iter(["v", "y"])
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda prompt: next(answers), output_fn=lambda text: None)
    runner.text_viewer = lambda view: seen.append((view.text, view.rows))
    confirmed, reason = await runner.confirm(ToolCall("delegate-1", "Delegate", [args]), DelegateTool(parent, [args]))
    assert (confirmed, reason) == (True, "")
    assert len(seen) == 1  # the `v` key opened the viewer exactly once
    viewed_order, header_rows = seen[0]
    assert viewed_order == order  # full, untruncated order
    assert all(line in viewed_order for line in order_lines)
    assert any(label == "title" and value == "fix things" for label, value in header_rows)
    assert any(label == "max_steps" and value == "7" for label, value in header_rows)
    assert any(label in {"provider", "model", "effort", "api"} for label, _ in header_rows)
    assert not any(label == "order" for label, _ in header_rows)  # order is shown in full in the viewer

async def test_delegate_view_reflects_a_worker_config_changed_by_c(tmp_path):
    """`c` then `v`: the viewer reports the configuration the send would run under, so it has to
    read that configuration when the key is pressed, not as it stood when the prompt was drawn."""
    from wizolt.base import ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner
    from wizolt.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    args = {"action": "send", "order": "do the thing"}
    seen = []
    answers = iter(["c", "v", "y"])
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda prompt: next(answers), output_fn=lambda text: None)
    runner.text_viewer = lambda view: seen.append(dict(view.rows))
    runner.worker_config_picker = lambda: setattr(parent.config, "worker_model", "chosen-in-the-c-cycle")

    confirmed, _ = await runner.confirm(ToolCall("delegate-1", "Delegate", [args]), DelegateTool(parent, [args]))

    assert confirmed
    assert seen and seen[0]["model"] == "chosen-in-the-c-cycle"

async def test_delegate_view_headless_fallback_prints_full_order(tmp_path):
    from wizolt.base import LogBlock, ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner
    from wizolt.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    order_lines = [f"line {i}" for i in range(1, 16)]
    order = "\n".join(order_lines)
    args = {"action": "send", "order": order}
    outputs = []
    answers = iter(["v", "n"])
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda prompt: next(answers), output_fn=outputs.append)
    # text_viewer stays None: headless / non-CommandLoop runners print the full order instead.
    confirmed, reason = await runner.confirm(ToolCall("delegate-1", "Delegate", [args]), DelegateTool(parent, [args]))
    assert (confirmed, reason) == (False, "")
    texts = [item.text for out in outputs if isinstance(out, LogBlock) for item, _ in out.walk()]
    assert all(line in texts for line in order_lines)  # nothing dropped, unlike the brief excerpt

async def test_delegate_view_empty_order_is_noop(tmp_path):
    from wizolt.base import LogBlock, ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner
    from wizolt.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    args = {"action": "send", "order": ""}
    seen = []
    outputs = []
    answers = iter(["v", "y"])
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda prompt: next(answers), output_fn=outputs.append)
    runner.text_viewer = lambda view: seen.append((view.text, view.rows))
    confirmed, reason = await runner.confirm(ToolCall("delegate-1", "Delegate", [args]), DelegateTool(parent, [args]))
    # Nothing to view, so `v` is not an action here and falls through to the one thing any other
    # unrecognized line means at this prompt: a refusal carrying what was typed as its reason.
    assert (confirmed, reason) == (False, "v")
    assert seen == []
    assert not any(isinstance(out, LogBlock) and any(item.label.strip() == "order" for item, _ in out.walk()) for out in outputs)

def test_delegate_approval_legend_mentions_view(tmp_path):
    from wizolt.base import LogRole
    from wizolt.tools import toolblocks
    from wizolt.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    children = toolblocks.delegate_approval_children(DelegateTool(parent, [{"action": "send", "order": "o"}]))
    legend = children[-1]
    assert legend.role is LogRole.META
    assert "v view order" in legend.text

async def test_confirm_cancelled_input_refuses_without_a_reason(tmp_path):
    # The TUI signals Ctrl-C / Ctrl-D-on-empty / app shutdown by returning None from request_input.
    # confirm() must read that as a plain refusal: not "" (the default approve) and not a reason,
    # which would reach the model as text the user never typed. Holds for every tool, not just
    # Delegate, so check the Delegate prompt and an ordinary one.
    from wizolt.base import ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner
    from wizolt.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    args = {"action": "send", "order": "o"}
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda prompt: None, output_fn=lambda text: None)
    assert await runner.confirm(ToolCall("delegate-1", "Delegate", [args]), DelegateTool(parent, [args])) == (False, "")

    call = ToolCall("bash-1", "Bash", ["echo hi"])
    assert await runner.confirm(call, TOOL_REGISTRY["Bash"](parent, ["echo hi"])) == (False, "")

async def test_approval_brief_prints_once_however_many_side_trips(tmp_path):
    # `v` and `c` come back to the same prompt, and each redraw used to stack another full copy of
    # the brief in the transcript. It is printed once; the side trips report themselves.
    from wizolt.base import LogBlock, ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner
    from wizolt.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    args = {"action": "send", "order": "line one\nline two", "title": "fix things"}
    answers = iter(["v", "c", "v", "y"])
    outputs, prompts = [], []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda prompt: prompts.append(prompt) or next(answers), output_fn=outputs.append)
    runner.text_viewer = lambda view: None
    confirmed, _ = await runner.confirm(ToolCall("delegate-1", "Delegate", [args]), DelegateTool(parent, [args]))

    assert confirmed is True
    assert len(prompts) == 4  # every side trip re-asked
    briefs = [out for out in outputs if isinstance(out, LogBlock) and any(item.label.strip() == "order" for item, _ in out.walk())]
    assert len(briefs) == 1

async def test_approval_form_actions_offered_per_tool_and_only_where_they_work(tmp_path):
    from wizolt.base import ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner
    from wizolt.tools import toolblocks
    from wizolt.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    declared = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda prompt: "y", output_fn=lambda text: None)
    runner.approval_form = lambda actions: bool(declared.append(list(actions))) or True

    # Approve is first because it is the default; Refuse is last because Escape already refuses in
    # one key, while every Tab spent reaching View order is a key the user actually presses.
    send = DelegateTool(parent, [{"action": "send", "order": "o"}])
    send_actions = toolblocks.approval_actions(send, True)
    assert send_actions == [("Approve", ""), ("View order", "v"), ("Worker config", "c"), ("Refuse", "n")]
    assert runner.declare_approval_form(send_actions) is True

    # No order means nothing to view, so the action is not offered rather than opening an empty one.
    orderless = DelegateTool(parent, [{"action": "send", "order": ""}])
    orderless_actions = toolblocks.approval_actions(orderless, True)
    assert orderless_actions == [("Approve", ""), ("Worker config", "c"), ("Refuse", "n")]
    assert runner.declare_approval_form(orderless_actions) is True

    # Every other tool gets approve/refuse: `c` and `v` are Delegate actions, Bash has no equivalent.
    bash = TOOL_REGISTRY["Bash"](parent, ["rm -rf build"])
    bash_actions = toolblocks.approval_actions(bash, False)
    assert bash_actions == [("Approve", ""), ("Refuse", "n")]
    assert runner.declare_approval_form(bash_actions) is True
    assert [len(actions) for actions in declared] == [4, 3, 2]

    # Headless: nothing is wired, so nothing is claimed and the typed protocol is what is offered.
    runner.approval_form = None
    assert runner.declare_approval_form(send_actions) is False
    assert toolblocks.approval_prompt(True, []) == "Approve delegation? [Y/n/c] "
    assert toolblocks.approval_prompt(False, []) == "Approve? [Y/n or reason] "
    assert toolblocks.approval_prompt(False, [("Approve", "")]) == "reason › "

    # Every action's answer is a line confirm() already understands, so the two paths cannot drift.
    for _, answer in [("Approve", ""), ("View order", "v"), ("Worker config", "c"), ("Refuse", "n")]:
        typed = ToolRunner(parent, ContextManager(parent), input_fn=lambda prompt, a=answer: a, output_fn=lambda text: None)
        typed.text_viewer = lambda view: None
        typed.worker_config_picker = lambda: None
        if answer in {"v", "c"}:
            continue  # these re-ask forever against a constant input_fn; covered by the side-trip test
        assert await typed.confirm(ToolCall("bash-1", "Bash", ["rm -rf build"]), bash) == ((True, "") if answer == "" else (False, ""))

def test_delegate_legend_prints_only_without_an_action_row(tmp_path):
    from wizolt.base import LogEdge
    from wizolt.tools import toolblocks
    from wizolt.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    tool = DelegateTool(parent, [{"action": "send", "order": "o"}])

    # Headless keeps the typed legend: those words plus Enter are all that path has.
    headless = toolblocks.delegate_approval_children(tool)
    assert headless[-1].text == "Y/Enter approve · n refuse · c worker config · v view order · else reason"
    assert headless[-1].edge is LogEdge.END

    # With a live action row the legend would be a stale duplicate, so the brief ends at its rows —
    # and the last one has to take over the closing edge.
    actions = toolblocks.approval_actions(tool, True)
    children = toolblocks.delegate_approval_children(tool, actions, actions)
    assert all("Y/Enter approve" not in (line.text or "") for line in children)
    assert children[-1].edge is LogEdge.END
    assert children[0].edge is LogEdge.BRANCH

def test_delegate_legend_offers_only_the_actions_the_call_has(tmp_path):
    """The action row already hid `View order` when the send carries no order, but the legend -- the
    only guidance a headless run gets -- still advertised it, and typing `v` then re-asked in silence
    with nothing viewed. Both are built from one list of actions, so they cannot disagree."""
    from wizolt.tools import toolblocks
    from wizolt.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    orderless = DelegateTool(parent, [{"action": "send"}])

    legend = toolblocks.delegate_approval_children(orderless)[-1].text or ""
    assert "v view order" not in legend
    assert "c worker config" in legend  # the actions it does have are untouched
    assert legend == toolblocks.approval_legend(toolblocks.approval_actions(orderless, True), "order")

def test_delegate_order_viewer_wraps_by_terminal_cells(monkeypatch):
    # A CJK order is two terminal cells per character. Wrapping by character count (textwrap) makes
    # every row twice as wide as the terminal, and the modal window does not wrap, so the overflow
    # is simply lost. No rendered row may exceed the terminal width.
    import os
    from types import SimpleNamespace

    from prompt_toolkit.utils import get_cwidth

    from wizolt.base import ApprovalView
    from wizolt.cli.modals import approval_text_viewer

    size = os.terminal_size((60, 40))
    monkeypatch.setattr("wizolt.cli.modals.shutil.get_terminal_size", lambda *args: size)

    captured = {}
    loop = SimpleNamespace(tui=SimpleNamespace(show_modal=lambda fragments_fn, key_fn, **kwargs: captured.update(fragments_fn=fragments_fn)))
    order = "\n".join(["把这个仓库里的审批快捷键改造一遍并补上测试" * 3, "", "```python", "def nested():", "    x = 1", "```"])
    approval_text_viewer(loop, ApprovalView("order", order, "", [("title", "中文标题" * 10)]))

    rows = "".join(text for _, text in captured["fragments_fn"]()).splitlines()
    assert rows, "the viewer rendered nothing"
    assert all(get_cwidth(row) <= 60 for row in rows), max(rows, key=get_cwidth)
    assert any("把这个仓库里的审批快捷键" in row for row in rows)  # the CJK text is still there, just wrapped
    # A fenced code block keeps its indentation, so code in an order stays readable.
    assert any(row.lstrip().startswith("def nested():") for row in rows)
    nested = [row for row in rows if "x = 1" in row]
    assert len(nested) == 1 and nested[0].startswith("       ")

def test_delegate_order_viewer_is_exclusive_and_scrolls(monkeypatch):
    import os
    from types import SimpleNamespace

    from wizolt.base import ApprovalView
    from wizolt.cli.modals import approval_text_viewer
    from wizolt.tui import TUI_MODAL_PENDING

    # Fixed terminal size keeps the viewport deterministic: 40 lines - 6 = 34 visible rows.
    size = os.terminal_size((120, 40))
    monkeypatch.setattr("wizolt.cli.modals.shutil.get_terminal_size", lambda *args: size)

    captured = {}
    loop = SimpleNamespace(
        tui=SimpleNamespace(
            show_modal=lambda fragments_fn, key_fn, **kwargs: captured.update(
                fragments_fn=fragments_fn, key_fn=key_fn, exclusive=kwargs.get("exclusive", False)
            )
        )
    )
    order_lines = [f"line {i} " + "word " * 30 for i in range(200)]  # wraps to ~400 lines
    approval_text_viewer(loop, ApprovalView("order", "\n".join(order_lines), "", [("title", "fix things")]))
    fragments = captured["fragments_fn"]
    handle_key = captured["key_fn"]
    assert captured["exclusive"] is True  # full-screen alternate-screen viewer

    def visible_text() -> str:
        return "".join(text for _, text in fragments())

    first = visible_text()
    assert "Order · read-only" in first
    assert "line 0 " in first

    # down/j scroll one line each; the visible slice changes.
    assert handle_key("down", "") is TUI_MODAL_PENDING
    second = visible_text()
    assert second != first
    assert handle_key("j", "") is TUI_MODAL_PENDING
    third = visible_text()
    assert third != second

    # c-d scrolls half a page (viewport 34 -> +17 rows).
    assert handle_key("c-d", "") is TUI_MODAL_PENDING
    fourth = visible_text()
    assert fourth != third

    # g returns to the top; G jumps to the bottom (clamped at render time).
    assert handle_key("g", "") is TUI_MODAL_PENDING
    assert visible_text() == first
    assert handle_key("G", "") is TUI_MODAL_PENDING
    bottom = visible_text()
    assert "line 199 " in bottom

    # escape/q/c-o close the viewer.
    assert handle_key("escape", "") is None
    assert handle_key("q", "") is None
    assert handle_key("c-o", "") is None

def test_delegate_order_viewer_renders_markdown(monkeypatch):
    import os
    from types import SimpleNamespace

    from wizolt.base import ApprovalView
    from wizolt.cli.modals import approval_text_viewer

    size = os.terminal_size((120, 40))
    monkeypatch.setattr("wizolt.cli.modals.shutil.get_terminal_size", lambda *args: size)

    captured = {}
    loop = SimpleNamespace(tui=SimpleNamespace(show_modal=lambda fragments_fn, key_fn, **kwargs: captured.update(fragments_fn=fragments_fn)))
    order = "## Section\n\n- item one\n- item two\n\n```python\nprint(1)\n```"
    approval_text_viewer(loop, ApprovalView("order", order, "", [("title", "fix things")]))

    rendered = "".join(text for _, text in captured["fragments_fn"]())
    assert "Section" in rendered
    assert "##" not in rendered  # heading marker consumed by the markdown renderer
    assert "```" not in rendered  # code fence consumed too
    assert "item one" in rendered
    assert "item two" in rendered
    assert "print(1)" in rendered

def test_delegate_order_viewer_keeps_source_line_breaks(monkeypatch):
    # An order's newlines are structural: file lists and step-per-line instructions must not be
    # folded into one paragraph the way Markdown folds in-paragraph newlines to spaces.
    import os
    from types import SimpleNamespace

    from wizolt.base import ApprovalView
    from wizolt.cli.modals import approval_text_viewer

    size = os.terminal_size((120, 40))
    monkeypatch.setattr("wizolt.cli.modals.shutil.get_terminal_size", lambda *args: size)

    captured = {}
    loop = SimpleNamespace(tui=SimpleNamespace(show_modal=lambda fragments_fn, key_fn, **kwargs: captured.update(fragments_fn=fragments_fn)))
    order = "Touch these files:\nwizolt/loop.py\nwizolt/parser.py\nDo not touch tests."
    approval_text_viewer(loop, ApprovalView("order", order, "", [("title", "fix things")]))

    rows = [row.strip() for row in "".join(text for _, text in captured["fragments_fn"]()).splitlines()]
    for source_line in order.splitlines():
        assert source_line in rows, f"{source_line!r} was folded into another line"

def test_delegate_order_viewer_field_header_alignment(monkeypatch):
    import os
    from types import SimpleNamespace

    from prompt_toolkit.utils import get_cwidth

    from wizolt.base import ApprovalView
    from wizolt.cli.modals import approval_text_viewer

    size = os.terminal_size((120, 40))
    monkeypatch.setattr("wizolt.cli.modals.shutil.get_terminal_size", lambda *args: size)

    captured = {}
    loop = SimpleNamespace(tui=SimpleNamespace(show_modal=lambda fragments_fn, key_fn, **kwargs: captured.update(fragments_fn=fragments_fn)))
    approval_text_viewer(loop, ApprovalView("order", "order", "", [("title", "fix"), ("lang", "python"), ("max_steps", "3")]))

    fragments = captured["fragments_fn"]()
    cyan = {text for style, text in fragments if style == "ansicyan" and text.strip() in {"title", "lang", "max_steps"}}
    assert len(cyan) == 3
    assert {get_cwidth(text) for text in cyan} == {9}  # every label padded to the widest one

def test_delegate_order_viewer_header_separator(monkeypatch):
    import os
    from types import SimpleNamespace

    from prompt_toolkit.utils import get_cwidth

    from wizolt.base import ApprovalView
    from wizolt.cli.modals import approval_text_viewer

    size = os.terminal_size((120, 40))
    monkeypatch.setattr("wizolt.cli.modals.shutil.get_terminal_size", lambda *args: size)

    captured = {}
    loop = SimpleNamespace(tui=SimpleNamespace(show_modal=lambda fragments_fn, key_fn, **kwargs: captured.update(fragments_fn=fragments_fn)))
    approval_text_viewer(loop, ApprovalView("order", "order", "", [("title", "fix things")]))

    lines = "".join(text for _, text in captured["fragments_fn"]()).splitlines()
    separators = [line for line in lines if line.strip() and set(line) <= {"─", " "}]
    assert separators
    assert all(get_cwidth(line) == 118 for line in separators)  # content width: 120 minus the two-space margins
    order_row = [index for index, line in enumerate(lines) if line.strip() == "order"]
    assert order_row
    assert lines.index(separators[0]) < order_row[0]  # separator sits after the fields, before the body

def test_delegate_order_viewer_markdown_fits_narrow_terminal(monkeypatch):
    import os
    from types import SimpleNamespace

    from prompt_toolkit.utils import get_cwidth

    from wizolt.base import ApprovalView
    from wizolt.cli.modals import approval_text_viewer

    size = os.terminal_size((60, 40))
    monkeypatch.setattr("wizolt.cli.modals.shutil.get_terminal_size", lambda *args: size)

    captured = {}
    loop = SimpleNamespace(tui=SimpleNamespace(show_modal=lambda fragments_fn, key_fn, **kwargs: captured.update(fragments_fn=fragments_fn)))
    order = '## 标题\n\n- 把这段中文说明加进审批流程并补充测试\n\n```python\nprint("中文")\n```'
    approval_text_viewer(loop, ApprovalView("order", order, "", [("title", "中文标题" * 10)]))

    rendered = "".join(text for _, text in captured["fragments_fn"]())
    rows = rendered.splitlines()
    assert rows, "the viewer rendered nothing"
    assert all(get_cwidth(row) <= 60 for row in rows), max(rows, key=get_cwidth)

async def test_delegate_yolo_without_authorization_still_confirms(tmp_path, monkeypatch):
    from wizolt.base import ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    parent.settings.yolo = True
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    prompts = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda prompt: prompts.append(prompt) or "y", output_fn=lambda text: None)

    status, _, _ = await runner.run_one(ToolCall("delegate-1", "Delegate", [{"action": "send", "order": "o"}]))
    assert status == "ok"
    assert len(prompts) == 1  # yolo alone does not skip a Delegate send

async def test_delegate_send_refused_does_not_run(tmp_path, monkeypatch):
    from wizolt.base import ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda prompt: "n", output_fn=lambda text: None)

    status, message, _ = await runner.run_one(ToolCall("delegate-1", "Delegate", [{"action": "send", "order": "o"}]))
    assert status == "refused"
    assert "refused" in message
    assert not model.requests  # the worker never ran
