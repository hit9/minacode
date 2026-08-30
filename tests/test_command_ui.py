"""Interactive command surfaces: the provider/model/api/reason selection chains, the diff
viewer, and the stored Bash output viewer."""


import pytest
from tui_harness import loop

from wizolt.base import (
    ImageRouteNotice,
    LogBlock,
    LogEdge,
    LogRole,
    ModelError,
)
from wizolt.cli import COMMANDS, CommandLoop
from wizolt.cli.commands import (
    api,
    config,
)
from wizolt.model import ModelClient
from wizolt.tui import TUI_MODAL_PENDING


def diff_loop(tmp_path):
    command_loop = loop(tmp_path)
    before = "".join(f"old {index}\n" for index in range(20))
    after = "".join(f"new {index}\n" for index in range(20))
    command_loop.session.store_turn_diff("tr.1", 1, "a.py", "unused", before=before, after=after, round=1)
    command_loop.session.store_turn_diff("tr.2", 2, "b.py", "unused", before="old\n", after="new\n", round=1)
    return command_loop


# The registry is the single source of command metadata; HELP stays a hand-written literal with
# manual wrapping and non-command sections, so every registered name and alias must appear in it.
# `/worker` is a pre-existing gap: it is registered in master's COMMAND_HANDLERS but missing from
# master's HELP literal. It is listed here so the omission stays visible instead of silent; any
# new registered command missing from HELP fails this test unless explicitly added to the set.
HELP_OMISSIONS = frozenset({"/worker"})


def test_registry_names_and_aliases_appear_in_help():
    missing = {name for command in COMMANDS for name in (command.name, *command.aliases) if name not in CommandLoop.HELP}
    assert missing <= HELP_OMISSIONS, f"registered commands missing from HELP: {sorted(missing - HELP_OMISSIONS)}"


def test_image_route_notice_matches_view_image_tree_vocabulary(tmp_path):
    command_loop = loop(tmp_path)
    blocks = []
    command_loop.tool_output = blocks.append

    command_loop.image_route_notice(ImageRouteNotice("main model rejected image input (400)", described_by="vision/model", images=("shot.png",)))

    [block] = blocks
    assert isinstance(block, LogBlock)
    root, children = block.items
    assert (root.label, root.text, root.role, root.meta) == (
        "Image",
        "shot.png",
        LogRole.META,
        " · main model rejected image input (400)",
    )
    [child] = children.items
    assert (child.label, child.text, child.role, child.edge) == ("described by", "vision/model", LogRole.TOOL, LogEdge.END)

    command_loop.image_route_notice(ImageRouteNotice("main model is text-only", described_by="vision/model", images=("a.png", "b.png")))
    multi_root = blocks[-1].items[0]
    assert (multi_root.label, multi_root.text) == ("Images", "2 attachments")


class ModalHarness:
    def __init__(self, keys, *, consumed=False):
        self.keys = list(keys)
        # consumed=True hands each key to the next modal in line instead of replaying the whole
        # sequence for every modal, which is how a multi-modal flow (list -> detail -> list) is
        # driven end to end.
        self.consumed = consumed
        self.pos = 0
        self.frames = []
        self.exclusive = []

    def show_modal(self, fragments_fn, key_fn, *, exclusive=False):
        self.exclusive.append(exclusive)
        self.frames.append(fragments_fn())
        result = TUI_MODAL_PENDING
        keys = self.keys[self.pos :] if self.consumed else self.keys
        for key in keys:
            if self.consumed:
                self.pos += 1
            result = key_fn(key, key if len(key) == 1 else "")
            self.frames.append(fragments_fn())
            if result is not TUI_MODAL_PENDING:
                return result
        return None






























































































def test_api_command_reports_an_incompatible_builtin_tools_configuration_without_clearing_it(tmp_path):
    """Switching /api reports inactive builtin tools and never rewrites provider config."""
    command_loop = loop(tmp_path)
    provider_config = command_loop.session.config.provider
    provider_config.url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    provider_config.model = "qwen3.8-max-preview"
    provider_config.key = "sk-test"
    provider_config.api = "responses"
    provider_config.builtin_tools = ({"type": "web_search"}, {"type": "web_extractor"})

    assert api(command_loop, "chat") == "Set provider.api = chat (wire: chat); builtin_tools inactive on chat"
    # The requested API value is applied and the provider configuration is left intact.
    assert provider_config.api == "chat"
    assert provider_config.builtin_tools == ({"type": "web_search"}, {"type": "web_extractor"})

    # The next request projects no provider-native tools on the mismatched wire.
    assert ModelClient(command_loop.session).builtin_tools() == []

    # Switching back restores the working Responses configuration without erasing it.
    assert api(command_loop, "responses") == "Set provider.api = responses (wire: responses)"
    assert provider_config.builtin_tools == ({"type": "web_search"}, {"type": "web_extractor"})


def test_api_command_reports_when_no_wire_accepts_the_configured_builtin_tools(tmp_path):
    """DeepSeek has no provider-side tools channel, so the shared config stays inactive."""
    command_loop = loop(tmp_path)
    provider_config = command_loop.session.config.provider
    provider_config.url = "https://api.deepseek.com/v1"
    provider_config.model = "deepseek-chat"
    provider_config.key = "sk-test"
    provider_config.builtin_tools = ({"type": "web_search"},)

    assert api(command_loop, "chat") == "Set provider.api = chat (wire: chat); builtin_tools inactive on chat"
    assert provider_config.builtin_tools == ({"type": "web_search"},)


def test_config_distinguishes_configured_and_active_builtin_tools(tmp_path):
    command_loop = loop(tmp_path)
    provider_config = command_loop.session.config.provider
    provider_config.url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    provider_config.model = "qwen3.8-max-preview"
    provider_config.api = "chat"
    provider_config.builtin_tools = ({"type": "web_search"}, {"type": "web_extractor"})

    inactive = config(command_loop, "")
    assert "provider.builtin_tools: web_search, web_extractor" in inactive
    assert "provider.resolved_builtin_tools: inactive on chat: web_search, web_extractor" in inactive

    provider_config.api = "responses"
    active = config(command_loop, "")
    assert "provider.resolved_builtin_tools: active: web_search, web_extractor" in active


def test_api_command_uses_the_same_entry_policy_as_the_request_boundary(tmp_path):
    """A valid wire with an unsupported entry is reported immediately, not only on send."""
    command_loop = loop(tmp_path)
    provider_config = command_loop.session.config.provider
    provider_config.url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    provider_config.model = "qwen3.8-max-preview"
    provider_config.key = "sk-test"
    provider_config.builtin_tools = ({"type": "code_interpreter"},)

    assert api(command_loop, "responses") == "Set provider.api = responses (wire: responses); unsupported builtin_tools: code_interpreter"
    with pytest.raises(ModelError):
        ModelClient(command_loop.session).builtin_tools()


# --- /worker pickers and tab completion, mirroring the /provider /model /reason surfaces. ---




# /worker api is the typed form of the [worker] api knob: it sets the override, "default" clears
# it back to inheriting the entry's own protocol, and an unknown value is rejected with usage.






# The confirm-time `c` loop reuses the shared choice selector: pick a knob, drive the matching
# /worker picker, and loop until done/Esc (or a non-interactive select yields nothing).


# The no-arg pickers follow the /provider picker pattern: select_choice is stubbed, and the
# selection runs the exact same set path as the typed form (live-apply, frozen-gate note).










# --- /worker provider cascade: the no-arg picker flows provider -> model -> reasoning. ---
















