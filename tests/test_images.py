import base64
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from PIL import Image
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory

import minacode.cli as loop_module
import minacode.tui as tui_module
from minacode.base import ModelError, ToolCall, ToolError
from minacode.cli import CommandLoop
from minacode.config import (
    Config,
    ProviderConfig,
)
from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.image import (
    IMAGE_MARKER,
    IMAGE_REFS_KEY,
    IMAGE_TEXT_ONLY_KEY,
    TOOL_IMAGE_OBSERVATION_KEY,
    TOOL_IMAGE_OBSERVATION_PREFIX,
    ImageInputs,
    ImageRef,
    UserInput,
)
from minacode.model import ModelClient
from minacode.runner import ToolRunner
from minacode.session import Session, SessionSnapshotStore
from minacode.tools import ViewImageTool
from minacode.tui import TuiApp


def image_file(path, *, size=(32, 24), image_format="PNG", color=(12, 34, 56)):
    Image.new("RGB", size, color).save(path, format=image_format)
    return path


def session(tmp_path):
    config = Config(data_dir=str(tmp_path / "data"))
    config.providers = {"default": ProviderConfig(url="http://test", key="test", model="vision")}
    return Session(cwd=str(tmp_path), config=config)


def test_recognize_local_image_paths_and_leave_other_tokens_alone(tmp_path):
    first = image_file(tmp_path / "one.png")
    image_file(tmp_path / "two words.webp", image_format="WEBP")

    value = ImageInputs(cwd=str(tmp_path)).recognize(f"review ({first.name}) and two\\ words.webp; missing.png stays")

    assert str(value).count(IMAGE_MARKER) == 2
    assert [image.name for image in value.images] == ["one.png", "two words.webp"]
    assert value.display_text() == "review ([Image #1 · one.png]) and [Image #2 · two words.webp]; missing.png stays"
    assert value.original_text() == f"review ({first.name}) and two\\ words.webp; missing.png stays"


def test_recognize_quoted_path_and_attach_duplicate_only_once(tmp_path):
    image_file(tmp_path / "same image.png")

    value = ImageInputs(cwd=str(tmp_path)).recognize("look at 'same image.png' and 'same image.png'")

    assert len(value.images) == 1
    assert value.original_text() == "look at 'same image.png' and 'same image.png'"


def test_animated_gif_is_not_recognized(tmp_path):
    path = tmp_path / "animated.gif"
    frames = [Image.new("RGB", (4, 4), color) for color in ("red", "blue")]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=10, loop=0)

    value = ImageInputs(cwd=str(tmp_path)).recognize(path.name)

    assert value == path.name
    assert value.images == ()


def test_session_stores_content_addressed_image_and_persists_refs(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "screen.png", size=(640, 480))
    value = s.images.recognize(f"describe {path.name}")

    message = s.images.message(value)
    s.messages.append(message)
    s.save_snapshot()

    image = ImageRef.from_json(message[IMAGE_REFS_KEY][0])
    assert image is not None
    asset = os.path.join(SessionSnapshotStore.project_dir(s.config.data_dir, s.cwd), s.uid + ".assets", image.ref)
    assert os.path.isfile(asset)
    with open(asset, "rb") as file:
        assert file.read() == path.read_bytes()

    restored = Session.load_snapshot(s.uid, config=s.config)
    assert restored.messages[0] == message
    assert ContextManager(restored).messages_text(restored.messages[:1]) == "user:\ndescribe [Image #1 · screen.png]"


def test_submission_revalidates_an_image_changed_after_recognition(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "changing.png")
    detected = s.images.recognize(path.name)
    original_ref = detected.images[0].ref
    image_file(path, color=(99, 88, 77))

    stored = s.images.prepare(detected)

    assert stored.images[0].ref != original_ref
    assert s.images.chat_content(s.images.message(stored))[0]["type"] == "image_url"


def test_missing_stored_asset_is_a_local_model_error(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "missing.png")
    message = s.images.message(s.images.recognize(path.name))
    image = ImageRef.from_json(message[IMAGE_REFS_KEY][0])
    assert image is not None
    asset = os.path.join(SessionSnapshotStore.project_dir(s.config.data_dir, s.cwd), s.uid + ".assets", image.ref)
    os.unlink(asset)

    with pytest.raises(ModelError, match="Stored image is missing"):
        s.images.responses_content(message)


def test_session_queue_round_trips_images_and_garbage_collects_assets(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "queued.jpg", image_format="JPEG")
    value = s.images.prepare(s.images.recognize(path.name))
    s.enqueue_user_input(value)
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config)
    queued = restored.pending_user_inputs[0]
    assert queued.text == "[Image #1 · queued.jpg]"
    assert queued.user_input().display_text() == queued.text

    assets = os.path.join(SessionSnapshotStore.project_dir(s.config.data_dir, s.cwd), s.uid + ".assets")
    assert os.path.isdir(assets)
    s.pending_user_inputs.clear()
    s.save_snapshot()
    assert not os.path.exists(assets)


def test_garbage_collect_spares_dotfile_staging_and_drops_stray_files(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "kept.png")
    s.messages.append(s.images.message(s.images.recognize(path.name)))
    s.save_snapshot()

    assets = os.path.join(SessionSnapshotStore.project_dir(s.config.data_dir, s.cwd), s.uid + ".assets")
    staged = os.path.join(assets, ".image-93e8u1vn")
    stray = os.path.join(assets, "orphan-deadbeef")
    with open(staged, "w") as file:
        file.write("staging")
    with open(stray, "w") as file:
        file.write("stray")

    s.save_snapshot()

    assert os.path.isfile(staged)  # .image-* staging from ImageInputs._store survives GC
    assert not os.path.exists(stray)  # unreferenced non-dotfile is still collected
    assert os.listdir(assets)  # the referenced asset is kept


def test_garbage_collect_clears_stale_image_staging_and_spares_fresh(tmp_path):
    """GC clears a .image-* staging file left by a crashed save, but spares one that could still
    be inside the copy+replace window: residue cannot pile up (or block the assets directory's
    removal) without racing a real write."""
    s = session(tmp_path)
    path = image_file(tmp_path / "kept.png")
    s.messages.append(s.images.message(s.images.recognize(path.name)))
    s.save_snapshot()

    assets = os.path.join(SessionSnapshotStore.project_dir(s.config.data_dir, s.cwd), s.uid + ".assets")
    stale = os.path.join(assets, ".image-93e8u1vn")
    fresh = os.path.join(assets, ".image-93e8u1vo")
    with open(stale, "w") as file:
        file.write("residue")
    with open(fresh, "w") as file:
        file.write("in flight")
    os.utime(stale, (time.time() - 3600, time.time() - 3600))

    s.save_snapshot()

    assert not os.path.exists(stale)  # crash residue is collected
    assert os.path.isfile(fresh)  # a staging file inside the copy+replace window is spared
    assert any(name for name in os.listdir(assets) if not name.startswith("."))  # the referenced asset is kept


def test_store_survives_snapshot_gc_between_mkstemp_and_replace(tmp_path, monkeypatch):
    s = session(tmp_path)
    settled = image_file(tmp_path / "settled.png", color=(1, 2, 3))
    s.messages.append(s.images.message(s.images.recognize(settled.name)))
    s.save_snapshot()  # a real snapshot, so the save during the race reaches garbage collection

    racing = image_file(tmp_path / "racing.png", color=(4, 5, 6))
    value = s.images.recognize(racing.name)

    real_replace = os.replace
    intercepted = {"hit": False}

    def racing_replace(source, destination):
        if not intercepted["hit"] and os.path.basename(source).startswith(".image-"):
            intercepted["hit"] = True
            s.save_snapshot()  # the agent thread's snapshot GC runs while staging is unreferenced
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", racing_replace)

    stored = s.images.prepare(value)

    assert intercepted["hit"] is True  # the race was actually exercised
    assert os.path.isfile(os.path.join(s.images.assets_dir(), stored.images[0].ref))
    with open(os.path.join(s.images.assets_dir(), stored.images[0].ref), "rb") as file:
        assert file.read() == racing.read_bytes()


def test_recalling_image_follow_up_keeps_asset_until_resubmission(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "recall.png")
    s.enqueue_user_input(s.images.prepare(s.images.recognize(path.name)))
    s.save_snapshot()
    command_loop = CommandLoop(Agent(s, output_fn=lambda _text: None), output_fn=lambda _text: None)

    recalled = command_loop.recall_pending_input(lambda: None)

    assert isinstance(recalled, UserInput)
    assert recalled.display_text() == "[Image #1 · recall.png]"
    assert s.images.chat_content(s.images.message(recalled))[0]["type"] == "image_url"


def test_simple_cli_preserves_images_when_combining_pending_inputs(tmp_path, monkeypatch):
    s = session(tmp_path)
    path = image_file(tmp_path / "queued.png")
    s.enqueue_user_input(s.images.prepare(s.images.recognize(path.name)))
    agent = Agent(s, output_fn=lambda _text: None)
    received = []
    monkeypatch.setattr(agent, "run", lambda value: received.append(value) or "done")
    monkeypatch.setattr(loop_module.UpdateChecker, "start", lambda _self: None)
    monkeypatch.setattr(loop_module.CodeIndex, "refresh_existing_async", lambda _self: False)

    def eof(_prompt):
        raise EOFError

    command_loop = CommandLoop(agent, input_fn=eof, output_fn=lambda _text: None)

    assert command_loop.run() == 0
    assert len(received) == 1
    assert isinstance(received[0], UserInput)
    assert [image.name for image in received[0].images] == ["queued.png"]


def test_expired_session_removes_its_image_assets(tmp_path):
    old = session(tmp_path)
    path = image_file(tmp_path / "expired.png")
    old.messages.append(old.images.message(old.images.recognize(path.name)))
    old.save_snapshot()
    log = SessionSnapshotStore.session_path(old.config.data_dir, old.cwd, old.uid)
    assets = log[: -len(".jsonl")] + ".assets"
    stale = time.time() - 3 * 86400
    os.utime(log, (stale, stale))
    current = session(tmp_path)
    current.settings.session_retention_days = 1

    assert SessionSnapshotStore.clean_expired(current) == 1
    assert not os.path.exists(log)
    assert not os.path.exists(assets)


def test_protocol_payloads_use_each_standard_image_shape(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "pixel.png", size=(1, 1))
    message = s.images.message(s.images.recognize("what is this? pixel.png"))
    encoded = base64.b64encode(path.read_bytes()).decode()
    data_url = "data:image/png;base64," + encoded

    assert s.images.chat_content(message) == [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": "what is this? [Image #1 · pixel.png]"},
    ]
    assert s.images.responses_content(message) == [
        {"type": "input_image", "image_url": data_url},
        {"type": "input_text", "text": "what is this? [Image #1 · pixel.png]"},
    ]
    assert s.images.anthropic_content(message) == [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}},
        {"type": "text", "text": "what is this? [Image #1 · pixel.png]"},
    ]

    model = ModelClient(s)
    assert model.responses_input([message]) == [{"role": "user", "content": s.images.responses_content(message)}]
    assert model.anthropic_messages([message]) == [{"role": "user", "content": s.images.anthropic_content(message)}]


def test_view_image_tool_validates_stores_and_builds_model_observation(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "screen shot.png", size=(640, 480))
    tool = ViewImageTool(s, [path.name, "read the error"])

    output = tool.call()
    observation = tool.model_observation()

    assert tool.needs_confirmation() is False
    assert 'path="screen shot.png"' in output
    assert 'media_type="image/png" width=640 height=480' in output
    assert observation is not None
    assert ImageInputs.is_tool_observation(observation)
    assert observation["content"] == TOOL_IMAGE_OBSERVATION_PREFIX + "\n[Image #1 · screen shot.png]"
    assert s.images.tool_observation_question(observation) == "read the error"
    assert s.images.chat_content(observation)[0]["type"] == "image_url"
    assert ImageInputs.is_tool_observation({key: value for key, value in observation.items() if key != IMAGE_REFS_KEY})
    assert not ImageInputs.is_tool_observation({"role": "user", "content": observation["content"]})
    without_marker = {key: value for key, value in observation.items() if key != TOOL_IMAGE_OBSERVATION_KEY}
    assert ContextManager(s).estimated_tokens([observation]) == ContextManager(s).estimated_tokens([without_marker])


def test_view_image_tool_rejects_invalid_or_disabled_input(tmp_path):
    s = session(tmp_path)
    (tmp_path / "not-image.png").write_text("not pixels", encoding="utf-8")

    with pytest.raises(ToolError, match="Cannot read image"):
        ViewImageTool(s, ["not-image.png"]).call()

    path = image_file(tmp_path / "disabled.png")
    s.config.provider.image_input = "off"
    with pytest.raises(ToolError, match="Image input is disabled"):
        ViewImageTool(s, [path.name]).call()


def test_view_image_tool_requires_confirmation_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = image_file(tmp_path / "outside.png")
    s = Session(cwd=str(workspace), config=Config(data_dir=str(tmp_path / "data")))
    tool = ViewImageTool(s, [str(outside)])
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda _prompt: "no", output_fn=lambda _text: None)

    assert tool.needs_confirmation() is True
    messages = runner.run([ToolCall("outside", "ViewImage", [str(outside)])])
    assert [message["role"] for message in messages] == ["tool"]
    assert "refused" in messages[0]["content"]
    assert s.tool_records == []


def test_view_image_batch_returns_all_tool_results_before_observation(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "screen.png")
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda _text: None)
    calls = [
        ToolCall("image", "ViewImage", ["screen.png"]),
        ToolCall("read", "Read", [{"path": "notes.txt", "ranges": [[0, 1]]}]),
    ]

    messages = runner.run(calls)

    assert [message["role"] for message in messages] == ["tool", "tool", "user"]
    assert [message["tool_call_id"] for message in messages[:2]] == ["image", "read"]
    assert ImageInputs.is_tool_observation(messages[-1])
    assert runner.parallel_safe(calls[0]) is False


def test_view_image_observation_round_trips_all_provider_protocols(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "screen.png", size=(8, 6))
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda _text: None)
    call = ToolCall("image", "ViewImage", ["screen.png"])
    assistant = Agent.assistant_turn_message({}, [call], "")
    active = [assistant, *runner.run([call])]
    model = ModelClient(s)

    chat = model.chat_messages(active)
    assert [message["role"] for message in chat] == ["assistant", "tool", "user"]
    assert [part["type"] for part in chat[-1]["content"]] == ["image_url", "text"]
    assert TOOL_IMAGE_OBSERVATION_KEY not in chat[-1]

    responses = model.responses_input(active)
    assert [item.get("type", item.get("role")) for item in responses] == ["function_call", "function_call_output", "user"]
    assert [part["type"] for part in responses[-1]["content"]] == ["input_image", "input_text"]

    anthropic = model.anthropic_messages(active)
    assert [message["role"] for message in anthropic] == ["assistant", "user"]
    assert [part["type"] for part in anthropic[-1]["content"]] == ["tool_result", "image", "text"]


def test_agent_persists_view_image_observation_without_replaying_it_as_user_input(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "screen.png")
    agent = Agent(s, output_fn=lambda _text: None)

    class Model:
        def __init__(self):
            self.requests = []

        def request(self, messages, tools=None):
            self.requests.append(messages)
            if len(self.requests) == 1:
                return {}, [ToolCall("image", "ViewImage", ["screen.png"])], ""
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = Model()

    assert agent.run("inspect the screenshot") == "done"
    assert [message["role"] for message in s.messages] == ["user", "assistant", "tool", "user", "assistant"]
    observation = s.messages[-2]
    assert ImageInputs.is_tool_observation(observation)
    assert ContextManager(s).latest_user_index(s.messages) == 0
    assert ContextManager(s).history_title(s.messages) == "inspect the screenshot"

    rendered = []
    command_loop = CommandLoop(agent, output_fn=lambda _text: None)
    command_loop.ui.emit_answer = lambda *args, **kwargs: rendered.append((args, kwargs))
    command_loop.render_transcript_message(observation)
    assert rendered == []

    s.save_snapshot()
    restored = Session.load_snapshot(s.uid, config=s.config)
    restored_observation = next(message for message in restored.messages if ImageInputs.is_tool_observation(message))
    assert ImageInputs.is_tool_observation(restored_observation)
    assert restored.images.chat_content(restored_observation)[0]["type"] == "image_url"


def test_disabled_image_input_degrades_historical_messages_to_labels(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "past.png")
    message = s.images.message(s.images.recognize("review past.png"))
    s.config.provider.image_input = "off"

    assert s.images.chat_content(message) == "review [Image #1 · past.png]"
    assert s.images.responses_content(message) == "review [Image #1 · past.png]"
    assert s.images.anthropic_content(message) == "review [Image #1 · past.png]"


def test_text_only_image_refs_never_reenter_provider_projection(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "bridged.png")
    message = s.images.message(s.images.recognize("review bridged.png"))
    message[IMAGE_TEXT_ONLY_KEY] = True
    plain = {"role": "user", "content": message["content"]}

    # Even a later positive capability verdict or provider switch cannot silently add pixels to a
    # historical request whose bridge already replaced them with text.
    s.config.provider.image_input = "on"
    assert s.images.refs(message)
    assert s.images.input_refs(message) == ()
    assert s.images.chat_content(message) == message["content"]
    assert s.images.responses_content(message) == message["content"]
    assert s.images.anthropic_content(message) == message["content"]
    assert ContextManager(s).estimated_tokens([message]) == ContextManager(s).estimated_tokens([plain])
    assert not s.images.note_error([message], ModelError("Error code: 400 - image input is not supported"))


def test_successful_image_request_is_learned_per_provider_and_model(tmp_path, monkeypatch):
    s = session(tmp_path)
    image_file(tmp_path / "learn.png")
    message = s.images.message(s.images.recognize("learn.png"))
    model = ModelClient(s)
    monkeypatch.setattr(model, "api_request", lambda _messages, _tools: ({"role": "assistant", "content": "ok"}, [], "ok"))

    assert s.images.support() is None
    model.request([message], [])
    assert s.images.support() is True

    app = TuiApp(images=s.images)
    app.input_buffer.insert_text("learn.png ")
    assert app.status_fragments() == [("class:prompt", app.input_prompt)]
    assert app.input_error_fragments() == []

    s.config.provider.model = "another-model"
    assert s.images.support() is None
    assert app.status_fragments() == [("class:prompt", app.input_prompt)]
    assert app.input_error_fragments() == []


def test_only_explicit_image_unsupported_error_is_learned(tmp_path, monkeypatch):
    s = session(tmp_path)
    image_file(tmp_path / "reject.png")
    value = s.images.prepare(s.images.recognize("reject.png"))
    message = s.images.message(value)
    model = ModelClient(s)

    def reject(_messages, _tools):
        raise ModelError("Error code: 400 - Failed to deserialize messages[4]: unknown variant `image_url`, expected `text`")

    monkeypatch.setattr(model, "api_request", reject)
    with pytest.raises(ModelError) as caught:
        model.request([message], [])
    assert str(caught.value) == ("default/vision does not support image input. Switch to an image-capable model, or continue with image labels only.")
    assert "unknown variant `image_url`" in str(caught.value.__cause__)
    assert s.images.support() is False
    with pytest.raises(ModelError, match="Image input is disabled"):
        s.images.prepare(value)

    s.config.provider.model = "unrelated-error-model"

    def unrelated(_messages, _tools):
        raise ModelError("status code: 400 unknown variant `text`, expected `image_url`")

    monkeypatch.setattr(model, "api_request", unrelated)
    with pytest.raises(ModelError, match="expected `image_url`"):
        model.request([message], [])
    assert s.images.support() is None

    s.config.provider.model = "text-format-error-model"

    def text_format_error(_messages, _tools):
        raise ModelError("Error code: 400 - response_format only supports text input")

    monkeypatch.setattr(model, "api_request", text_format_error)
    with pytest.raises(ModelError, match="response_format only supports text input"):
        model.request([message], [])
    assert s.images.support() is None


def test_non_numeric_sdk_code_does_not_bypass_image_error_status_gate(tmp_path, monkeypatch):
    s = session(tmp_path)
    image_file(tmp_path / "denied.png")
    message = s.images.message(s.images.recognize("denied.png"))
    model = ModelClient(s)

    class ProviderError(Exception):
        code = "permission_error"

    def denied(_messages, _tools):
        try:
            raise ProviderError
        except ProviderError as cause:
            raise ModelError("Error code: 403 - vision is not enabled for this key") from cause

    monkeypatch.setattr(model, "api_request", denied)
    with pytest.raises(ModelError) as caught:
        model.request([message], [])

    assert str(caught.value) == "Error code: 403 - vision is not enabled for this key"
    assert s.images.support() is None


def test_timeout_value_is_not_mistaken_for_an_image_http_status(tmp_path):
    s = session(tmp_path)
    s.config.providers["vision"] = ProviderConfig(url="http://vision", key="test", model="vision")
    s.config.vision_provider = "vision"
    image_file(tmp_path / "slow.png")
    message = s.images.message(s.images.recognize("slow.png"))

    assert not s.images.can_retry_with_bridge(
        [message],
        ModelError("Model response exceeded provider.response_timeout=400s"),
    )


def test_error_is_not_reclassified_when_historical_images_are_degraded(tmp_path, monkeypatch):
    s = session(tmp_path)
    image_file(tmp_path / "history.png")
    message = s.images.message(s.images.recognize("history.png"))
    s.config.provider.image_input = "off"
    model = ModelClient(s)

    def reject(_messages, _tools):
        raise ModelError("Error code: 400 - vision modality is not supported for this deployment")

    monkeypatch.setattr(model, "api_request", reject)
    with pytest.raises(ModelError) as caught:
        model.request([message], [])

    assert str(caught.value) == "Error code: 400 - vision modality is not supported for this deployment"
    assert s.images.chat_content(message) == "[Image #1 · history.png]"


def test_anthropic_merges_text_mention_after_image_user_message(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "pixel.png", size=(1, 1))
    message = s.images.message(s.images.recognize("pixel.png"))

    converted = ModelClient(s).anthropic_messages([message, {"role": "user", "content": "mention context"}])

    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert [part["type"] for part in converted[0]["content"]] == ["image", "text", "text"]
    assert converted[0]["content"][-1] == {"type": "text", "text": "mention context"}


def test_chat_request_does_not_leak_internal_image_metadata(tmp_path, monkeypatch):
    s = session(tmp_path)
    image_file(tmp_path / "pixel.png", size=(1, 1))
    message = s.images.message(s.images.recognize("pixel.png"))
    captured = {}

    def create(**params):
        captured.update(params)
        response = SimpleNamespace(usage=None)
        response.choices = [SimpleNamespace(message=SimpleNamespace(role="assistant", content="ok", tool_calls=None))]
        return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None)
    monkeypatch.setattr(ModelClient, "client", lambda _self, **kwargs: client)

    ModelClient(s).chat_request([message], None)

    assert IMAGE_REFS_KEY not in json.dumps(captured)
    assert captured["messages"][0]["content"] == s.images.chat_content(message)


def test_context_estimates_image_from_dimensions_without_base64(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "large.png", size=(1024, 513))
    message = s.images.message(s.images.recognize("large.png"))
    plain = {"role": "user", "content": message["content"]}

    context = ContextManager(s)
    difference = context.estimated_tokens([message]) - context.estimated_tokens([plain])

    assert difference == 85 + 170 * 4
    assert difference < len(s.images.chat_content(message)[0]["image_url"]["url"]) // 4

    assert s.images.note_error([message], ModelError("Error code: 400 - image input is not supported")) is True
    assert context.estimated_tokens([message]) == context.estimated_tokens([plain])


def test_tui_replaces_image_path_with_atomic_label_and_keeps_history_readable(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "ui.png")
    received = []
    history = FileHistory(str(tmp_path / "history"))
    app = TuiApp(on_chat_submit=received.append, images=s.images, history=history)

    app.input_buffer.insert_text(f"inspect {path.name} ")

    assert app.input_buffer.text == f"inspect {IMAGE_MARKER} "
    assert app.input_images[0].name == "ui.png"
    assert app.status_fragments() == [("class:prompt", app.input_prompt)]
    assert app.input_error_fragments() == []
    app.input_buffer.validate_and_handle()
    assert received[0].display_text() == "inspect [Image #1 · ui.png] "
    assert list(history.load_history_strings())[-1] == "inspect ui.png "


def test_tui_reports_and_blocks_disabled_image_input_without_clearing_draft(tmp_path):
    s = session(tmp_path)
    s.config.provider.image_input = "off"
    path = image_file(tmp_path / "disabled.png")
    received = []
    app = TuiApp(on_chat_submit=received.append, images=s.images)

    app.input_buffer.insert_text(path.name + " ")

    assert app.status_fragments() == [("class:prompt", app.input_prompt)]
    assert app.input_error_fragments() == [("class:input.error", "Error: Image input is disabled for the active provider/model")]
    app.input_buffer.validate_and_handle()
    assert received == []
    assert app.input_buffer.text == IMAGE_MARKER + " "
    assert "Image input is disabled" in app.input_error


def test_tui_deleting_first_atomic_label_removes_the_matching_image(tmp_path):
    image_file(tmp_path / "first.png")
    image_file(tmp_path / "second.png", color=(65, 43, 21))
    app = TuiApp(image_cwd=str(tmp_path))
    app.input_buffer.insert_text("first.png second.png ")

    app.input_buffer.cursor_position = 1
    app.input_buffer.delete_before_cursor(1)

    assert [image.name for image in app.input_images] == ["second.png"]
    assert UserInput(app.input_buffer.text, app.input_images).display_text() == " [Image #1 · second.png] "


def test_image_label_processor_maps_the_whole_label_to_one_source_cell(tmp_path):
    path = image_file(tmp_path / "chip.png")
    value = ImageInputs(cwd=str(tmp_path)).recognize(path.name)
    processor = tui_module.ImageLabelProcessor(lambda: value.images)
    document = Document(str(value))
    transformation_input = SimpleNamespace(document=document, lineno=0, fragments=[("", str(value))])

    transformed = processor.apply_transformation(transformation_input)

    assert transformed.fragments == [("class:image.attachment", "[Image #1 · chip.png]")]
    assert transformed.source_to_display(1) == len("[Image #1 · chip.png]")
    assert transformed.display_to_source(10) == 1


def test_missing_recognized_image_keeps_tui_draft_on_submit(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "gone.png")
    received = []
    app = TuiApp(on_chat_submit=received.append, images=s.images)
    app.input_buffer.insert_text(path.name + " ")
    path.unlink()

    app.input_buffer.validate_and_handle()

    assert received == []
    assert app.input_buffer.text == IMAGE_MARKER + " "
    assert "Cannot read image" in app.input_error


def test_tmux_renders_image_error_above_inline_label_without_control_character(tmp_path):
    executable = shutil.which("tmux")
    if executable is None:
        return
    image_file(tmp_path / "tmux.png")
    probe = tmp_path / "image_tui_probe.py"
    probe.write_text(
        "from minacode.session import Session\n"
        "from minacode.tui import TuiApp\n"
        f"session = Session(cwd={str(tmp_path)!r})\n"
        'session.config.provider.image_input = "off"\n'
        "TuiApp(images=session.images).run()\n"
    )
    socket = "minacode-image-test-" + tmp_path.name
    command = [executable, "-L", socket]
    pane_command = f"{shlex.quote(sys.executable)} {shlex.quote(str(probe))}"
    try:
        subprocess.run([*command, "new-session", "-d", "-s", "probe", "-x", "100", "-y", "20", pane_command], check=True)
        time.sleep(0.1)
        subprocess.run([*command, "send-keys", "-t", "probe", "-l", "tmux.png "], check=True)
        deadline = time.monotonic() + 3
        screen = ""
        while "[Image #1 · tmux.png]" not in screen and time.monotonic() < deadline:
            time.sleep(0.02)
            screen = subprocess.run([*command, "capture-pane", "-p", "-t", "probe"], check=True, capture_output=True, text=True).stdout
        assert "[Image #1 · tmux.png]" in screen
        assert "Error: Image input is disabled for the active provider/model" in screen
        assert "^J" not in screen
        lines = screen.splitlines()
        error_line = next(index for index, line in enumerate(lines) if "Error: Image input" in line)
        prompt_line = next(index for index, line in enumerate(lines) if "> [Image #1" in line)
        assert prompt_line == error_line + 1
    finally:
        subprocess.run([*command, "kill-server"], check=False, capture_output=True)
