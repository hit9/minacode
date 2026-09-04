import base64
import json
import os
import time
from types import SimpleNamespace

import pytest
from model_harness import async_create
from PIL import Image
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory

import wizolt.cli.loop as loop_module
import wizolt.tui as tui_module
from wizolt.base import ModelError, ToolCall, ToolError
from wizolt.cli import CommandLoop
from wizolt.config import (
    Config,
    ProviderConfig,
)
from wizolt.context import ContextManager
from wizolt.engine import Agent
from wizolt.image import (
    IMAGE_MARKER,
    IMAGE_REFS_KEY,
    IMAGE_TEXT_ONLY_KEY,
    TOOL_IMAGE_OBSERVATION_KEY,
    TOOL_IMAGE_OBSERVATION_PREFIX,
    ImageInputs,
    ImageRef,
    UserInput,
)
from wizolt.model import ModelClient
from wizolt.runner import ToolRunner
from wizolt.session import Session, SessionSnapshotStore
from wizolt.tools import ViewImageTool
from wizolt.tui import TuiApp


async def _no_close():
    """The async close the provider SDK clients expose; awaited by ModelClient.call_client."""


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


async def test_session_stores_content_addressed_image_and_persists_refs(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "screen.png", size=(640, 480))
    value = s.images.recognize(f"describe {path.name}")

    message = s.images.message(value)
    s.messages.append(message)
    await s.save_snapshot()

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


async def test_session_queue_round_trips_images_and_garbage_collects_assets(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "queued.jpg", image_format="JPEG")
    value = s.images.prepare(s.images.recognize(path.name))
    s.enqueue_user_input(value)
    await s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config)
    queued = restored.pending_user_inputs[0]
    assert queued.text == "[Image #1 · queued.jpg]"
    assert queued.user_input().display_text() == queued.text

    assets = os.path.join(SessionSnapshotStore.project_dir(s.config.data_dir, s.cwd), s.uid + ".assets")
    assert os.path.isdir(assets)
    s.pending_user_inputs.clear()
    await s.save_snapshot()
    assert not os.path.exists(assets)


async def test_garbage_collect_spares_dotfile_staging_and_drops_stray_files(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "kept.png")
    s.messages.append(s.images.message(s.images.recognize(path.name)))
    await s.save_snapshot()

    assets = os.path.join(SessionSnapshotStore.project_dir(s.config.data_dir, s.cwd), s.uid + ".assets")
    staged = os.path.join(assets, ".image-93e8u1vn")
    stray = os.path.join(assets, "orphan-deadbeef")
    with open(staged, "w") as file:
        file.write("staging")
    with open(stray, "w") as file:
        file.write("stray")

    await s.save_snapshot()

    assert os.path.isfile(staged)  # .image-* staging from ImageInputs._store survives GC
    assert not os.path.exists(stray)  # unreferenced non-dotfile is still collected
    assert os.listdir(assets)  # the referenced asset is kept


async def test_garbage_collect_clears_stale_image_staging_and_spares_fresh(tmp_path):
    """GC clears a .image-* staging file left by a crashed save, but spares one that could still
    be inside the copy+replace window: residue cannot pile up (or block the assets directory's
    removal) without racing a real write."""
    s = session(tmp_path)
    path = image_file(tmp_path / "kept.png")
    s.messages.append(s.images.message(s.images.recognize(path.name)))
    await s.save_snapshot()

    assets = os.path.join(SessionSnapshotStore.project_dir(s.config.data_dir, s.cwd), s.uid + ".assets")
    stale = os.path.join(assets, ".image-93e8u1vn")
    fresh = os.path.join(assets, ".image-93e8u1vo")
    with open(stale, "w") as file:
        file.write("residue")
    with open(fresh, "w") as file:
        file.write("in flight")
    os.utime(stale, (time.time() - 3600, time.time() - 3600))

    await s.save_snapshot()

    assert not os.path.exists(stale)  # crash residue is collected
    assert os.path.isfile(fresh)  # a staging file inside the copy+replace window is spared
    assert any(name for name in os.listdir(assets) if not name.startswith("."))  # the referenced asset is kept


async def test_store_survives_snapshot_gc_between_mkstemp_and_replace(tmp_path, monkeypatch):
    s = session(tmp_path)
    settled = image_file(tmp_path / "settled.png", color=(1, 2, 3))
    s.messages.append(s.images.message(s.images.recognize(settled.name)))
    await s.save_snapshot()  # a real snapshot, so the save during the race reaches garbage collection

    racing = image_file(tmp_path / "racing.png", color=(4, 5, 6))
    value = s.images.recognize(racing.name)

    real_replace = os.replace
    intercepted = {"hit": False}

    def racing_replace(source, destination):
        if not intercepted["hit"] and os.path.basename(source).startswith(".image-"):
            intercepted["hit"] = True
            # The snapshot worker's asset collector runs while the staging file is unreferenced --
            # exactly the window this guards. Driven directly rather than through `save_snapshot`
            # because the interception point is a synchronous callback, which is also where the
            # real race comes from: a worker collecting assets under a staging replace.
            racing_plan = SessionSnapshotStore(s).plan()
            assert racing_plan is not None
            racing_plan.execute()
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", racing_replace)

    stored = s.images.prepare(value)

    assert intercepted["hit"] is True  # the race was actually exercised
    assert os.path.isfile(os.path.join(s.images.assets_dir(), stored.images[0].ref))
    with open(os.path.join(s.images.assets_dir(), stored.images[0].ref), "rb") as file:
        assert file.read() == racing.read_bytes()


async def test_recalling_image_follow_up_keeps_asset_until_resubmission(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "recall.png")
    s.enqueue_user_input(s.images.prepare(s.images.recognize(path.name)))
    await s.save_snapshot()
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

    async def run(value):
        received.append(value)
        return "done"

    monkeypatch.setattr(agent, "run", run)
    monkeypatch.setattr(loop_module.UpdateChecker, "load_cached", lambda _self: False)

    def eof(_prompt):
        raise EOFError

    command_loop = CommandLoop(agent, input_fn=eof, output_fn=lambda _text: None)

    assert command_loop.run() == 0
    assert len(received) == 1
    assert isinstance(received[0], UserInput)
    assert [image.name for image in received[0].images] == ["queued.png"]


async def test_expired_session_removes_its_image_assets(tmp_path):
    old = session(tmp_path)
    path = image_file(tmp_path / "expired.png")
    old.messages.append(old.images.message(old.images.recognize(path.name)))
    await old.save_snapshot()
    log = SessionSnapshotStore.session_path(old.config.data_dir, old.cwd, old.uid)
    assets = log[: -len(".jsonl")] + ".assets"
    stale = time.time() - 3 * 86400
    os.utime(log, (stale, stale))
    current = session(tmp_path)
    current.settings.session_retention_days = 1

    assert SessionSnapshotStore.clean_expired(current.config.data_dir, current.uid, current.settings.session_retention_days) == 1
    assert not os.path.exists(log)
    assert not os.path.exists(assets)


def test_protocol_payloads_use_each_standard_image_shape(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "pixel.png", size=(1, 1))
    message = s.images.message(s.images.recognize("what is this? pixel.png"))
    encoded = base64.b64encode(path.read_bytes()).decode()
    data_url = "data:image/png;base64," + encoded
    [image] = s.images.refs(message)
    model_text = (
        "what is this? [Image #1 · pixel.png]\n\n"
        f'[Attached image assets]\n- {{"image": 1, "name": "pixel.png", "path": {json.dumps(s.images.asset_path(image))}}}'
    )

    assert s.images.chat_content(message) == [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": model_text},
    ]
    assert s.images.responses_content(message) == [
        {"type": "input_image", "image_url": data_url},
        {"type": "input_text", "text": model_text},
    ]
    assert s.images.anthropic_content(message) == [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}},
        {"type": "text", "text": model_text},
    ]

    model = ModelClient(s)
    assert model.wire(ProviderConfig(api="responses", model="gpt-5")).messages([message]) == [{"role": "user", "content": s.images.responses_content(message)}]
    assert model.wire(ProviderConfig(api="anthropic", model="claude")).messages([message]) == [{"role": "user", "content": s.images.anthropic_content(message)}]


async def test_view_image_tool_validates_stores_and_builds_model_observation(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "screen shot.png", size=(640, 480))
    tool = ViewImageTool(s, [path.name, "read the error"])

    output = await tool.call()
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


async def test_view_image_tool_rejects_invalid_input(tmp_path):
    s = session(tmp_path)
    (tmp_path / "not-image.png").write_text("not pixels", encoding="utf-8")

    with pytest.raises(ToolError, match="Cannot read image"):
        await ViewImageTool(s, ["not-image.png"]).call()


async def test_view_image_tool_requires_confirmation_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = image_file(tmp_path / "outside.png")
    s = Session(cwd=str(workspace), config=Config(data_dir=str(tmp_path / "data")))
    tool = ViewImageTool(s, [str(outside)])
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda _prompt: "no", output_fn=lambda _text: None)

    assert tool.needs_confirmation() is True
    messages = await runner.run([ToolCall("outside", "ViewImage", [str(outside)])])
    assert [message["role"] for message in messages] == ["tool"]
    assert "refused" in messages[0]["content"]
    assert s.tool_records == []


async def test_view_image_batch_returns_all_tool_results_before_observation(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "screen.png")
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda _text: None)
    calls = [
        ToolCall("image", "ViewImage", ["screen.png"]),
        ToolCall("read", "Read", [{"path": "notes.txt", "ranges": [[0, 1]]}]),
    ]

    messages = await runner.run(calls)

    assert [message["role"] for message in messages] == ["tool", "tool", "user"]
    assert [message["tool_call_id"] for message in messages[:2]] == ["image", "read"]
    assert ImageInputs.is_tool_observation(messages[-1])
    assert runner.parallel_safe(calls[0]) is False


async def test_view_image_observation_round_trips_all_provider_protocols(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "screen.png", size=(8, 6))
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda _text: None)
    call = ToolCall("image", "ViewImage", ["screen.png"])
    assistant = Agent.assistant_turn_message({}, [call], "")
    active = [assistant, *await runner.run([call])]
    model = ModelClient(s)

    chat = model.wire(model.session.config.provider).messages(active)
    assert [message["role"] for message in chat] == ["assistant", "tool", "user"]
    assert [part["type"] for part in chat[-1]["content"]] == ["image_url", "text"]
    assert TOOL_IMAGE_OBSERVATION_KEY not in chat[-1]

    responses = model.wire(ProviderConfig(api="responses", model="gpt-5")).messages(active)
    assert [item.get("type", item.get("role")) for item in responses] == ["function_call", "function_call_output", "user"]
    assert [part["type"] for part in responses[-1]["content"]] == ["input_image", "input_text"]

    anthropic = model.wire(ProviderConfig(api="anthropic", model="claude")).messages(active)
    assert [message["role"] for message in anthropic] == ["assistant", "user"]
    assert [part["type"] for part in anthropic[-1]["content"]] == ["tool_result", "image", "text"]


async def test_agent_persists_view_image_observation_without_replaying_it_as_user_input(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "screen.png")
    agent = Agent(s, output_fn=lambda _text: None)

    class Model:
        def __init__(self):
            self.requests = []

        async def request(self, messages, tools=None):
            self.requests.append(messages)
            if len(self.requests) == 1:
                return {}, [ToolCall("image", "ViewImage", ["screen.png"])], ""
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = Model()

    assert await agent.run("inspect the screenshot") == "done"
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

    await s.save_snapshot()
    restored = Session.load_snapshot(s.uid, config=s.config)
    restored_observation = next(message for message in restored.messages if ImageInputs.is_tool_observation(message))
    assert ImageInputs.is_tool_observation(restored_observation)
    assert restored.images.chat_content(restored_observation)[0]["type"] == "image_url"


def test_text_only_image_refs_never_reenter_provider_projection(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "bridged.png")
    message = s.images.message(s.images.recognize("review bridged.png"))
    message[IMAGE_TEXT_ONLY_KEY] = True
    plain = {"role": "user", "content": message["content"]}

    # A settled failed occurrence stays text-only even if the provider changes later.
    assert s.images.refs(message)
    assert s.images.input_refs(message) == ()
    assert s.images.chat_content(message) == message["content"]
    assert s.images.responses_content(message) == message["content"]
    assert s.images.anthropic_content(message) == message["content"]
    assert ContextManager(s).estimated_tokens([message]) == ContextManager(s).estimated_tokens([plain])


def test_anthropic_merges_text_mention_after_image_user_message(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "pixel.png", size=(1, 1))
    message = s.images.message(s.images.recognize("pixel.png"))

    converted = ModelClient(s).wire(ProviderConfig(api="anthropic", model="claude")).messages([message, {"role": "user", "content": "mention context"}])

    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert [part["type"] for part in converted[0]["content"]] == ["image", "text", "text"]
    assert converted[0]["content"][-1] == {"type": "text", "text": "mention context"}


async def test_chat_request_does_not_leak_internal_image_metadata(tmp_path, monkeypatch):
    s = session(tmp_path)
    image_file(tmp_path / "pixel.png", size=(1, 1))
    message = s.images.message(s.images.recognize("pixel.png"))
    captured = {}

    def create(**params):
        captured.update(params)
        response = SimpleNamespace(usage=None)
        response.choices = [SimpleNamespace(message=SimpleNamespace(role="assistant", content="ok", tool_calls=None))]
        return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=async_create(create))), close=_no_close)
    monkeypatch.setattr(ModelClient, "client", lambda _self, **kwargs: client)

    await ModelClient(s).request([message], None)

    assert IMAGE_REFS_KEY not in json.dumps(captured)
    assert captured["messages"][0]["content"] == s.images.chat_content(message)


def test_context_estimates_image_from_dimensions_without_base64(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "large.png", size=(1024, 513))
    message = s.images.message(s.images.recognize("large.png"))
    plain = {"role": "user", "content": message["content"]}

    context = ContextManager(s)
    difference = context.estimated_tokens([message]) - context.estimated_tokens([plain])

    [image] = s.images.refs(message)
    asset_tokens = (len("\n\n" + s.images.asset_context((image,))) + 3) // 4
    assert difference == 85 + 170 * 4 + asset_tokens
    assert difference < len(s.images.chat_content(message)[0]["image_url"]["url"]) // 4

    s.images.settle_failed_messages([message])
    settled_plain = {"role": "user", "content": message["content"]}
    assert context.estimated_tokens([message]) == context.estimated_tokens([settled_plain])


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


def test_tui_submits_images_without_a_capability_precheck(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "disabled.png")
    received = []
    app = TuiApp(on_chat_submit=received.append, images=s.images)

    app.input_buffer.insert_text(path.name + " ")

    assert app.status_fragments() == [("class:prompt", app.input_prompt)]
    assert app.input_error_fragments() == []
    app.input_buffer.validate_and_handle()
    assert received and received[0].images
    assert app.input_buffer.text == ""


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


def test_tui_recognition_is_instant_but_submission_no_longer_stores_on_the_callback(tmp_path):
    """A submission callback only recognizes: typing an image path still becomes a marker at once,
    and submitting hands the unstored draft to the caller -- the runtime's admission step owns the
    copy, not the keystroke callback."""
    s = session(tmp_path)
    path = image_file(tmp_path / "gone.png")
    received = []
    app = TuiApp(on_chat_submit=received.append, images=s.images)
    app.input_buffer.insert_text(path.name + " ")

    app.input_buffer.validate_and_handle()

    assert len(received) == 1 and received[0].images and received[0].images[0].source_path != ""
    assert app.input_buffer.text == ""  # cleared for the next draft
    assert app.input_error == ""


def test_tui_restore_submission_puts_the_refused_draft_back(tmp_path):
    """A failed admission hands the draft back to the editor with the error, when the buffer is
    still empty."""
    s = session(tmp_path)
    app = TuiApp(images=s.images)
    app.invalidate = lambda: None
    app._reset_input("")

    app.restore_submission("describe [Image #1 \u00b7 gone.png]", "Cannot read image gone.png")

    assert app.input_buffer.text == "describe [Image #1 \u00b7 gone.png]"
    assert "Cannot read image" in app.input_error


def test_tui_restore_submission_never_overwrites_a_new_draft(tmp_path):
    """While the user is already typing the next line, a refused draft is reported but not forced
    back over their text."""
    s = session(tmp_path)
    app = TuiApp(images=s.images)
    app.invalidate = lambda: None
    app._reset_input("")
    app.input_buffer.insert_text("next line")

    app.restore_submission("old draft", "refused")

    assert app.input_buffer.text == "next line"
    assert app.input_error == "refused"


async def test_admission_refuses_an_image_removed_after_recognition_without_touching_the_draft(tmp_path):
    """The runtime's admission step re-validates the recognized source off the loop; a file deleted
    between recognition and submission refuses the draft, stores nothing, and leaves the draft's
    reference intact for the editor."""
    s = session(tmp_path)
    path = image_file(tmp_path / "gone.png")
    value = s.images.recognize(path.name)  # file present: recognized and referenced
    assert value.images and value.images[0].source_path
    path.unlink()

    with pytest.raises(ModelError, match="Cannot read image"):
        await s.images.admit(value)

    assert not os.path.exists(os.path.join(s.images.assets_dir(), value.images[0].ref))
    assert value.images[0].source_path == str(path)


async def test_load_payloads_reads_each_distinct_asset_once_and_matches_direct_wire(tmp_path, monkeypatch):
    """A request loads every distinct image ref exactly once, and the wire parts built from those
    payloads are byte-identical to the per-image reads they replace."""
    s = session(tmp_path)
    stored = []
    for name in ("a.png", "b.png"):
        value = s.images.recognize(image_file(tmp_path / name).name)
        stored.append(s.images.prepare(value).images[0])
    first, second = stored
    messages = [
        {"role": "user", "content": "one", IMAGE_REFS_KEY: [first.to_json()]},
        {"role": "user", "content": "two", IMAGE_REFS_KEY: [first.to_json(), second.to_json()]},
    ]
    reads: list[str] = []
    real_bytes = ImageInputs._bytes

    def counting_bytes(self, image, *, payloads=None):
        reads.append(image.ref)
        return real_bytes(self, image, payloads=payloads)

    monkeypatch.setattr(ImageInputs, "_bytes", counting_bytes)
    payloads = await s.images.load_payloads(messages)

    assert sorted(payloads) == sorted({first.ref, second.ref})
    assert sorted(reads) == sorted({first.ref, second.ref})  # the shared ref was read once
    assert s.images.chat_content(messages[0], payloads=payloads) == s.images.chat_content(messages[0])


async def test_load_payloads_rejects_corrupt_and_missing_assets(tmp_path):
    """Payload loading keeps the domain error of a per-image read: a corrupt or deleted asset
    names itself instead of failing downstream with a provider error."""
    s = session(tmp_path)
    value = s.images.recognize(image_file(tmp_path / "x.png").name)
    image = s.images.prepare(value).images[0]
    message = {"role": "user", "content": "x", IMAGE_REFS_KEY: [image.to_json()]}
    asset = os.path.join(s.images.assets_dir(), image.ref)

    with open(asset, "wb") as file:
        file.write(b"tampered")
    with pytest.raises(ModelError, match="corrupt"):
        await s.images.load_payloads([message])

    os.unlink(asset)
    with pytest.raises(ModelError, match="missing"):
        await s.images.load_payloads([message])


async def test_load_payloads_skips_text_only_refs(tmp_path):
    """A text-only route never sends raw blocks, so its refs are not loaded into payloads."""
    s = session(tmp_path)
    value = s.images.recognize(image_file(tmp_path / "x.png").name)
    image = s.images.prepare(value).images[0]
    message = {"role": "user", "content": "x", IMAGE_REFS_KEY: [image.to_json()], IMAGE_TEXT_ONLY_KEY: True}

    assert await s.images.load_payloads([message]) == {}


async def test_admission_copy_blocked_still_lets_the_loop_advance(tmp_path, monkeypatch):
    """Admission runs the image copy on the executor, so a slow disk cannot stall the loop while a
    submitted image is being stored."""
    import asyncio
    import shutil
    import threading

    s = session(tmp_path)
    path = image_file(tmp_path / "slow.png", size=(64, 64))
    value = s.images.recognize(path.name)
    entered, release = threading.Event(), threading.Event()
    real_copyfile = shutil.copyfile

    def slow_copy(source, destination):
        entered.set()
        release.wait(5)
        return real_copyfile(source, destination)

    monkeypatch.setattr(shutil, "copyfile", slow_copy)

    beats = 0

    async def heartbeat():
        nonlocal beats
        while True:
            beats += 1
            await asyncio.sleep(0.001)

    pulse = asyncio.create_task(heartbeat())
    admitting = asyncio.create_task(s.images.admit(value))
    await asyncio.to_thread(entered.wait, 5)
    await asyncio.sleep(0.02)
    assert beats > 0, "the loop stalled behind an image admission copy"
    release.set()
    stored = await admitting
    pulse.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pulse
    assert stored.images and not stored.images[0].source_path


async def test_cancelling_admission_quiesces_and_leaves_no_staging_residue(tmp_path, monkeypatch):
    """A cancelled admission waits for its copy worker (run_blocking) and leaves no `.image-*`
    staging file behind; the content-addressed asset either exists complete or not at all."""
    import asyncio
    import shutil
    import threading

    s = session(tmp_path)
    path = image_file(tmp_path / "slow.png", size=(64, 64))
    value = s.images.recognize(path.name)
    entered, release = threading.Event(), threading.Event()
    real_copyfile = shutil.copyfile

    def slow_copy(source, destination):
        entered.set()
        release.wait(5)
        return real_copyfile(source, destination)

    monkeypatch.setattr(shutil, "copyfile", slow_copy)
    admitting = asyncio.create_task(s.images.admit(value))
    await asyncio.to_thread(entered.wait, 5)
    admitting.cancel()
    await asyncio.sleep(0.05)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await admitting

    assets = s.images.assets_dir()
    assert not any(name.startswith(".image-") for name in os.listdir(assets))  # staging cleaned up
