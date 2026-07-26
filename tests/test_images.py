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

import minacode as n


def image_file(path, *, size=(32, 24), image_format="PNG", color=(12, 34, 56)):
    Image.new("RGB", size, color).save(path, format=image_format)
    return path


def session(tmp_path):
    config = n.Config(data_dir=str(tmp_path / "data"))
    config.providers = {"default": n.ProviderConfig(url="http://test", key="test", model="vision")}
    return n.Session(cwd=str(tmp_path), config=config)


def test_recognize_local_image_paths_and_leave_other_tokens_alone(tmp_path):
    first = image_file(tmp_path / "one.png")
    image_file(tmp_path / "two words.webp", image_format="WEBP")

    value = n.ImageInputs(cwd=str(tmp_path)).recognize(f"review ({first.name}) and two\\ words.webp; missing.png stays")

    assert str(value).count(n.IMAGE_MARKER) == 2
    assert [image.name for image in value.images] == ["one.png", "two words.webp"]
    assert value.display_text() == "review ([Image #1 · one.png]) and [Image #2 · two words.webp]; missing.png stays"
    assert value.original_text() == f"review ({first.name}) and two\\ words.webp; missing.png stays"


def test_recognize_quoted_path_and_attach_duplicate_only_once(tmp_path):
    image_file(tmp_path / "same image.png")

    value = n.ImageInputs(cwd=str(tmp_path)).recognize("look at 'same image.png' and 'same image.png'")

    assert len(value.images) == 1
    assert value.original_text() == "look at 'same image.png' and 'same image.png'"


def test_animated_gif_is_not_recognized(tmp_path):
    path = tmp_path / "animated.gif"
    frames = [Image.new("RGB", (4, 4), color) for color in ("red", "blue")]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=10, loop=0)

    value = n.ImageInputs(cwd=str(tmp_path)).recognize(path.name)

    assert value == path.name
    assert value.images == ()


def test_session_stores_content_addressed_image_and_persists_refs(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "screen.png", size=(640, 480))
    value = s.images.recognize(f"describe {path.name}")

    message = s.images.message(value)
    s.messages.append(message)
    s.save_snapshot()

    image = n.ImageRef.from_json(message[n.IMAGE_REFS_KEY][0])
    assert image is not None
    asset = os.path.join(n.SessionSnapshotStore.project_dir(s.config.data_dir, s.cwd), s.uid + ".assets", image.ref)
    assert os.path.isfile(asset)
    assert open(asset, "rb").read() == path.read_bytes()

    restored = n.Session.load_snapshot(s.uid, config=s.config)
    assert restored.messages[0] == message
    assert n.ContextManager(restored).messages_text(restored.messages[:1]) == "user:\ndescribe [Image #1 · screen.png]"


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
    image = n.ImageRef.from_json(message[n.IMAGE_REFS_KEY][0])
    assert image is not None
    asset = os.path.join(n.SessionSnapshotStore.project_dir(s.config.data_dir, s.cwd), s.uid + ".assets", image.ref)
    os.unlink(asset)

    with pytest.raises(n.ModelError, match="Stored image is missing"):
        s.images.responses_content(message)


def test_session_queue_round_trips_images_and_garbage_collects_assets(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "queued.jpg", image_format="JPEG")
    value = s.images.prepare(s.images.recognize(path.name))
    s.enqueue_user_input(value)
    s.save_snapshot()

    restored = n.Session.load_snapshot(s.uid, config=s.config)
    queued = restored.pending_user_inputs[0]
    assert queued.text == "[Image #1 · queued.jpg]"
    assert queued.user_input().display_text() == queued.text

    assets = os.path.join(n.SessionSnapshotStore.project_dir(s.config.data_dir, s.cwd), s.uid + ".assets")
    assert os.path.isdir(assets)
    s.pending_user_inputs.clear()
    s.save_snapshot()
    assert not os.path.exists(assets)


def test_recalling_image_follow_up_keeps_asset_until_resubmission(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "recall.png")
    s.enqueue_user_input(s.images.prepare(s.images.recognize(path.name)))
    s.save_snapshot()
    command_loop = n.CommandLoop(n.Agent(s, output_fn=lambda _text: None), output_fn=lambda _text: None)

    recalled = command_loop.recall_pending_input(lambda: None)

    assert isinstance(recalled, n.UserInput)
    assert recalled.display_text() == "[Image #1 · recall.png]"
    assert s.images.chat_content(s.images.message(recalled))[0]["type"] == "image_url"


def test_expired_session_removes_its_image_assets(tmp_path):
    old = session(tmp_path)
    path = image_file(tmp_path / "expired.png")
    old.messages.append(old.images.message(old.images.recognize(path.name)))
    old.save_snapshot()
    log = n.SessionSnapshotStore.session_path(old.config.data_dir, old.cwd, old.uid)
    assets = log[: -len(".jsonl")] + ".assets"
    stale = time.time() - 3 * 86400
    os.utime(log, (stale, stale))
    current = session(tmp_path)
    current.settings.session_retention_days = 1

    assert n.SessionSnapshotStore.clean_expired(current) == 1
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

    model = n.ModelClient(s)
    assert model.responses_input([message]) == [{"role": "user", "content": s.images.responses_content(message)}]
    assert model.anthropic_messages([message]) == [{"role": "user", "content": s.images.anthropic_content(message)}]


def test_disabled_image_input_degrades_historical_messages_to_labels(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "past.png")
    message = s.images.message(s.images.recognize("review past.png"))
    s.config.provider.image_input = "off"

    assert s.images.chat_content(message) == "review [Image #1 · past.png]"
    assert s.images.responses_content(message) == "review [Image #1 · past.png]"
    assert s.images.anthropic_content(message) == "review [Image #1 · past.png]"


def test_successful_image_request_is_learned_per_provider_and_model(tmp_path, monkeypatch):
    s = session(tmp_path)
    image_file(tmp_path / "learn.png")
    message = s.images.message(s.images.recognize("learn.png"))
    model = n.ModelClient(s)
    monkeypatch.setattr(model, "api_request", lambda _messages, _tools: ({"role": "assistant", "content": "ok"}, [], "ok"))

    assert s.images.support() is None
    model.request([message], [])
    assert s.images.support() is True

    app = n.TuiApp(images=s.images)
    app.input_buffer.insert_text("learn.png ")
    assert app.status_fragments()[0] == ("class:input.notice", "Image attached · supported\n")

    s.config.provider.model = "another-model"
    assert s.images.support() is None
    assert app.status_fragments()[0] == ("class:input.notice", "Image attached · model capability unknown\n")


def test_only_explicit_image_unsupported_error_is_learned(tmp_path, monkeypatch):
    s = session(tmp_path)
    image_file(tmp_path / "reject.png")
    value = s.images.prepare(s.images.recognize("reject.png"))
    message = s.images.message(value)
    model = n.ModelClient(s)

    def reject(_messages, _tools):
        raise n.ModelError("status code: 400 image inputs are not supported")

    monkeypatch.setattr(model, "api_request", reject)
    with pytest.raises(n.ModelError, match="Active provider/model does not support image input"):
        model.request([message], [])
    assert s.images.support() is False
    with pytest.raises(n.ModelError, match="Image input is disabled"):
        s.images.prepare(value)

    s.config.provider.model = "unrelated-error-model"

    def unrelated(_messages, _tools):
        raise n.ModelError("status code: 400 invalid request")

    monkeypatch.setattr(model, "api_request", unrelated)
    with pytest.raises(n.ModelError, match="invalid request"):
        model.request([message], [])
    assert s.images.support() is None


def test_anthropic_merges_text_mention_after_image_user_message(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "pixel.png", size=(1, 1))
    message = s.images.message(s.images.recognize("pixel.png"))

    converted = n.ModelClient(s).anthropic_messages([message, {"role": "user", "content": "mention context"}])

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
    monkeypatch.setattr(n.ModelClient, "client", lambda _self: client)

    n.ModelClient(s).chat_request([message], None)

    assert n.IMAGE_REFS_KEY not in json.dumps(captured)
    assert captured["messages"][0]["content"] == s.images.chat_content(message)


def test_context_estimates_image_from_dimensions_without_base64(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "large.png", size=(1024, 513))
    message = s.images.message(s.images.recognize("large.png"))
    plain = {"role": "user", "content": message["content"]}

    difference = n.ContextManager.estimated_tokens([message]) - n.ContextManager.estimated_tokens([plain])

    assert difference == 85 + 170 * 4
    assert difference < len(s.images.chat_content(message)[0]["image_url"]["url"]) // 4


def test_tui_replaces_image_path_with_atomic_label_and_keeps_history_readable(tmp_path):
    s = session(tmp_path)
    path = image_file(tmp_path / "ui.png")
    received = []
    history = FileHistory(str(tmp_path / "history"))
    app = n.TuiApp(on_chat_submit=received.append, images=s.images, history=history)

    app.input_buffer.insert_text(f"inspect {path.name} ")

    assert app.input_buffer.text == f"inspect {n.IMAGE_MARKER} "
    assert app.input_images[0].name == "ui.png"
    assert app.status_fragments()[0] == ("class:input.notice", "Image attached · model capability unknown\n")
    app.input_buffer.validate_and_handle()
    assert received[0].display_text() == "inspect [Image #1 · ui.png] "
    assert list(history.load_history_strings())[-1] == "inspect ui.png "


def test_tui_reports_and_blocks_disabled_image_input_without_clearing_draft(tmp_path):
    s = session(tmp_path)
    s.config.provider.image_input = "off"
    path = image_file(tmp_path / "disabled.png")
    received = []
    app = n.TuiApp(on_chat_submit=received.append, images=s.images)

    app.input_buffer.insert_text(path.name + " ")

    assert app.status_fragments()[0] == ("class:input.error", "Error: Image input is disabled for the active provider/model\n")
    app.input_buffer.validate_and_handle()
    assert received == []
    assert app.input_buffer.text == n.IMAGE_MARKER + " "
    assert "Image input is disabled" in app.input_error


def test_tui_deleting_first_atomic_label_removes_the_matching_image(tmp_path):
    image_file(tmp_path / "first.png")
    image_file(tmp_path / "second.png", color=(65, 43, 21))
    app = n.TuiApp(image_cwd=str(tmp_path))
    app.input_buffer.insert_text("first.png second.png ")

    app.input_buffer.cursor_position = 1
    app.input_buffer.delete_before_cursor(1)

    assert [image.name for image in app.input_images] == ["second.png"]
    assert n.UserInput(app.input_buffer.text, app.input_images).display_text() == " [Image #1 · second.png] "


def test_image_label_processor_maps_the_whole_label_to_one_source_cell(tmp_path):
    path = image_file(tmp_path / "chip.png")
    value = n.ImageInputs(cwd=str(tmp_path)).recognize(path.name)
    processor = n.tui.ImageLabelProcessor(lambda: value.images)
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
    app = n.TuiApp(on_chat_submit=received.append, images=s.images)
    app.input_buffer.insert_text(path.name + " ")
    path.unlink()

    app.input_buffer.validate_and_handle()

    assert received == []
    assert app.input_buffer.text == n.IMAGE_MARKER + " "
    assert "Cannot read image" in app.input_error


def test_tmux_renders_recognized_image_as_inline_label(tmp_path):
    executable = shutil.which("tmux")
    if executable is None:
        return
    image_file(tmp_path / "tmux.png")
    probe = tmp_path / "image_tui_probe.py"
    probe.write_text(f"import minacode as n\nn.TuiApp(image_cwd={str(tmp_path)!r}).run()\n")
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
    finally:
        subprocess.run([*command, "kill-server"], check=False, capture_output=True)
