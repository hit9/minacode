"""Attachment vision bridge: images pasted or referenced in a user message get one automatic
observation pass when the main model cannot see, mirroring ViewImage's bridge routing.

Black-box acceptance per the bridge spec: the routing rule is identical to ViewImage
(configured [vision] entry and `images.support() is not True`); one request carries all
attached images; the observation is injected into the user message as plain text and the
message carries no IMAGE_REFS_KEY, so the main model never sees image blocks yet can still
ViewImage the stored assets for follow-up detail.
"""

import json
import os

import pytest
from PIL import Image

from minacode.base import ModelError
from minacode.config import Config, ProviderConfig
from minacode.engine import Agent
from minacode.image import (
    ATTACHMENT_VISION_OBSERVATION_PREFIX,
    IMAGE_REFS_KEY,
)
from minacode.model import ModelClient
from minacode.prompts import VISION_OBSERVE_DEFAULT_QUESTION
from minacode.session import Session, SessionSnapshotStore
from minacode.tools import ViewImageTool
from minacode.tui import TuiApp

OBSERVATION_TEXT = "The screenshot shows a terminal with a red error line."


def image_file(path, *, size=(32, 24), image_format="PNG", color=(12, 34, 56)):
    Image.new("RGB", size, color).save(path, format=image_format)
    return path


def session(tmp_path, *, image_input="auto", vision=True, api="chat"):
    config = Config(data_dir=str(tmp_path / "data"))
    config.providers = {"default": ProviderConfig(url="http://main.test", key="key", model="main-model")}
    if vision:
        config.providers["v"] = ProviderConfig(url="http://vision.test", key="vkey", model="vision-model", api=api)
        config.vision_provider = "v"
    config.provider.image_input = image_input
    return Session(cwd=str(tmp_path), config=config)


def bridged_agent(s, monkeypatch, *, observation=OBSERVATION_TEXT):
    """Wire up an agent whose main-model request is faked (recorded, answers immediately) and
    whose vision requests are captured; the network is never touched."""

    calls = {"vision": [], "main": []}

    def fake_api_request(self, messages, tools, *, allow_stream=True, response_timeout=None, provider=None, json_object=False):
        calls["vision"].append((messages, provider, allow_stream))
        return {"role": "assistant", "content": observation}, [], observation

    def fake_main_request(self, messages, tools=None):
        calls["main"].append(messages)
        return {"role": "assistant", "content": "done"}, [], "done"

    monkeypatch.setattr(ModelClient, "api_request", fake_api_request)
    monkeypatch.setattr(ModelClient, "request", fake_main_request)
    return Agent(s, output_fn=lambda _text: None), calls


# --- 桥接注入（验收标准 1、4）---


def test_attachment_bridge_injects_observation_and_strips_image_refs(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off")
    assert s.images.support() is False
    shot = image_file(tmp_path / "shot.png")
    agent, calls = bridged_agent(s, monkeypatch)

    agent.run(s.images.recognize(f"看看 {shot.name} 里的报错"))

    # One vision request: image block + the user's own words as the question.
    assert len(calls["vision"]) == 1
    vision_messages, provider, allow_stream = calls["vision"][0]
    assert provider is s.config.providers["v"]
    assert allow_stream is False
    vision_user = vision_messages[1]
    assert [part["type"] for part in vision_user["content"]] == ["image_url", "text"]
    assert vision_user["content"][-1]["text"] == f"看看 {shot.name} 里的报错"
    assert IMAGE_REFS_KEY not in json.dumps(vision_messages)

    # The main-model user message is plain text: original words kept, observation appended,
    # no IMAGE_REFS_KEY on any request of the whole turn.
    assert len(calls["main"]) == 1
    assert not any(IMAGE_REFS_KEY in message for message in calls["main"][0])
    user = s.messages[0]
    assert user["role"] == "user"
    assert IMAGE_REFS_KEY not in user
    # The image renders as its inline label (as on the direct path), the words are kept, and the
    # observation block follows after a blank line.
    assert "看看 [Image #1 · shot.png] 里的报错" in user["content"].split("\n\n")[0]
    assert f'{ATTACHMENT_VISION_OBSERVATION_PREFIX} vision="v/vision-model"' in user["content"]
    assert OBSERVATION_TEXT in user["content"]

    # The wire projection carries no image content block either.
    projected = ModelClient(s).chat_messages(calls["main"][0])
    assert not any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"]) for m in projected)


def test_attachment_bridge_uses_default_question_for_image_only_input(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off")
    shot = image_file(tmp_path / "shot.png")
    agent, calls = bridged_agent(s, monkeypatch)

    agent.run(s.images.recognize(shot.name))  # path only, no words

    assert len(calls["vision"]) == 1
    text_block = calls["vision"][0][0][1]["content"][-1]
    assert text_block["text"] == VISION_OBSERVE_DEFAULT_QUESTION


# --- 多图一次请求（验收标准 2）---


def test_attachment_bridge_sends_all_images_in_one_request(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off")
    first = image_file(tmp_path / "first.png")
    second = image_file(tmp_path / "second.png", color=(65, 43, 21))
    agent, calls = bridged_agent(s, monkeypatch)

    agent.run(s.images.recognize(f"{first.name} 和 {second.name} 对比一下"))

    assert len(calls["vision"]) == 1
    content = calls["vision"][0][0][1]["content"]
    assert [part["type"] for part in content] == ["image_url", "image_url", "text"]
    assert content[-1]["text"] == f"{first.name} 和 {second.name} 对比一下"
    # Both file names surface in the injected block header.
    user = s.messages[0]
    assert "[Image #1 · first.png]" in user["content"]
    assert "[Image #2 · second.png]" in user["content"]


# --- 追问闭环（验收标准 4）---


def test_attachment_bridge_persists_assets_for_view_image_followup(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off")
    shot = image_file(tmp_path / "shot.png")
    agent, calls = bridged_agent(s, monkeypatch)

    agent.run(s.images.recognize(shot.name))

    # The image is stored in the session's assets, named by the observation block header.
    assets = os.path.join(SessionSnapshotStore.project_dir(s.config.data_dir, s.cwd), s.uid + ".assets")
    assert os.path.isdir(assets)
    assert os.listdir(assets)
    assert "[Image #1 · shot.png]" in s.messages[0]["content"]

    # ViewImage on the same file rides the same bridge for a follow-up question.
    output = ViewImageTool(s, [shot.name, "报错原文是什么"]).call()
    assert 'vision="v/vision-model"' in output.splitlines()[0]
    assert OBSERVATION_TEXT in output
    assert len(calls["vision"]) == 2


# --- 不桥接的两态不变（验收标准 5）---


def test_attachment_stays_inline_when_main_model_supports_images(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="on")
    assert s.images.support() is True
    shot = image_file(tmp_path / "shot.png")
    agent, calls = bridged_agent(s, monkeypatch)

    agent.run(s.images.recognize(f"看看 {shot.name}"))

    assert calls["vision"] == []
    user = s.messages[0]
    assert IMAGE_REFS_KEY in user
    assert s.images.chat_content(user)[0]["type"] == "image_url"
    assert ATTACHMENT_VISION_OBSERVATION_PREFIX not in user["content"]


def test_attachment_stays_disabled_without_vision_config(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off", vision=False)
    shot = image_file(tmp_path / "shot.png")
    agent, calls = bridged_agent(s, monkeypatch)

    with pytest.raises(ModelError, match="Image input is disabled"):
        agent.run(s.images.recognize(shot.name))

    assert calls["vision"] == []


# --- vision 失败可定位（验收标准 6）---


def test_attachment_vision_failure_raises_model_error_naming_entry(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off")
    shot = image_file(tmp_path / "shot.png")

    def failing(self, messages, tools, **kwargs):
        raise ModelError("upstream 502")

    monkeypatch.setattr(ModelClient, "api_request", failing)
    monkeypatch.setattr(ModelClient, "request", lambda self, messages, tools=None: ({"role": "assistant", "content": "done"}, [], "done"))

    with pytest.raises(ModelError) as caught:
        Agent(s, output_fn=lambda _text: None).run(s.images.recognize(shot.name))

    message = str(caught.value)
    assert "[vision]" in message
    assert "`v`" in message
    assert "upstream 502" in message


# --- UI 输入区预检（验收标准 7）---


def test_tui_precheck_still_blocks_without_vision(tmp_path):
    s = session(tmp_path, image_input="off", vision=False)
    shot = image_file(tmp_path / "shot.png")
    app = TuiApp(images=s.images)

    app.input_buffer.insert_text(shot.name + " ")

    assert app.input_error_fragments() == [("class:input.error", "Error: Image input is disabled for the active provider/model")]


def test_tui_precheck_suppresses_error_when_vision_is_configured(tmp_path):
    s = session(tmp_path, image_input="off", vision=True)
    shot = image_file(tmp_path / "shot.png")
    app = TuiApp(images=s.images)

    app.input_buffer.insert_text(shot.name + " ")

    assert app.input_error_fragments() == []


def test_tui_submit_allows_bridged_attachment(tmp_path):
    s = session(tmp_path, image_input="off", vision=True)
    shot = image_file(tmp_path / "shot.png")
    received = []
    app = TuiApp(on_chat_submit=received.append, images=s.images)

    app.input_buffer.insert_text(shot.name + " ")
    app.input_buffer.validate_and_handle()

    assert len(received) == 1
    assert received[0].images


# --- 回合中途排队的带图输入（live follow-up）---


def test_queued_image_followup_is_observed_not_inlined(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off")
    shot = image_file(tmp_path / "shot.png")
    agent, calls = bridged_agent(s, monkeypatch)

    queued = s.images.recognize(f"second look {shot.name}")
    s.enqueue_user_input(queued)  # must not raise on a bridging session
    assert len(s.pending_user_inputs) == 1

    agent.run("continue")

    # One observation request for the queued image, never an inline image block for the main model.
    assert len(calls["vision"]) == 1
    assert not any(IMAGE_REFS_KEY in message for message in calls["main"][0])
    followup = next(message for message in calls["main"][0] if OBSERVATION_TEXT in str(message.get("content")))
    assert followup["role"] == "user"
    assert "second look" in followup["content"]
    assert f'{ATTACHMENT_VISION_OBSERVATION_PREFIX} vision="v/vision-model"' in followup["content"]
    assert OBSERVATION_TEXT in followup["content"]


def test_queued_observation_runs_once_across_request_retries(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off")
    shot = image_file(tmp_path / "shot.png")
    agent, calls = bridged_agent(s, monkeypatch)

    s.enqueue_user_input(s.images.recognize(f"look {shot.name}"))
    turn = [{"role": "user", "content": "continue"}]
    first = agent.prepare_request(turn)
    second = agent.prepare_request(turn)  # a retry claims the same pending input again

    assert len(calls["vision"]) == 1
    observed = [message for message in second.messages if OBSERVATION_TEXT in str(message.get("content"))]
    assert observed and not any(IMAGE_REFS_KEY in message for message in second.messages)
    assert first.pending == second.pending


def test_enqueue_image_without_vision_still_refuses(tmp_path):
    s = session(tmp_path, image_input="off", vision=False)
    shot = image_file(tmp_path / "shot.png")

    with pytest.raises(ModelError, match="Image input is disabled"):
        s.enqueue_user_input(s.images.recognize(f"look {shot.name}"))


def test_bridge_logs_one_transcript_line_per_observation(tmp_path, monkeypatch):
    s = session(tmp_path, image_input="off")
    shot = image_file(tmp_path / "shot.png")
    agent, calls = bridged_agent(s, monkeypatch)
    logged = []
    agent.model.on_vision_observe = lambda label, detail: logged.append((label, detail))

    agent.run(s.images.recognize(f"look {shot.name}"))

    # One line per observation, naming the vision entry and the image count; fired before the
    # request, so a failure to observe still leaves the line in the log.
    assert logged == [("v/vision-model", "observing 1 image via the vision bridge")]
    assert len(calls["vision"]) == 1
