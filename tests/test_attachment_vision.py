"""Direct attachment routing and occurrence-local failed-image settlement."""

import json
import os

import pytest
from PIL import Image

from wizolt.base import ModelError, ToolCall
from wizolt.config import Config, ProviderConfig
from wizolt.engine import Agent
from wizolt.image import (
    FAILED_IMAGE_CONTEXT_PREFIX,
    IMAGE_ASSET_CONTEXT_PREFIX,
    IMAGE_MARKER,
    IMAGE_TEXT_ONLY_KEY,
    ImageInputs,
    UserInput,
)
from wizolt.session import Session, SessionSnapshotCodec


def image_file(path, *, color=(12, 34, 56)):
    Image.new("RGB", (32, 24), color).save(path, format="PNG")
    return path


def session(tmp_path, *, vision=True):
    config = Config(data_dir=str(tmp_path / "data"))
    config.providers = {"default": ProviderConfig(url="http://main.test", key="key", model="main-model")}
    if vision:
        config.providers["v"] = ProviderConfig(url="http://vision.test", key="vkey", model="vision-model")
        config.vision_provider = "v"
    return Session(cwd=str(tmp_path), config=config)


class SequenceModel:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.requests = []

    async def request_async(self, messages, tools=None):
        self.requests.append(messages)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return {"role": "assistant", "content": outcome}, [], outcome

    def cancel_active_request(self):
        pass


@pytest.mark.parametrize("vision", [False, True])
def test_attachment_always_goes_to_main_model_and_never_calls_vision(tmp_path, vision):
    s = session(tmp_path, vision=vision)
    image_file(tmp_path / "shot.png")
    model = SequenceModel(["done"])
    agent = Agent(s, output_fn=lambda _text: None)
    agent.model = model

    assert agent.run(s.images.recognize("inspect shot.png")) == "done"

    sent = [message for message in model.requests[0] if ImageInputs.input_refs(message)]
    assert len(sent) == 1
    assert sent[0]["content"] == "inspect [Image #1 · shot.png]"
    assert not hasattr(s.state, "image_support")

    [image] = ImageInputs.refs(sent[0])
    wire = s.images.responses_content(sent[0])
    assert wire[-1]["text"] == (
        f'inspect [Image #1 · shot.png]\n\n{IMAGE_ASSET_CONTEXT_PREFIX}\n- {{"image": 1, "name": "shot.png", "path": "{s.images.asset_path(image)}"}}'
    )
    # The mapping belongs only to request projection: stored history and the visible transcript
    # stay byte-identical to what the user submitted.
    assert IMAGE_ASSET_CONTEXT_PREFIX not in s.messages[0]["content"]
    assert IMAGE_ASSET_CONTEXT_PREFIX not in s.transcript_messages[0]["content"]


def test_projected_asset_mapping_is_structured_and_complete_for_multiple_images(tmp_path):
    s = session(tmp_path)
    first_path = image_file(tmp_path / 'odd "name".png')
    second_path = image_file(tmp_path / "second.png", color=(65, 43, 21))
    images = (s.images.load(str(first_path)), s.images.load(str(second_path)))
    message = s.images.message(UserInput(f"compare {IMAGE_MARKER} and {IMAGE_MARKER}", images))

    projected = s.images.responses_content(message)
    asset_lines = projected[-1]["text"].split(IMAGE_ASSET_CONTEXT_PREFIX + "\n", 1)[1].splitlines()
    assets = [json.loads(line.removeprefix("- ")) for line in asset_lines]

    assert assets == [
        {"image": 1, "name": 'odd "name".png', "path": s.images.asset_path(images[0])},
        {"image": 2, "name": "second.png", "path": s.images.asset_path(images[1])},
    ]
    assert IMAGE_ASSET_CONTEXT_PREFIX not in message["content"]


def test_failed_image_turn_is_replay_safe_and_next_text_turn_succeeds(tmp_path):
    s = session(tmp_path)
    source = image_file(tmp_path / "shot.png")
    rejected = ModelError("provider rejected the image")
    model = SequenceModel([rejected, "recovered"])
    agent = Agent(s, output_fn=lambda _text: None)
    agent.model = model

    with pytest.raises(ModelError) as caught:
        agent.run(s.images.recognize("inspect shot.png"))
    assert caught.value is rejected
    assert len(model.requests) == 1

    failed = s.messages[0]
    assert failed[IMAGE_TEXT_ONLY_KEY] is True
    assert ImageInputs.refs(failed) and not ImageInputs.input_refs(failed)
    assert FAILED_IMAGE_CONTEXT_PREFIX in failed["content"]
    [image] = ImageInputs.refs(failed)
    assert s.images.asset_path(image) in failed["content"]
    assert os.path.isfile(s.images.asset_path(image))
    assert FAILED_IMAGE_CONTEXT_PREFIX not in s.transcript_messages[0]["content"]
    assert failed["content"].count(s.images.asset_path(image)) == 1

    source.unlink()
    assert agent.run("continue without replaying pixels") == "recovered"
    assert not any(ImageInputs.input_refs(message) for message in model.requests[1])


def test_later_new_image_is_attempted_after_an_earlier_image_failure(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "first.png")
    image_file(tmp_path / "second.png", color=(65, 43, 21))
    model = SequenceModel([ModelError("first failed"), "second worked"])
    agent = Agent(s, output_fn=lambda _text: None)
    agent.model = model

    with pytest.raises(ModelError):
        agent.run(s.images.recognize("first.png"))
    assert agent.run(s.images.recognize("second.png")) == "second worked"

    second_refs = [image for message in model.requests[1] for image in ImageInputs.input_refs(message)]
    assert [image.name for image in second_refs] == ["second.png"]


def test_failed_queued_image_is_committed_text_only_instead_of_requeued(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "queued.png")

    class QueuingModel:
        def __init__(self):
            self.requests = []

        async def request_async(self, messages, tools=None):
            self.requests.append(messages)
            if len(self.requests) == 1:
                s.enqueue_user_input(s.images.recognize("look queued.png"))
                return {}, [ToolCall("read", "Read", ["missing.txt"])], ""
            raise ModelError("queued image rejected")

        def cancel_active_request(self):
            pass

    model = QueuingModel()
    agent = Agent(s, output_fn=lambda _text: None)
    agent.model = model

    with pytest.raises(ModelError, match="queued image rejected"):
        agent.run("start")

    assert s.pending_user_inputs == []
    settled = [message for message in s.messages if ImageInputs.refs(message)]
    assert len(settled) == 1
    assert settled[0][IMAGE_TEXT_ONLY_KEY] is True
    assert FAILED_IMAGE_CONTEXT_PREFIX in settled[0]["content"]


def test_multiple_failed_images_survive_snapshot_as_text_only_assets(tmp_path):
    s = session(tmp_path)
    image_file(tmp_path / "one.png")
    image_file(tmp_path / "two.png", color=(65, 43, 21))
    agent = Agent(s, output_fn=lambda _text: None)
    agent.model = SequenceModel([ModelError("no images")])

    with pytest.raises(ModelError):
        agent.run(s.images.recognize("compare one.png two.png"))
    s.save_snapshot()
    resumed = Session.load_snapshot(s.uid, config=s.config)

    failed = resumed.messages[0]
    assert [image.name for image in resumed.images.refs(failed)] == ["one.png", "two.png"]
    assert resumed.images.input_refs(failed) == ()
    for image in resumed.images.refs(failed):
        assert os.path.isfile(resumed.images.asset_path(image))


def test_obsolete_snapshot_state_is_ignored_and_not_written():
    state = SessionSnapshotCodec.agent_state({"goal": "keep", "image_support": {"old": False}})
    assert state.goal == "keep"
    assert not hasattr(state, "image_support")
