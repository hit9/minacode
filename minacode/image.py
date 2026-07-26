"""Local image input lifecycle and protocol payloads."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import os
import re
import shlex
import shutil
import tempfile
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable

from PIL import Image, UnidentifiedImageError

from minacode.base import Json, ModelError

if TYPE_CHECKING:
    from minacode.session import Session


IMAGE_MARKER = "\ufffc"
IMAGE_REFS_KEY = "_images"
SUPPORTED_FORMATS = {
    "GIF": "image/gif",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


@dataclass(frozen=True)
class ImageRef:
    ref: str
    name: str
    media_type: str
    width: int
    height: int
    size: int
    source_text: str = ""
    source_path: str = ""

    def to_json(self) -> Json:
        return {
            "ref": self.ref,
            "name": self.name,
            "media_type": self.media_type,
            "width": self.width,
            "height": self.height,
            "size": self.size,
            "source_text": self.source_text,
        }

    @classmethod
    def from_json(cls, value: object) -> ImageRef | None:
        if not isinstance(value, dict):
            return None
        try:
            ref = str(value["ref"])
            name = str(value["name"])
            media_type = str(value["media_type"])
            width = int(value["width"])
            height = int(value["height"])
            size = int(value["size"])
        except (KeyError, TypeError, ValueError):
            return None
        name = cls._safe_name(name)
        if not re.fullmatch(r"[0-9a-f]{64}", ref) or not name or media_type not in SUPPORTED_FORMATS.values() or width <= 0 or height <= 0 or size <= 0:
            return None
        return cls(ref, name, media_type, width, height, size, str(value.get("source_text") or ""))

    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r"[\x00-\x1f\x7f]", "\ufffd", os.path.basename(name))


class UserInput(str):
    """A draft string whose one-cell image markers map to immutable image references."""

    images: tuple[ImageRef, ...]

    def __new__(cls, text: str, images: tuple[ImageRef, ...] = ()) -> UserInput:
        value = super().__new__(cls, text)
        value.images = images
        if text.count(IMAGE_MARKER) != len(images):
            raise ValueError("image marker count does not match image references")
        return value

    def display_text(self) -> str:
        return self._expanded(lambda index, image: f"[Image #{index} \u00b7 {image.name}]")

    def original_text(self) -> str:
        return self._expanded(lambda _index, image: image.source_text or image.name)

    def _expanded(self, replacement: Callable[[int, ImageRef], str]) -> str:
        parts = str(self).split(IMAGE_MARKER)
        output = [parts[0]]
        for index, image in enumerate(self.images, 1):
            output.extend((replacement(index, image), parts[index]))
        return "".join(output)


class ImageInputs:
    """Own image recognition, storage, transport, and learned model capability for a session."""

    _TOKEN_RE = re.compile(r"(?:'[^'\n]*'|\"(?:\\.|[^\"\n])*\"|(?:\\.|[^\s])+)")
    _LEADING_PUNCTUATION = "([{<"
    _TRAILING_PUNCTUATION = ",;:!?)]}>"
    _MODALITY_TERMS = ("image", "vision", "multimodal", "input_image", "image_url", "modality")
    _UNSUPPORTED_TERMS = ("unsupported", "not supported", "does not support", "only supports text", "not enabled")

    def __init__(self, session: Session | None = None, *, cwd: str = "") -> None:
        self.session = session
        self.cwd = session.cwd if session is not None else cwd or os.getcwd()
        self.retained_refs: set[str] = set()
        self._learned_support: dict[tuple[str, str, str, str], bool] = {}

    @staticmethod
    def refs(message: Json) -> tuple[ImageRef, ...]:
        raw = message.get(IMAGE_REFS_KEY)
        if not isinstance(raw, list):
            return ()
        return tuple(image for value in raw if (image := ImageRef.from_json(value)) is not None)

    @classmethod
    def label_text(cls, message: Json) -> str:
        content = str(message.get("content") or "")
        images = cls.refs(message)
        if not images:
            return content
        # Current messages store labels in content. Keep older or hand-written structured
        # messages readable when their metadata has no visible labels.
        if any(f"[Image #{index}" in content for index in range(1, len(images) + 1)):
            return content
        labels = " ".join(f"[Image #{index} \u00b7 {image.name}]" for index, image in enumerate(images, 1))
        return " ".join(part for part in (labels, content) if part)

    def recognize(self, text: str, existing: tuple[ImageRef, ...] = ()) -> UserInput:
        """Replace readable local image path tokens with markers, preserving existing markers."""

        if text.lstrip().startswith("/") and "\n" not in text:
            return UserInput(text, existing)
        replacements: list[tuple[int, int, ImageRef]] = []
        known_refs = {image.ref for image in existing}
        for match in self._TOKEN_RE.finditer(text):
            raw = match.group(0)
            if IMAGE_MARKER in raw:
                continue
            left_trimmed = raw.lstrip(self._LEADING_PUNCTUATION)
            candidate_raw = left_trimmed.rstrip(self._TRAILING_PUNCTUATION)
            leading = len(raw) - len(left_trimmed)
            trailing = len(left_trimmed) - len(candidate_raw)
            if not candidate_raw:
                continue
            decoded = self._decode_path_token(candidate_raw)
            if not decoded:
                continue
            decoded = os.path.expanduser(decoded)
            path = os.path.abspath(decoded if os.path.isabs(decoded) else os.path.join(self.cwd, decoded))
            image = self._inspect(path, source_text=candidate_raw, strict=False)
            if image is None or image.ref in known_refs:
                continue
            known_refs.add(image.ref)
            replacements.append((match.start() + leading, match.end() - trailing, image))
        if not replacements:
            return UserInput(text, existing)
        by_start = {start: (end, image) for start, end, image in replacements}
        old_images = iter(existing)
        found: list[ImageRef] = []
        output: list[str] = []
        position = 0
        while position < len(text):
            if replacement := by_start.get(position):
                end, image = replacement
                output.append(IMAGE_MARKER)
                found.append(image)
                position = end
                continue
            char = text[position]
            output.append(char)
            if char == IMAGE_MARKER:
                found.append(next(old_images))
            position += 1
        return UserInput("".join(output), tuple(found))

    def prepare(self, value: str | UserInput) -> UserInput:
        if not isinstance(value, UserInput) or not value.images:
            return UserInput(str(value))
        if self.support() is False:
            raise ModelError("Image input is disabled for the active provider/model")
        if self.session is None:
            return value
        return UserInput(str(value), tuple(self._store(image) for image in value.images))

    def message(self, value: str | UserInput) -> Json:
        stored = self.prepare(value)
        if not stored.images:
            return {"role": "user", "content": str(stored)}
        self.retained_refs.difference_update(image.ref for image in stored.images)
        return {
            "role": "user",
            "content": stored.display_text(),
            IMAGE_REFS_KEY: [image.to_json() for image in stored.images],
        }

    def retain(self, images: tuple[ImageRef, ...]) -> None:
        self.retained_refs.update(image.ref for image in images)

    def support(self) -> bool | None:
        if self.session is None:
            return None
        configured = self.session.config.provider.image_input
        if configured == "on":
            return True
        if configured == "off":
            return False
        return self._learned_support.get(self._capability_key())

    def note_success(self, messages: list[Json]) -> None:
        if self.session is not None and self.session.config.provider.image_input == "auto" and self.support() is not False and self.has_images(messages):
            self._learned_support[self._capability_key()] = True

    def note_error(self, messages: list[Json], error: Exception) -> bool:
        unsupported = self.has_images(messages) and self.support() is not False and self._explicit_unsupported_error(error)
        if unsupported and self.session is not None and self.session.config.provider.image_input == "auto":
            self._learned_support[self._capability_key()] = False
        return unsupported

    @classmethod
    def has_images(cls, messages: list[Json]) -> bool:
        return any(cls.refs(message) for message in messages)

    def chat_content(self, message: Json) -> str | list[Json]:
        images = self.refs(message)
        if not images or self.support() is False:
            return self.label_text(message)
        parts: list[Json] = [{"type": "image_url", "image_url": {"url": self._data_url(image)}} for image in images]
        if text := str(message.get("content") or ""):
            parts.append({"type": "text", "text": text})
        return parts

    def responses_content(self, message: Json) -> str | list[Json]:
        images = self.refs(message)
        if not images or self.support() is False:
            return self.label_text(message)
        parts: list[Json] = [{"type": "input_image", "image_url": self._data_url(image)} for image in images]
        if text := str(message.get("content") or ""):
            parts.append({"type": "input_text", "text": text})
        return parts

    def anthropic_content(self, message: Json) -> str | list[Json]:
        images = self.refs(message)
        if not images or self.support() is False:
            return self.label_text(message)
        parts: list[Json] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.media_type,
                    "data": base64.b64encode(self._bytes(image)).decode("ascii"),
                },
            }
            for image in images
        ]
        if text := str(message.get("content") or ""):
            parts.append({"type": "text", "text": text})
        return parts

    @classmethod
    def estimated_tokens(cls, messages: list[Json]) -> int:
        return sum(cls._estimated_tokens(image) for message in messages for image in cls.refs(message))

    def assets_dir(self) -> str:
        session = self._session()
        from minacode.session import SessionSnapshotStore

        path = SessionSnapshotStore.session_path(session.config.data_dir, session.cwd, session.uid)
        return path[: -len(".jsonl")] + ".assets"

    @staticmethod
    def _inspect(path: str, *, source_text: str = "", strict: bool = True) -> ImageRef | None:
        try:
            if not os.path.isfile(path):
                raise OSError("not a regular file")
            size = os.path.getsize(path)
            with Image.open(path) as opened:
                image_format = str(opened.format or "").upper()
                width, height = opened.size
                frames = int(getattr(opened, "n_frames", 1))
                opened.verify()
            media_type = SUPPORTED_FORMATS.get(image_format)
            if media_type is None or (image_format == "GIF" and frames != 1):
                raise ValueError("supported formats are PNG, JPEG, WebP, and single-frame GIF")
            with open(path, "rb") as file:
                ref = hashlib.file_digest(file, "sha256").hexdigest()
            return ImageRef(ref, ImageRef._safe_name(os.path.basename(path)), media_type, width, height, size, source_text, path)
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError, ValueError) as error:
            if strict:
                raise ModelError(f"Cannot read image {source_text or path}: {error}") from error
            return None

    def _store(self, image: ImageRef) -> ImageRef:
        if image.source_path:
            current = self._inspect(image.source_path, source_text=image.source_text)
            assert current is not None
            image = replace(current, source_text=image.source_text)
            destination = self._asset_path(image)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            if not self._asset_matches(destination, image.ref):
                fd, temporary = tempfile.mkstemp(prefix=".image-", dir=os.path.dirname(destination))
                os.close(fd)
                try:
                    shutil.copyfile(image.source_path, temporary)
                    with open(temporary, "rb") as file:
                        copied_ref = hashlib.file_digest(file, "sha256").hexdigest()
                    if copied_ref != image.ref:
                        raise ModelError(f"Image changed while it was being read: {image.source_text or image.name}")
                    os.replace(temporary, destination)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
        elif not os.path.isfile(self._asset_path(image)):
            raise ModelError(f"Stored image is missing: {image.name} ({image.ref[:12]})")
        return replace(image, source_path="")

    def _bytes(self, image: ImageRef) -> bytes:
        path = self._asset_path(image)
        try:
            with open(path, "rb") as file:
                data = file.read()
        except OSError as error:
            raise ModelError(f"Stored image is missing: {image.name} ({image.ref[:12]})") from error
        if hashlib.sha256(data).hexdigest() != image.ref:
            raise ModelError(f"Stored image is corrupt: {image.name} ({image.ref[:12]})")
        return data

    def _data_url(self, image: ImageRef) -> str:
        return f"data:{image.media_type};base64,{base64.b64encode(self._bytes(image)).decode('ascii')}"

    def _asset_path(self, image: ImageRef) -> str:
        return os.path.join(self.assets_dir(), image.ref)

    def _capability_key(self) -> tuple[str, str, str, str]:
        session = self._session()
        provider = session.config.provider
        resolved = provider.resolve()
        return session.config.active_provider, resolved.api, resolved.base_url, provider.model

    @classmethod
    def _explicit_unsupported_error(cls, error: Exception) -> bool:
        text = str(error).lower()
        cause = getattr(error, "__cause__", None)
        status = getattr(cause, "status_code", None) or getattr(cause, "code", None)
        numeric_status: int | None = None
        with contextlib.suppress(TypeError, ValueError):
            numeric_status = int(status) if status is not None else None
        if numeric_status is not None and numeric_status not in {400, 415, 422}:
            return False
        if numeric_status is None and not re.search(r"\b(?:400|415|422)\b", text):
            return False
        return any(term in text for term in cls._MODALITY_TERMS) and any(term in text for term in cls._UNSUPPORTED_TERMS)

    def _session(self) -> Session:
        if self.session is None:
            raise ModelError("Image input is not attached to a session")
        return self.session

    @staticmethod
    def _decode_path_token(token: str) -> str:
        try:
            values = shlex.split(token)
        except ValueError:
            return ""
        return values[0] if len(values) == 1 else ""

    @staticmethod
    def _asset_matches(path: str, ref: str) -> bool:
        try:
            with open(path, "rb") as file:
                return hashlib.file_digest(file, "sha256").hexdigest() == ref
        except OSError:
            return False

    @staticmethod
    def _estimated_tokens(image: ImageRef) -> int:
        """Use the common 512px-tile estimate without putting encoded bytes in context."""

        tiles = max(1, (image.width + 511) // 512) * max(1, (image.height + 511) // 512)
        return 85 + 170 * tiles
