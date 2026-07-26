"""Local image inputs and protocol-neutral image payload helpers."""

from __future__ import annotations

import base64
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
_TOKEN_RE = re.compile(r"(?:'[^'\n]*'|\"(?:\\.|[^\"\n])*\"|(?:\\.|[^\s])+)")
_LEADING_PUNCTUATION = "([{<"
_TRAILING_PUNCTUATION = ",;:!?)]}>"


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
        name = _safe_name(name)
        if not re.fullmatch(r"[0-9a-f]{64}", ref) or not name or media_type not in SUPPORTED_FORMATS.values() or width <= 0 or height <= 0 or size <= 0:
            return None
        return cls(ref, name, media_type, width, height, size, str(value.get("source_text") or ""))


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


def image_refs(message: Json) -> tuple[ImageRef, ...]:
    raw = message.get(IMAGE_REFS_KEY)
    if not isinstance(raw, list):
        return ()
    return tuple(image for value in raw if (image := ImageRef.from_json(value)) is not None)


def image_label_text(message: Json) -> str:
    content = str(message.get("content") or "")
    images = image_refs(message)
    if not images:
        return content
    # Current messages already store labels in content. This fallback also keeps an older or
    # hand-written structured message readable when it contains metadata but no visible label.
    if any(f"[Image #{index}" in content for index in range(1, len(images) + 1)):
        return content
    labels = " ".join(f"[Image #{index} \u00b7 {image.name}]" for index, image in enumerate(images, 1))
    return " ".join(part for part in (labels, content) if part)


def recognize_images(text: str, cwd: str, existing: tuple[ImageRef, ...] = ()) -> UserInput:
    """Replace readable local image path tokens with markers, preserving existing markers."""

    if text.lstrip().startswith("/") and "\n" not in text:
        return UserInput(text, existing)
    replacements: list[tuple[int, int, ImageRef]] = []
    known_refs = {image.ref for image in existing}
    for match in _TOKEN_RE.finditer(text):
        raw = match.group(0)
        if IMAGE_MARKER in raw:
            continue
        left_trimmed = raw.lstrip(_LEADING_PUNCTUATION)
        candidate_raw = left_trimmed.rstrip(_TRAILING_PUNCTUATION)
        leading = len(raw) - len(left_trimmed)
        trailing = len(left_trimmed) - len(candidate_raw)
        if not candidate_raw:
            continue
        decoded = _decode_path_token(candidate_raw)
        if not decoded:
            continue
        decoded = os.path.expanduser(decoded)
        path = os.path.abspath(decoded if os.path.isabs(decoded) else os.path.join(cwd, decoded))
        image = inspect_image(path, source_text=candidate_raw, strict=False)
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


def inspect_image(path: str, *, source_text: str = "", strict: bool = True) -> ImageRef | None:
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
        return ImageRef(ref, _safe_name(os.path.basename(path)), media_type, width, height, size, source_text, path)
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, ValueError) as error:
        if strict:
            raise ModelError(f"Cannot read image {source_text or path}: {error}") from error
        return None


def store_user_input(session: Session, value: str | UserInput) -> Json:
    stored_input = store_input(session, value)
    if not stored_input.images:
        return {"role": "user", "content": str(stored_input)}
    return {
        "role": "user",
        "content": stored_input.display_text(),
        IMAGE_REFS_KEY: [image.to_json() for image in stored_input.images],
    }


def store_input(session: Session, value: str | UserInput) -> UserInput:
    if not isinstance(value, UserInput) or not value.images:
        return UserInput(str(value))
    return UserInput(str(value), tuple(_store_image(session, image) for image in value.images))


def assets_dir(session: Session) -> str:
    from minacode.session import SessionSnapshotStore

    path = SessionSnapshotStore.session_path(session.config.data_dir, session.cwd, session.uid)
    return path[: -len(".jsonl")] + ".assets"


def asset_path(session: Session, image: ImageRef) -> str:
    return os.path.join(assets_dir(session), image.ref)


def image_bytes(session: Session, image: ImageRef) -> bytes:
    path = asset_path(session, image)
    try:
        with open(path, "rb") as file:
            data = file.read()
    except OSError as error:
        raise ModelError(f"Stored image is missing: {image.name} ({image.ref[:12]})") from error
    if hashlib.sha256(data).hexdigest() != image.ref:
        raise ModelError(f"Stored image is corrupt: {image.name} ({image.ref[:12]})")
    return data


def data_url(session: Session, image: ImageRef) -> str:
    return f"data:{image.media_type};base64,{base64.b64encode(image_bytes(session, image)).decode('ascii')}"


def chat_content(session: Session, message: Json) -> str | list[Json]:
    images = image_refs(message)
    if not images:
        return str(message.get("content") or "")
    parts: list[Json] = [{"type": "image_url", "image_url": {"url": data_url(session, image)}} for image in images]
    if text := str(message.get("content") or ""):
        parts.append({"type": "text", "text": text})
    return parts


def responses_content(session: Session, message: Json) -> str | list[Json]:
    images = image_refs(message)
    if not images:
        return str(message.get("content") or "")
    parts: list[Json] = [{"type": "input_image", "image_url": data_url(session, image)} for image in images]
    if text := str(message.get("content") or ""):
        parts.append({"type": "input_text", "text": text})
    return parts


def anthropic_content(session: Session, message: Json) -> str | list[Json]:
    images = image_refs(message)
    if not images:
        return str(message.get("content") or "")
    parts: list[Json] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.media_type,
                "data": base64.b64encode(image_bytes(session, image)).decode("ascii"),
            },
        }
        for image in images
    ]
    if text := str(message.get("content") or ""):
        parts.append({"type": "text", "text": text})
    return parts


def estimated_image_tokens(image: ImageRef) -> int:
    """Use the common 512px-tile vision estimate without encoding image bytes into context."""

    tiles = max(1, (image.width + 511) // 512) * max(1, (image.height + 511) // 512)
    return 85 + 170 * tiles


def _store_image(session: Session, image: ImageRef) -> ImageRef:
    if image.source_path:
        current = inspect_image(image.source_path, source_text=image.source_text)
        assert current is not None
        image = replace(current, source_text=image.source_text)
        destination = asset_path(session, image)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if not _asset_matches(destination, image.ref):
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
    elif not os.path.isfile(asset_path(session, image)):
        raise ModelError(f"Stored image is missing: {image.name} ({image.ref[:12]})")
    return replace(image, source_path="")


def _decode_path_token(token: str) -> str:
    try:
        values = shlex.split(token)
    except ValueError:
        return ""
    return values[0] if len(values) == 1 else ""


def _safe_name(name: str) -> str:
    return re.sub(r"[\x00-\x1f\x7f]", "\ufffd", os.path.basename(name))


def _asset_matches(path: str, ref: str) -> bool:
    try:
        with open(path, "rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest() == ref
    except OSError:
        return False
