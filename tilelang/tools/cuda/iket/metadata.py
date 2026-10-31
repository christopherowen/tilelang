"""Self-contained IKET event metadata carried through TIR."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


TOKEN_PREFIX = "__tl_iket_v1_"
MAX_EVENT_NAME_BYTES = 32
MAX_RANGE_NAME_BYTES = 36
_TOKEN_VERSION = 1

_VALID_EVENT_KINDS = (
    "mark",
    "range",
    "range_start",
    "range_end",
    "range_start_end",
    "range_step",
)


class RangeScope(IntEnum):
    """Hardware hierarchy scope for pairing IKET range events."""

    INTRA_WARP = 0
    INTER_WARP_IN_CTA = 1
    INTER_WARP_IN_SM = 2
    INTER_WARP_IN_TPC = 4
    INTER_WARP_IN_GPC = 8


class RangeType(IntEnum):
    """Pairing model for IKET ranges."""

    INVALID = 0
    START_END = 1
    PUSH_POP = 2
    MULTI_EVENT = 3


class EventPairMode(IntEnum):
    """How START_END or MULTI_EVENT range endpoints are matched."""

    INVALID = 0
    NAME_ONLY = 1
    NAME_AND_PAYLOAD = 2


class EventPos(IntEnum):
    """Position of an event within an IKET range."""

    NOT_IN_RANGE = 0
    RANGE_START = 1
    RANGE_END = 2
    RANGE_START_END = 4
    RANGE_INTERNAL = 8


class InstrumentMethod(IntEnum):
    """IKET CUBIN instrumentation method."""

    PMTRIG5 = 1
    PMTRIG6 = 2
    NATIVE_DUMP = 3
    UTRACE_EVENT = 4
    EXTENDED_NATIVE_DUMP = 5


class PayloadType(IntEnum):
    """IKET scalar payload type identifiers."""

    NO_PAYLOAD = 0
    I8 = 1
    UI8 = 2
    I16 = 3
    UI16 = 4
    I32 = 5
    UI32 = 6
    I64 = 7
    FP4 = 9
    FP8 = 10
    FP16 = 11
    BF16 = 12
    FP32 = 13
    FP64 = 14
    POINTER = 15
    UI64 = 16


class Colors(IntEnum):
    """Perfetto track/slice color presets supported by IKET metadata."""

    AUTO = -1
    BLACK = 0x000000
    WHITE = 0xFFFFFF
    RED = 0xFF0000
    GREEN = 0x00FF00
    BLUE = 0x0000FF
    YELLOW = 0xFFFF00
    CYAN = 0x00FFFF
    MAGENTA = 0xFF00FF
    ORANGE = 0xFFA500
    PURPLE = 0x800080
    GRAY = 0x808080
    LIGHT_GRAY = 0xD3D3D3
    DARK_GRAY = 0xA9A9A9
    BROWN = 0xA52A2A
    PINK = 0xFFC0CB
    LIGHT_GREEN = 0x90EE90
    LIGHT_BLUE = 0xADD8E6


_SCOPE_ALIASES: dict[str, RangeScope] = {
    "intra_warp": RangeScope.INTRA_WARP,
    "warp": RangeScope.INTRA_WARP,
    "inter_warp_in_cta": RangeScope.INTER_WARP_IN_CTA,
    "cta": RangeScope.INTER_WARP_IN_CTA,
    "block": RangeScope.INTER_WARP_IN_CTA,
    "inter_warp_in_sm": RangeScope.INTER_WARP_IN_SM,
    "sm": RangeScope.INTER_WARP_IN_SM,
    "inter_warp_in_tpc": RangeScope.INTER_WARP_IN_TPC,
    "tpc": RangeScope.INTER_WARP_IN_TPC,
    "inter_warp_in_gpc": RangeScope.INTER_WARP_IN_GPC,
    "gpc": RangeScope.INTER_WARP_IN_GPC,
}

_RANGE_TYPE_ALIASES: dict[str, RangeType] = {
    "invalid": RangeType.INVALID,
    "start_end": RangeType.START_END,
    "push_pop": RangeType.PUSH_POP,
    "multi_event": RangeType.MULTI_EVENT,
}

_PAIR_MODE_ALIASES: dict[str, EventPairMode] = {
    "invalid": EventPairMode.INVALID,
    "none": EventPairMode.INVALID,
    "name_only": EventPairMode.NAME_ONLY,
    "name": EventPairMode.NAME_ONLY,
    "name_and_payload": EventPairMode.NAME_AND_PAYLOAD,
    "payload": EventPairMode.NAME_AND_PAYLOAD,
}

_COLOR_ALIASES: dict[str, Colors] = {member.name.lower(): member for member in Colors}


def parse_range_scope(value: int | str | RangeScope) -> RangeScope:
    if isinstance(value, RangeScope):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            return RangeScope(value)
        except ValueError as exc:
            raise ValueError(f"Unsupported IKET range scope: {value!r}") from exc
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _SCOPE_ALIASES:
            return _SCOPE_ALIASES[key]
    raise ValueError(f"Unsupported IKET range scope: {value!r}")


def parse_range_type(value: int | str | RangeType) -> RangeType:
    if isinstance(value, RangeType):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            return RangeType(value)
        except ValueError as exc:
            raise ValueError(f"Unsupported IKET range type: {value!r}") from exc
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _RANGE_TYPE_ALIASES:
            return _RANGE_TYPE_ALIASES[key]
    raise ValueError(f"Unsupported IKET range type: {value!r}")


def parse_pair_mode(value: int | str | EventPairMode) -> EventPairMode:
    if isinstance(value, EventPairMode):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            return EventPairMode(value)
        except ValueError as exc:
            raise ValueError(f"Unsupported IKET event pair mode: {value!r}") from exc
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _PAIR_MODE_ALIASES:
            return _PAIR_MODE_ALIASES[key]
    raise ValueError(f"Unsupported IKET event pair mode: {value!r}")


def parse_color(value: int | str | Colors) -> int:
    if isinstance(value, Colors):
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        if value == -1 or 0 <= value <= 0xFFFFFF:
            return int(value)
        raise ValueError(f"IKET color integer must be -1 (AUTO) or 0x000000..0xFFFFFF: {value!r}")
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _COLOR_ALIASES:
            return int(_COLOR_ALIASES[key])
        if key.startswith("#") and len(key) == 7:
            try:
                return int(key[1:], 16)
            except ValueError as exc:
                raise ValueError(f"Unsupported IKET color string: {value!r}") from exc
    raise ValueError(f"Unsupported IKET color: {value!r}")


def _default_range_fields(kind: str, name: str, range_id: int) -> tuple[str, int, int, int, int, int]:
    if kind == "mark" or range_id == 0:
        return ("", int(RangeType.INVALID), int(RangeScope.INTRA_WARP), int(EventPairMode.INVALID), int(Colors.AUTO), int(EventPos.NOT_IN_RANGE))
    if kind == "range":
        return (name, int(RangeType.PUSH_POP), int(RangeScope.INTRA_WARP), int(EventPairMode.INVALID), int(Colors.AUTO), int(EventPos.RANGE_START))
    if kind == "range_start_end":
        return (name, int(RangeType.START_END), int(RangeScope.INTRA_WARP), int(EventPairMode.NAME_ONLY), int(Colors.AUTO), int(EventPos.RANGE_START_END))
    if kind == "range_start":
        return (name, int(RangeType.START_END), int(RangeScope.INTRA_WARP), int(EventPairMode.NAME_ONLY), int(Colors.AUTO), int(EventPos.RANGE_START))
    if kind == "range_end":
        return (name, int(RangeType.START_END), int(RangeScope.INTRA_WARP), int(EventPairMode.NAME_ONLY), int(Colors.AUTO), int(EventPos.RANGE_END))
    if kind == "range_step":
        return (name, int(RangeType.MULTI_EVENT), int(RangeScope.INTRA_WARP), int(EventPairMode.NAME_ONLY), int(Colors.AUTO), int(EventPos.RANGE_INTERNAL))
    return ("", 0, 0, 0, -1, 0)


@dataclass(frozen=True)
class Event:
    """Metadata for one IKET event emitted by a TileLang kernel."""

    name: str
    event_id: int
    kind: str
    range_id: int = 0
    payload_type: str = "NoPayload"
    payload_iket_id: int = 0
    range_name: str = ""
    range_type: int = 0
    range_scope: int = 0
    pair_mode: int = 0
    color: int = -1
    event_pos: int = 0

    def __post_init__(self) -> None:
        def_rname, def_rtype, def_rscope, def_pmode, def_color, def_epos = _default_range_fields(
            self.kind, self.name, self.range_id
        )
        if not self.range_name and def_rname:
            object.__setattr__(self, "range_name", def_rname)
        if self.range_type == 0 and def_rtype != 0:
            object.__setattr__(self, "range_type", def_rtype)
        if self.pair_mode == 0 and def_pmode != 0:
            object.__setattr__(self, "pair_mode", def_pmode)
        if self.event_pos == 0 and def_epos != 0:
            object.__setattr__(self, "event_pos", def_epos)


def encode_event(event: Event) -> str:
    """Encode an event as a source-safe token stored in a TIR StringImm."""
    data: dict[str, Any] = {
        "version": _TOKEN_VERSION,
        "name": event.name,
        "event_id": event.event_id,
        "kind": event.kind,
        "range_id": event.range_id,
        "payload_type": event.payload_type,
        "payload_iket_id": event.payload_iket_id,
    }
    def_rname, def_rtype, def_rscope, def_pmode, def_color, def_epos = _default_range_fields(
        event.kind, event.name, event.range_id
    )
    if event.range_name != def_rname:
        data["range_name"] = event.range_name
    if event.range_type != def_rtype:
        data["range_type"] = int(event.range_type)
    if event.range_scope != def_rscope:
        data["range_scope"] = int(event.range_scope)
    if event.pair_mode != def_pmode:
        data["pair_mode"] = int(event.pair_mode)
    if event.color != def_color:
        data["color"] = int(event.color)
    if event.event_pos != def_epos:
        data["event_pos"] = int(event.event_pos)

    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return TOKEN_PREFIX + encoded


def decode_event(token: str) -> Event:
    """Decode and validate an event token recovered from generated CUDA."""
    if not token.startswith(TOKEN_PREFIX):
        raise ValueError("Invalid TileLang IKET metadata token prefix")

    encoded = token.removeprefix(TOKEN_PREFIX)
    try:
        padding = "=" * (-len(encoded) % 4)
        data: dict[str, Any] = json.loads(base64.urlsafe_b64decode(encoded + padding).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid TileLang IKET metadata token") from exc

    required_fields = {
        "version",
        "name",
        "event_id",
        "kind",
        "range_id",
        "payload_type",
        "payload_iket_id",
    }
    optional_fields = {
        "range_name",
        "range_type",
        "range_scope",
        "pair_mode",
        "color",
        "event_pos",
    }
    if not required_fields.issubset(data) or not set(data).issubset(required_fields | optional_fields) or data["version"] != _TOKEN_VERSION:
        raise ValueError("Unsupported TileLang IKET metadata schema")
    if not isinstance(data["name"], str) or not data["name"]:
        raise ValueError("IKET event metadata must contain a non-empty name")
    if len(data["name"].encode("utf-8")) > MAX_EVENT_NAME_BYTES:
        raise ValueError(f"IKET event name must be at most {MAX_EVENT_NAME_BYTES} UTF-8 bytes")
    if data["kind"] not in _VALID_EVENT_KINDS:
        raise ValueError(f"Unsupported IKET event kind: {data['kind']!r}")
    for field in ("event_id", "range_id", "payload_iket_id"):
        if not isinstance(data[field], int) or isinstance(data[field], bool) or data[field] < 0:
            raise ValueError(f"Invalid IKET metadata field {field!r}")
    if data["event_id"] == 0 or data["event_id"] == 31 or data["event_id"] >= 4095:
        raise ValueError(f"Invalid IKET event id: {data['event_id']}")
    if not isinstance(data["payload_type"], str):
        raise ValueError("Invalid IKET payload type metadata")
    if "range_name" in data:
        if not isinstance(data["range_name"], str):
            raise ValueError("Invalid IKET range_name metadata")
        if len(data["range_name"].encode("utf-8")) > MAX_RANGE_NAME_BYTES:
            raise ValueError(f"IKET range name must be at most {MAX_RANGE_NAME_BYTES} UTF-8 bytes")
    for int_field in ("range_type", "range_scope", "pair_mode", "color", "event_pos"):
        if int_field in data and (not isinstance(data[int_field], int) or isinstance(data[int_field], bool)):
            raise ValueError(f"Invalid IKET metadata field {int_field!r}")

    data.pop("version")
    def_rname, def_rtype, def_rscope, def_pmode, def_color, def_epos = _default_range_fields(
        data["kind"], data["name"], data["range_id"]
    )
    data.setdefault("range_name", def_rname)
    data.setdefault("range_type", def_rtype)
    data.setdefault("range_scope", def_rscope)
    data.setdefault("pair_mode", def_pmode)
    data.setdefault("color", def_color)
    data.setdefault("event_pos", def_epos)
    return Event(**data)
