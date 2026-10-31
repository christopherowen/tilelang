"""Kernel-side IKET event helpers."""

from __future__ import annotations

import builtins
import threading
import zlib
from dataclasses import dataclass, replace
from typing import Any

from tvm import tirx
from tvm.tirx.script.builder import evaluate as T_evaluate

from .metadata import (
    MAX_EVENT_NAME_BYTES,
    MAX_RANGE_NAME_BYTES,
    Colors,
    Event,
    EventPairMode,
    EventPos,
    PayloadType,
    RangeScope,
    RangeType,
    decode_event,
    encode_event,
    parse_color,
    parse_pair_mode,
    parse_range_scope,
    parse_range_type,
)


_RANGE_END_EVENT_ID = 31
_EVENT_EXTERN = "TL_IKET_EVENT"
_EVENT_PAYLOAD_U32_EXTERN = "TL_IKET_EVENT_PAYLOAD_U32"
_EVENT_PAYLOAD_F32_EXTERN = "TL_IKET_EVENT_PAYLOAD_F32"
_EVENT_PAYLOAD_F16_EXTERN = "TL_IKET_EVENT_PAYLOAD_F16"
_EVENT_PAYLOAD_BF16_EXTERN = "TL_IKET_EVENT_PAYLOAD_BF16"
_EVENT_PAYLOAD_U64_EXTERN = "TL_IKET_EVENT_PAYLOAD_U64"
_EVENT_PAYLOAD_F64_EXTERN = "TL_IKET_EVENT_PAYLOAD_F64"
_EVENT_PAYLOAD_PTR_EXTERN = "TL_IKET_EVENT_PAYLOAD_PTR"


@dataclass(frozen=True)
class PayloadSpec:
    """IKET payload value descriptor for a TileLang CUDA event."""

    expr: Any
    dtype: str
    iket_id: int


@dataclass(frozen=True)
class _StartEndConfig:
    range_name: str
    range_id: int
    range_type: int
    range_scope: int
    pair_mode: int
    color: int
    distinct_end_event: bool
    start_payload_type: str


_events: dict[tuple[str, str], Event] = {}
_ranges: dict[str, int] = {}
_start_end_configs: dict[str, _StartEndConfig] = {}
_range_stacks = threading.local()
_start_end_stacks = threading.local()
_registry_lock = threading.RLock()
_next_event_id = 1


def _range_stack() -> list[str]:
    stack = getattr(_range_stacks, "value", None)
    if stack is None:
        stack = []
        _range_stacks.value = stack
    return stack


def _start_end_stack() -> list[str]:
    stack = getattr(_start_end_stacks, "value", None)
    if stack is None:
        stack = []
        _start_end_stacks.value = stack
    return stack


def reset() -> None:
    """Reset frontend event-id allocation for kernels built afterward."""
    global _next_event_id
    with _registry_lock:
        _events.clear()
        _ranges.clear()
        _start_end_configs.clear()
        _range_stack().clear()
        _start_end_stack().clear()
        _next_event_id = 1


def event_table() -> list[dict[str, int | str]]:
    """Return events registered while constructing recent kernels."""
    from .session import runtime_payloads_enabled

    runtime_payloads = runtime_payloads_enabled()
    with _registry_lock:
        return [
            {
                "name": event.name,
                "event_id": event.event_id,
                "kind": event.kind,
                "range_id": event.range_id,
                "range_name": event.range_name,
                "range_type": event.range_type,
                "range_scope": event.range_scope,
                "pair_mode": event.pair_mode,
                "color": event.color,
                "event_pos": event.event_pos,
                "payload_type": event.payload_type,
                "payload_iket_id": event.payload_iket_id,
                "runtime_payload_iket_id": event.payload_iket_id if runtime_payloads else 0,
            }
            for event in sorted(_events.values(), key=lambda item: item.event_id)
        ]


def payload(expr: Any, dtype: str | None = None) -> PayloadSpec:
    """Attach a supported scalar payload value to an IKET event."""
    resolved_dtype = _normalize_payload_dtype(dtype or _infer_payload_dtype(expr))
    _validate_runtime_payload_dtype(resolved_dtype)
    return PayloadSpec(expr=expr, dtype=resolved_dtype, iket_id=_payload_iket_id(resolved_dtype))


def mark(name: str, payload: Any = None):
    """Emit an IKET instant marker at the current program point."""
    payload_spec = _payload_spec(payload)
    event = _get_event(name, "mark", payload_spec=payload_spec)
    return _event_call(event, payload_spec)


def range_push(
    name: str,
    payload: Any = None,
    *,
    scope: int | str | RangeScope = RangeScope.INTRA_WARP,
    color: int | str | Colors = Colors.AUTO,
):
    """Emit an IKET push-range start event at the current program point."""
    payload_spec = _payload_spec(payload)
    resolved_scope = int(parse_range_scope(scope))
    resolved_color = parse_color(color)
    event = _get_event(
        name,
        "range",
        payload_spec=payload_spec,
        range_name=name,
        range_type=int(RangeType.PUSH_POP),
        range_scope=resolved_scope,
        pair_mode=int(EventPairMode.INVALID),
        color=resolved_color,
        event_pos=int(EventPos.RANGE_START),
    )
    _range_stack().append(name)
    return _event_call(event, payload_spec)


def range_pop(name: str | None = None):
    """Emit an IKET range-end event for the current warp-local push range."""
    stack = _range_stack()
    if not stack:
        raise RuntimeError("iket.range_pop() called without a matching iket.range_push()")
    started = stack.pop()
    if name is not None and name != started:
        raise RuntimeError(f"iket.range_pop({name!r}) does not match active range {started!r}")
    return tirx.call_extern("handle", _EVENT_EXTERN, _RANGE_END_EVENT_ID)


def range_start(
    name: str,
    payload: Any = None,
    *,
    scope: int | str | RangeScope = RangeScope.INTRA_WARP,
    pair_mode: int | str | EventPairMode | None = None,
    color: int | str | Colors = Colors.AUTO,
    distinct_end_event: bool = False,
):
    """Emit an IKET START_END range-start event at the current program point.

    Unlike :func:`range_push`, ``START_END`` ranges support:
    * Overlapping / software-pipelined lifetimes without strict LIFO pop ordering
    * Payloads on both ``range_start`` and ``range_end``
    * Stage pairing via ``pair_mode="name_and_payload"``
    * Cross-warp / cross-CTA producer-consumer ranges via ``scope="cta"`` / ``"sm"`` / ``"tpc"`` / ``"gpc"``
    """
    _validate_range_name(name)
    payload_spec = _payload_spec(payload)
    resolved_scope = int(parse_range_scope(scope))
    resolved_pair_mode = (
        int(EventPairMode.NAME_ONLY)
        if pair_mode is None
        else int(parse_pair_mode(pair_mode))
    )
    if resolved_pair_mode == int(EventPairMode.INVALID):
        resolved_pair_mode = int(EventPairMode.NAME_ONLY)
    resolved_color = parse_color(color)

    use_distinct = (
        distinct_end_event
        or resolved_scope != int(RangeScope.INTRA_WARP)
        or resolved_pair_mode == int(EventPairMode.NAME_AND_PAYLOAD)
    )
    kind = "range_start" if use_distinct else "range_start_end"
    event_pos = int(EventPos.RANGE_START) if use_distinct else int(EventPos.RANGE_START_END)

    event = _get_event(
        name,
        kind,
        payload_spec=payload_spec,
        range_name=name,
        range_type=int(RangeType.START_END),
        range_scope=resolved_scope,
        pair_mode=resolved_pair_mode,
        color=resolved_color,
        event_pos=event_pos,
    )
    with _registry_lock:
        _start_end_configs[name] = _StartEndConfig(
            range_name=name,
            range_id=event.range_id,
            range_type=int(RangeType.START_END),
            range_scope=resolved_scope,
            pair_mode=resolved_pair_mode,
            color=resolved_color,
            distinct_end_event=use_distinct,
            start_payload_type=event.payload_type,
        )
    _start_end_stack().append(name)
    return _event_call(event, payload_spec)


def range_end(
    name: str | Any | None = None,
    payload: Any = None,
    *,
    scope: int | str | RangeScope | None = None,
    pair_mode: int | str | EventPairMode | None = None,
    color: int | str | Colors | None = None,
    distinct_end_event: bool | None = None,
):
    """Emit an IKET START_END range-end event for a named or active range."""
    resolved_name = _resolve_range_end_name(name)
    if resolved_name is None:
        # Backward compatibility if range_end() was called to close a range_push()
        if _range_stack():
            return range_pop()
        raise RuntimeError("iket.range_end() called without a matching iket.range_start()")

    _validate_range_name(resolved_name)
    payload_spec = _payload_spec(payload)
    end_payload_type = payload_spec.dtype if payload_spec is not None else "NoPayload"

    with _registry_lock:
        cfg = _start_end_configs.get(resolved_name)

    resolved_scope = (
        int(parse_range_scope(scope))
        if scope is not None
        else (cfg.range_scope if cfg is not None else int(RangeScope.INTRA_WARP))
    )
    resolved_pair_mode = (
        int(parse_pair_mode(pair_mode))
        if pair_mode is not None
        else (cfg.pair_mode if cfg is not None else int(EventPairMode.NAME_ONLY))
    )
    if resolved_pair_mode == int(EventPairMode.INVALID):
        resolved_pair_mode = int(EventPairMode.NAME_ONLY)
    resolved_color = (
        parse_color(color)
        if color is not None
        else (cfg.color if cfg is not None else int(Colors.AUTO))
    )
    resolved_range_type = cfg.range_type if cfg is not None else int(RangeType.START_END)

    use_distinct = (
        bool(distinct_end_event)
        or (cfg is not None and cfg.distinct_end_event)
        or resolved_scope != int(RangeScope.INTRA_WARP)
        or resolved_pair_mode == int(EventPairMode.NAME_AND_PAYLOAD)
        or resolved_range_type == int(RangeType.MULTI_EVENT)
        or (cfg is not None and cfg.start_payload_type != end_payload_type)
        or cfg is None
    )

    if not use_distinct:
        event = _get_event(
            resolved_name,
            "range_start_end",
            payload_spec=payload_spec,
            range_name=resolved_name,
            range_type=resolved_range_type,
            range_scope=resolved_scope,
            pair_mode=resolved_pair_mode,
            color=resolved_color,
            event_pos=int(EventPos.RANGE_START_END),
        )
        return _event_call(event, payload_spec)

    # Upgrade any prior single-event start in the registry to RANGE_START
    with _registry_lock:
        se_key = ("range_start_end", resolved_name)
        if se_key in _events:
            prev = _events.pop(se_key)
            upgraded = replace(prev, kind="range_start", event_pos=int(EventPos.RANGE_START))
            _events[("range_start", resolved_name)] = upgraded

    end_evt_name = _end_event_name(resolved_name)
    event = _get_event(
        end_evt_name,
        "range_end",
        payload_spec=payload_spec,
        range_name=resolved_name,
        range_type=resolved_range_type,
        range_scope=resolved_scope,
        pair_mode=resolved_pair_mode,
        color=resolved_color,
        event_pos=int(EventPos.RANGE_END),
    )
    return _event_call(event, payload_spec)


def range_step(
    range_name: str,
    step_name: str | None = None,
    payload: Any = None,
):
    """Emit an internal checkpoint event inside a MULTI_EVENT range."""
    _validate_range_name(range_name)
    evt_name = step_name if step_name is not None else _step_event_name(range_name)
    payload_spec = _payload_spec(payload)

    with _registry_lock:
        cfg = _start_end_configs.get(range_name)
        if cfg is not None:
            _start_end_configs[range_name] = replace(
                cfg,
                range_type=int(RangeType.MULTI_EVENT),
                distinct_end_event=True,
            )
        for prior_kind in ("range_start_end", "range"):
            prior_key = (prior_kind, range_name)
            if prior_key in _events:
                prev = _events.pop(prior_key)
                upgraded_pmode = (
                    int(EventPairMode.NAME_ONLY)
                    if prev.pair_mode == int(EventPairMode.INVALID)
                    else prev.pair_mode
                )
                upgraded = replace(
                    prev,
                    kind="range_start",
                    range_type=int(RangeType.MULTI_EVENT),
                    pair_mode=upgraded_pmode,
                    event_pos=int(EventPos.RANGE_START),
                )
                _events[("range_start", range_name)] = upgraded
                if cfg is None:
                    cfg = _StartEndConfig(
                        range_name=range_name,
                        range_id=upgraded.range_id,
                        range_type=int(RangeType.MULTI_EVENT),
                        range_scope=upgraded.range_scope,
                        pair_mode=upgraded_pmode,
                        color=upgraded.color,
                        distinct_end_event=True,
                        start_payload_type=upgraded.payload_type,
                    )
                    _start_end_configs[range_name] = cfg

    resolved_scope = cfg.range_scope if cfg is not None else int(RangeScope.INTRA_WARP)
    resolved_pair_mode = cfg.pair_mode if cfg is not None else int(EventPairMode.NAME_ONLY)
    resolved_color = cfg.color if cfg is not None else int(Colors.AUTO)

    event = _get_event(
        evt_name,
        "range_step",
        payload_spec=payload_spec,
        range_name=range_name,
        range_type=int(RangeType.MULTI_EVENT),
        range_scope=resolved_scope,
        pair_mode=resolved_pair_mode,
        color=resolved_color,
        event_pos=int(EventPos.RANGE_INTERNAL),
    )
    return _event_call(event, payload_spec)


class _RangeScope:
    def __init__(
        self,
        name: str,
        payload: Any = None,
        *,
        end_payload: Any = None,
        range_type: int | str | RangeType = RangeType.PUSH_POP,
        scope: int | str | RangeScope = RangeScope.INTRA_WARP,
        pair_mode: int | str | EventPairMode | None = None,
        color: int | str | Colors = Colors.AUTO,
    ):
        self.name = name
        self.payload = payload
        self.end_payload = end_payload
        resolved_type = parse_range_type(range_type)
        resolved_scope = parse_range_scope(scope)
        if (
            end_payload is not None
            or pair_mode is not None
            or resolved_scope != RangeScope.INTRA_WARP
        ) and resolved_type == RangeType.PUSH_POP:
            resolved_type = RangeType.START_END
        self.range_type = resolved_type
        self.scope = resolved_scope
        self.pair_mode = pair_mode
        self.color = color

    def __enter__(self):
        if self.range_type in (RangeType.START_END, RangeType.MULTI_EVENT):
            T_evaluate(
                range_start(
                    self.name,
                    payload=self.payload,
                    scope=self.scope,
                    pair_mode=self.pair_mode,
                    color=self.color,
                    distinct_end_event=(self.range_type == RangeType.MULTI_EVENT),
                )
            )
        else:
            T_evaluate(
                range_push(
                    self.name,
                    payload=self.payload,
                    scope=self.scope,
                    color=self.color,
                )
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        with _registry_lock:
            cfg = _start_end_configs.get(self.name)
        upgraded_to_multi = (
            self.range_type == RangeType.PUSH_POP
            and cfg is not None
            and cfg.range_type == int(RangeType.MULTI_EVENT)
        )
        if self.range_type in (RangeType.START_END, RangeType.MULTI_EVENT) or upgraded_to_multi:
            if upgraded_to_multi:
                stack = _range_stack()
                if stack and stack[-1] == self.name:
                    stack.pop()
            T_evaluate(
                range_end(
                    self.name,
                    payload=self.end_payload,
                    scope=self.scope,
                    pair_mode=self.pair_mode,
                    color=self.color,
                )
            )
        else:
            T_evaluate(range_pop(self.name))
        return False


def range(
    name: str,
    payload: Any = None,
    *,
    end_payload: Any = None,
    range_type: int | str | RangeType = RangeType.PUSH_POP,
    scope: int | str | RangeScope = RangeScope.INTRA_WARP,
    pair_mode: int | str | EventPairMode | None = None,
    color: int | str | Colors = Colors.AUTO,
) -> _RangeScope:
    """Return a scope that emits IKET range-start and range-end events."""
    return _RangeScope(
        name,
        payload=payload,
        end_payload=end_payload,
        range_type=range_type,
        scope=scope,
        pair_mode=pair_mode,
        color=color,
    )


def _resolve_range_end_name(name: str | Any | None) -> str | None:
    se_stack = _start_end_stack()
    if name is None:
        if se_stack:
            return se_stack.pop()
        return None
    if isinstance(name, tirx.Call) and name.args:
        last_arg = name.args[-1]
        if isinstance(last_arg, tirx.StringImm):
            ev = decode_event(str(last_arg.value))
            resolved = ev.range_name or ev.name
            _remove_from_start_end_stack(resolved)
            return resolved
    if isinstance(name, str):
        _remove_from_start_end_stack(name)
        return name
    raise TypeError(f"Unsupported iket.range_end target: {name!r}")


def _remove_from_start_end_stack(name: str) -> None:
    se_stack = _start_end_stack()
    for idx in builtins.range(len(se_stack) - 1, -1, -1):
        if se_stack[idx] == name:
            se_stack.pop(idx)
            break


def _validate_range_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("IKET range name must be a non-empty string")
    if len(name.encode("utf-8")) > MAX_EVENT_NAME_BYTES:
        raise ValueError(f"IKET event name must be at most {MAX_EVENT_NAME_BYTES} UTF-8 bytes: {name!r}")


def _truncate_utf8(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _end_event_name(range_name: str) -> str:
    suffix = ":end"
    prefix = _truncate_utf8(range_name, MAX_EVENT_NAME_BYTES - len(suffix))
    return prefix + suffix


def _step_event_name(range_name: str) -> str:
    suffix = ":step"
    prefix = _truncate_utf8(range_name, MAX_EVENT_NAME_BYTES - len(suffix))
    return prefix + suffix


def _payload_spec(value: Any) -> PayloadSpec | None:
    if value is None:
        return None
    if isinstance(value, PayloadSpec):
        return value
    return payload(value)


def _get_event(
    name: str,
    kind: str,
    payload_spec: PayloadSpec | None = None,
    *,
    range_name: str = "",
    range_type: int = 0,
    range_scope: int = 0,
    pair_mode: int = 0,
    color: int = -1,
    event_pos: int = 0,
) -> Event:
    if not isinstance(name, str) or not name:
        raise ValueError("IKET event name must be a non-empty string")
    if len(name.encode("utf-8")) > MAX_EVENT_NAME_BYTES:
        raise ValueError(f"IKET event name must be at most {MAX_EVENT_NAME_BYTES} UTF-8 bytes: {name!r}")
    if range_name and len(range_name.encode("utf-8")) > MAX_RANGE_NAME_BYTES:
        raise ValueError(f"IKET range name must be at most {MAX_RANGE_NAME_BYTES} UTF-8 bytes: {range_name!r}")

    key = (kind, name)
    with _registry_lock:
        if key in _events:
            event = _events[key]
            _validate_payload_compat(event, payload_spec)
            return event

        global _next_event_id
        while _next_event_id == _RANGE_END_EVENT_ID:
            _next_event_id += 1
        event_id = _next_event_id
        _next_event_id += 1

        resolved_range_name = range_name or (name if kind != "mark" else "")
        range_id = 0
        if kind != "mark":
            range_id = _range_id(resolved_range_name)
            _ranges[resolved_range_name] = range_id

        event = Event(
            name=name,
            event_id=event_id,
            kind=kind,
            range_id=range_id,
            payload_type=payload_spec.dtype if payload_spec is not None else "NoPayload",
            payload_iket_id=payload_spec.iket_id if payload_spec is not None else 0,
            range_name=resolved_range_name,
            range_type=range_type,
            range_scope=range_scope,
            pair_mode=pair_mode,
            color=color,
            event_pos=event_pos,
        )
        _events[key] = event
        return event


def _validate_payload_compat(event: Event, payload_spec: PayloadSpec | None) -> None:
    payload_type = payload_spec.dtype if payload_spec is not None else "NoPayload"
    if event.payload_type != payload_type:
        raise ValueError(
            f"IKET event {event.name!r} was first registered with payload type "
            f"{event.payload_type!r}, but is now used with {payload_type!r}"
        )


def _event_call(event: Event, payload_spec: PayloadSpec | None):
    token = encode_event(event)
    if payload_spec is None:
        return tirx.call_extern("handle", _EVENT_EXTERN, event.event_id, token)
    extern = _payload_extern_name(payload_spec.dtype)
    return tirx.call_extern("handle", extern, event.event_id, payload_spec.expr, token)


def _payload_extern_name(dtype: str) -> str:
    if dtype == "float32":
        return _EVENT_PAYLOAD_F32_EXTERN
    if dtype == "float16":
        return _EVENT_PAYLOAD_F16_EXTERN
    if dtype == "bfloat16":
        return _EVENT_PAYLOAD_BF16_EXTERN
    if dtype == "float64":
        return _EVENT_PAYLOAD_F64_EXTERN
    if dtype in ("int64", "uint64"):
        return _EVENT_PAYLOAD_U64_EXTERN
    if dtype == "pointer":
        return _EVENT_PAYLOAD_PTR_EXTERN
    return _EVENT_PAYLOAD_U32_EXTERN


def _infer_payload_dtype(expr: Any) -> str:
    dtype = getattr(expr, "dtype", None)
    if dtype is not None:
        return str(dtype)
    if isinstance(expr, bool):
        return "uint32"
    if isinstance(expr, int):
        return "int32"
    if isinstance(expr, float):
        return "float32"
    raise TypeError(
        "Cannot infer IKET payload dtype. Use iket.payload(expr, dtype=...) with a supported payload dtype."
    )


_PAYLOAD_DTYPE_ALIASES: dict[str, str] = {
    "i8": "int8",
    "int8": "int8",
    "u8": "uint8",
    "ui8": "uint8",
    "uint8": "uint8",
    "i16": "int16",
    "int16": "int16",
    "u16": "uint16",
    "ui16": "uint16",
    "uint16": "uint16",
    "int": "int32",
    "i32": "int32",
    "int32": "int32",
    "uint": "uint32",
    "u32": "uint32",
    "ui32": "uint32",
    "uint32": "uint32",
    "long": "int64",
    "i64": "int64",
    "int64": "int64",
    "ulong": "uint64",
    "u64": "uint64",
    "ui64": "uint64",
    "uint64": "uint64",
    "fp4": "float4",
    "float4": "float4",
    "float4_e2m1": "float4",
    "float4_e2m1fn": "float4",
    "fp8": "float8",
    "float8": "float8",
    "float8_e4m3": "float8",
    "float8_e4m3fn": "float8",
    "float8_e4m3fnuz": "float8",
    "float8_e5m2": "float8",
    "half": "float16",
    "fp16": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "float": "float32",
    "fp32": "float32",
    "float32": "float32",
    "double": "float64",
    "fp64": "float64",
    "float64": "float64",
    "ptr": "pointer",
    "handle": "pointer",
    "pointer": "pointer",
}

_PAYLOAD_IKET_IDS: dict[str, int] = {
    "int8": int(PayloadType.I8),
    "uint8": int(PayloadType.UI8),
    "int16": int(PayloadType.I16),
    "uint16": int(PayloadType.UI16),
    "int32": int(PayloadType.I32),
    "uint32": int(PayloadType.UI32),
    "int64": int(PayloadType.I64),
    "float4": int(PayloadType.FP4),
    "float8": int(PayloadType.FP8),
    "float16": int(PayloadType.FP16),
    "bfloat16": int(PayloadType.BF16),
    "float32": int(PayloadType.FP32),
    "float64": int(PayloadType.FP64),
    "pointer": int(PayloadType.POINTER),
    "uint64": int(PayloadType.UI64),
}


def _normalize_payload_dtype(dtype: str) -> str:
    key = dtype.strip().lower()
    normalized = _PAYLOAD_DTYPE_ALIASES.get(key, key)
    _payload_iket_id(normalized)
    return normalized


def _validate_runtime_payload_dtype(dtype: str) -> None:
    if dtype not in _PAYLOAD_IKET_IDS:
        raise NotImplementedError(f"Unsupported TileLang IKET runtime payload dtype: {dtype!r}")


def _payload_iket_id(dtype: str) -> int:
    key = dtype.lower()
    if key not in _PAYLOAD_IKET_IDS:
        raise ValueError(f"Unsupported IKET payload dtype: {dtype!r}")
    return _PAYLOAD_IKET_IDS[key]


def _range_id(name: str) -> int:
    value = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
    return value or 1
