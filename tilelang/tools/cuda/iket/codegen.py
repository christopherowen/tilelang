"""CUDA source instrumentation for IKET events."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from .metadata import (
    TOKEN_PREFIX,
    Colors,
    Event,
    EventPairMode,
    EventPos,
    InstrumentMethod,
    RangeScope,
    RangeType,
    decode_event,
    encode_event,
)


_RANGE_END_EVENT_ID = 31
_EXTENDED_MAX_EVENT_ID = 4095
_EXTENDED_START_EVENT_ID = 64
_MAX_NATIVE_EVENTS = 30
_MAX_EXTENDED_EVENTS = _EXTENDED_MAX_EVENT_ID - _EXTENDED_START_EVENT_ID
_IKET_MAGIC = [157, 241, 190, 186]
_INJECTION_MARKER = "__tilelang_iket_tool__"
_TOKEN_PATTERN = re.compile(re.escape(TOKEN_PREFIX) + r"[A-Za-z0-9_-]+")
_EVENT_CALL_PATTERN = re.compile(
    r"(?P<prefix>\bTL_IKET_EVENT(?:_PAYLOAD_(?:U32|F32|F16|BF16|U64|F64|PTR))?\(\s*)"
    r"(?P<event_id>[0-9]+)"
    r"(?P<suffix>[^\n]*?\"(?P<token>" + re.escape(TOKEN_PREFIX) + r"[A-Za-z0-9_-]+)\"\s*\))"
)


@dataclass(frozen=True)
class _RangeMeta:
    name: str
    range_id: int
    range_type: int
    range_scope: int
    pair_mode: int
    color: int


def parse_instrument_method(method: str | int | InstrumentMethod | None) -> int | str:
    if method is None:
        return "auto"
    if isinstance(method, InstrumentMethod):
        return int(method)
    if isinstance(method, int) and not isinstance(method, bool):
        try:
            return int(InstrumentMethod(method))
        except ValueError as exc:
            raise ValueError(f"Unsupported IKET instrument method: {method!r}") from exc
    if isinstance(method, str):
        key = method.strip().lower()
        aliases: dict[str, int | str] = {
            "auto": "auto",
            "native": int(InstrumentMethod.NATIVE_DUMP),
            "native_dump": int(InstrumentMethod.NATIVE_DUMP),
            "extended": int(InstrumentMethod.EXTENDED_NATIVE_DUMP),
            "extended_native_dump": int(InstrumentMethod.EXTENDED_NATIVE_DUMP),
            "utrace": int(InstrumentMethod.UTRACE_EVENT),
            "utrace_event": int(InstrumentMethod.UTRACE_EVENT),
        }
        if key in aliases:
            return aliases[key]
    raise ValueError(f"Unsupported IKET instrument method: {method!r}")


def _resolve_effective_method(num_events: int, requested: str | int | InstrumentMethod | None) -> int:
    parsed = parse_instrument_method(requested)
    if parsed == "auto":
        if num_events <= _MAX_NATIVE_EVENTS:
            return int(InstrumentMethod.NATIVE_DUMP)
        if num_events <= _MAX_EXTENDED_EVENTS:
            return int(InstrumentMethod.EXTENDED_NATIVE_DUMP)
        raise ValueError(
            f"IKET module contains {num_events} events, exceeding the ExtendedNativeDump limit of {_MAX_EXTENDED_EVENTS}"
        )
    method_int = int(parsed)
    if method_int in (int(InstrumentMethod.NATIVE_DUMP), int(InstrumentMethod.UTRACE_EVENT)):
        if num_events > _MAX_NATIVE_EVENTS:
            raise ValueError(
                f"IKET InstrumentMethod({method_int}) supports at most {_MAX_NATIVE_EVENTS} events per module, "
                f"got {num_events}; use instrument_method='extended' or 'auto'"
            )
    elif method_int == int(InstrumentMethod.EXTENDED_NATIVE_DUMP):
        if num_events > _MAX_EXTENDED_EVENTS:
            raise ValueError(
                f"IKET ExtendedNativeDump supports at most {_MAX_EXTENDED_EVENTS} events per module, got {num_events}"
            )
    return method_int


def _canonicalize_cuda_events(
    code: str,
    *,
    instrument_method: str | int | InstrumentMethod | None = "auto",
) -> tuple[str, list[Event], int]:
    """Recover events and assign module-wide IDs to their generated calls."""
    matches = list(_EVENT_CALL_PATTERN.finditer(code))
    if len(matches) != len(_TOKEN_PATTERN.findall(code)):
        raise ValueError("IKET metadata token is not attached to a supported event call")
    if not matches:
        return code, [], int(InstrumentMethod.NATIVE_DUMP)

    decoded_by_token: dict[str, Event] = {}
    ordered_tokens: list[str] = []
    range_has_separate_end: set[int] = set()
    range_has_step: set[int] = set()

    for match in matches:
        token = match.group("token")
        if token not in decoded_by_token:
            ev = decode_event(token)
            decoded_by_token[token] = ev
            ordered_tokens.append(token)
            if ev.range_id:
                if ev.kind == "range_end":
                    range_has_separate_end.add(ev.range_id)
                elif ev.kind == "range_step":
                    range_has_step.add(ev.range_id)
                    range_has_separate_end.add(ev.range_id)

    normalized_by_token: dict[str, Event] = {}
    for token in ordered_tokens:
        ev = decoded_by_token[token]
        if ev.range_id in range_has_separate_end and ev.kind in ("range_start_end", "range"):
            ev = replace(ev, kind="range_start", event_pos=int(EventPos.RANGE_START))
        if ev.range_id in range_has_step:
            new_pmode = (
                int(EventPairMode.NAME_ONLY)
                if ev.pair_mode == int(EventPairMode.INVALID)
                else ev.pair_mode
            )
            ev = replace(ev, range_type=int(RangeType.MULTI_EVENT), pair_mode=new_pmode)
        normalized_by_token[token] = ev

    by_key: dict[tuple[str, str], Event] = {}
    for token in ordered_tokens:
        event = normalized_by_token[token]
        key = (event.kind, event.name)
        previous_key = by_key.get(key)
        if previous_key is not None:
            if replace(previous_key, event_id=event.event_id) != event:
                raise ValueError(f"Conflicting IKET metadata for event {event.name!r}")
            continue
        by_key[key] = event

    effective_method = _resolve_effective_method(len(by_key), instrument_method)
    next_event_id = (
        _EXTENDED_START_EVENT_ID
        if effective_method == int(InstrumentMethod.EXTENDED_NATIVE_DUMP)
        else 1
    )

    canonical_by_key: dict[tuple[str, str], Event] = {}
    for key, event in by_key.items():
        while next_event_id == _RANGE_END_EVENT_ID:
            next_event_id += 1
        canonical_by_key[key] = replace(event, event_id=next_event_id)
        next_event_id += 1

    token_ids: dict[str, int] = {}
    canonical_tokens: dict[str, str] = {}
    for token in ordered_tokens:
        event = normalized_by_token[token]
        canonical = canonical_by_key[(event.kind, event.name)]
        token_ids[token] = canonical.event_id
        canonical_tokens[token] = encode_event(canonical)

    def rewrite_id(match: re.Match[str]) -> str:
        token = match.group("token")
        suffix = match.group("suffix").replace(token, canonical_tokens[token])
        return match.group("prefix") + str(token_ids[token]) + suffix

    return _EVENT_CALL_PATTERN.sub(rewrite_id, code), list(canonical_by_key.values()), effective_method


def _safe_symbol(name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not safe or safe[0].isdigit():
        safe = "_" + safe
    return safe


def _u32(value: int) -> list[int]:
    return [(value >> shift) & 0xFF for shift in (0, 8, 16, 24)]


def _attrs(size: int, assignments: dict[int, list[int]], name: str | None = None) -> list[int]:
    data = [0] * size
    data[0:4] = _u32(size)
    for offset, bytes_ in assignments.items():
        data[offset : offset + len(bytes_)] = bytes_
    if name is not None:
        encoded = name.encode("utf-8")
        if len(encoded) > size - 28:
            raise ValueError(f"IKET event name is too long for minimal metadata: {name!r}")
        data[24:28] = _u32(len(encoded))
        data[28 : 28 + len(encoded)] = list(encoded)
    return data


def _range_attrs(
    name: str,
    range_id: int,
    *,
    range_type: int = int(RangeType.PUSH_POP),
    range_scope: int = int(RangeScope.INTRA_WARP),
    pair_mode: int = int(EventPairMode.INVALID),
    color: int = int(Colors.AUTO),
    runtime_payloads: bool = False,
) -> list[int]:
    encoded = name.encode("utf-8")
    size = 72
    if len(encoded) > size - 36:
        raise ValueError(f"IKET range name is too long for minimal metadata: {name!r}")
    effective_pair_mode = pair_mode
    if effective_pair_mode == int(EventPairMode.NAME_AND_PAYLOAD) and not runtime_payloads:
        effective_pair_mode = int(EventPairMode.NAME_ONLY)
    data = [0] * size
    data[0:4] = _u32(size)
    data[4:8] = _u32(range_scope & 0xFFFFFFFF)
    data[8:12] = _u32(range_id)
    data[12:16] = _u32(color & 0xFFFFFFFF)
    data[16:20] = _u32(range_type & 0xFFFFFFFF)
    data[20:24] = _u32(effective_pair_mode & 0xFFFFFFFF)
    data[32:36] = _u32(len(encoded))
    data[36 : 36 + len(encoded)] = list(encoded)
    return data


def _c_array(symbol: str, data: list[int]) -> str:
    values = ", ".join(str(x) for x in data)
    return f"__device__ __attribute__((used, aligned(1))) unsigned char {symbol}[{len(data)}] = {{{values}}};"


def _metadata_decls(
    events: Iterable[Event],
    *,
    runtime_payloads: bool,
    instrument_method: int = int(InstrumentMethod.NATIVE_DUMP),
) -> str:
    events = list(events)
    if not events:
        return ""

    lines = ['extern "C" {']
    ranges: dict[int, _RangeMeta] = {}
    max_event_id = 0

    for event in events:
        max_event_id = max(max_event_id, event.event_id)
        payload_id = (
            event.payload_iket_id
            if (runtime_payloads and instrument_method != int(InstrumentMethod.UTRACE_EVENT))
            else 0
        )
        assignments = {
            4: _u32(event.event_id),
            8: _u32(instrument_method),
            12: _u32(payload_id),
        }
        if event.range_id != 0:
            assignments[16] = _u32(event.event_pos or int(EventPos.RANGE_START))
            assignments[20] = _u32(event.range_id)
            rname = event.range_name or event.name
            rtype = event.range_type or int(RangeType.PUSH_POP)
            prev_range = ranges.get(event.range_id)
            if prev_range is not None and prev_range.range_type == int(RangeType.MULTI_EVENT):
                rtype = int(RangeType.MULTI_EVENT)
            ranges[event.range_id] = _RangeMeta(
                name=rname,
                range_id=event.range_id,
                range_type=rtype,
                range_scope=event.range_scope,
                pair_mode=event.pair_mode,
                color=event.color,
            )
        symbol = f"__iket_evt_decl_{_safe_symbol(event.name)}_{event.event_id}_attrs"
        lines.append(_c_array(symbol, _attrs(60, assignments, event.name)))

    for rmeta in ranges.values():
        symbol = f"__iket_range_decl_{_safe_symbol(rmeta.name)}_{rmeta.range_id}_attrs"
        lines.append(
            _c_array(
                symbol,
                _range_attrs(
                    rmeta.name,
                    rmeta.range_id,
                    range_type=rmeta.range_type,
                    range_scope=rmeta.range_scope,
                    pair_mode=rmeta.pair_mode,
                    color=rmeta.color,
                    runtime_payloads=runtime_payloads,
                ),
            )
        )

    meta_max_eid = (
        _EXTENDED_MAX_EVENT_ID
        if instrument_method == int(InstrumentMethod.EXTENDED_NATIVE_DUMP)
        else _RANGE_END_EVENT_ID
    )
    meta = [0] * 48
    meta[0:4] = _u32(48)
    meta[8:12] = _u32(len(events) + len(ranges))
    meta[12:16] = _u32(meta_max_eid)
    meta[16:20] = _u32(32)
    meta[20:24] = _u32(60)
    meta[24:28] = _IKET_MAGIC
    meta[32:36] = _u32(max_event_id)
    lines.append(_c_array("__iket_meta_info", meta))
    lines.append("}")
    return "\n".join(lines)


def _target_sm(target: Any) -> int | None:
    attrs = getattr(target, "attrs", None)
    arch = None
    if attrs is not None:
        try:
            arch = attrs.get("arch", None)
        except (AttributeError, KeyError, TypeError):
            arch = None
    candidates = [str(arch)] if arch is not None else []
    candidates.append(str(target))
    for candidate in candidates:
        match = re.search(r"(?:sm|compute)_([0-9]+)", candidate)
        if match is not None:
            return int(match.group(1))
    return None


def _cluster_rank_instruction(target: Any) -> str:
    sm = _target_sm(target)
    if sm is not None and sm >= 90:
        return "mov.b32 r, %cluster_ctarank;"
    return "mov.u32 r, 0;"


def _event_macros(
    target: Any,
    *,
    runtime_payloads: bool,
    instrument_method: int = int(InstrumentMethod.NATIVE_DUMP),
) -> str:
    rank_instruction = _cluster_rank_instruction(target)
    if instrument_method == int(InstrumentMethod.EXTENDED_NATIVE_DUMP):
        event = rf"""
#define TL_IKET_EVENT(ID, ...) \
  asm volatile("{{\n\t" \
               ".reg .b32 r, t, evtid;\n\t" \
               ".reg .b64 ts_evtid;\n\t" \
               "{rank_instruction}\n\t" \
               "mov.u32 t, %globaltimer_lo;\n\t" \
               "mad.lo.u32 r, r, 0x1000000, 0x20;\n\t" \
               "mov.b32 evtid, " #ID ";\n\t" \
               "mov.b64 ts_evtid, {{t, evtid}};\n\t" \
               "st.weak.shared.u64 [r], ts_evtid;\n\t" \
               "pmevent.mask " #ID ";\n\t" \
               "}}" ::: "memory")
"""
        if not runtime_payloads:
            return (
                event
                + r"""
#define TL_IKET_EVENT_PAYLOAD_U32(ID, VALUE, ...) TL_IKET_EVENT(ID)
#define TL_IKET_EVENT_PAYLOAD_F32(ID, VALUE, ...) TL_IKET_EVENT(ID)
#define TL_IKET_EVENT_PAYLOAD_F16(ID, VALUE, ...) TL_IKET_EVENT(ID)
#define TL_IKET_EVENT_PAYLOAD_BF16(ID, VALUE, ...) TL_IKET_EVENT(ID)
#define TL_IKET_EVENT_PAYLOAD_U64(ID, VALUE, ...) TL_IKET_EVENT(ID)
#define TL_IKET_EVENT_PAYLOAD_F64(ID, VALUE, ...) TL_IKET_EVENT(ID)
#define TL_IKET_EVENT_PAYLOAD_PTR(ID, VALUE, ...) TL_IKET_EVENT(ID)
"""
            )
        payload = rf"""
#define TL_IKET_EVENT_PAYLOAD_U32(ID, VALUE, ...) do {{ \
  unsigned int __tl_iket_payload_u32 = (unsigned int)(VALUE); \
  asm volatile("{{\n\t" \
               ".reg .b32 r, t, evtid;\n\t" \
               ".reg .b64 ts_evtid;\n\t" \
               "{rank_instruction}\n\t" \
               "mov.u32 t, %globaltimer_lo;\n\t" \
               "mad.lo.u32 r, r, 0x1000000, 0x20;\n\t" \
               "mov.b32 evtid, " #ID ";\n\t" \
               "mov.b64 ts_evtid, {{t, evtid}};\n\t" \
               "st.weak.shared.u64 [r], ts_evtid;\n\t" \
               "}}" ::: "memory"); \
  asm volatile("{{\n\t" \
               ".reg .b32 r, p;\n\t" \
               "{rank_instruction}\n\t" \
               "mov.u32 p, %0;\n\t" \
               "mad.lo.u32 r, r, 0x1000000, 0x28;\n\t" \
               "st.volatile.shared.u32 [r], p;\n\t" \
               "pmevent.mask " #ID ";\n\t" \
               "}}" :: "r"(__tl_iket_payload_u32) : "memory"); \
}} while (0)

#define TL_IKET_EVENT_PAYLOAD_U64(ID, VALUE, ...) do {{ \
  unsigned long long __tl_iket_payload_u64 = (unsigned long long)(VALUE); \
  asm volatile("{{\n\t" \
               ".reg .b32 r, t, evtid;\n\t" \
               ".reg .b64 ts_evtid;\n\t" \
               "{rank_instruction}\n\t" \
               "mov.u32 t, %globaltimer_lo;\n\t" \
               "mad.lo.u32 r, r, 0x1000000, 0x20;\n\t" \
               "mov.b32 evtid, " #ID ";\n\t" \
               "mov.b64 ts_evtid, {{t, evtid}};\n\t" \
               "st.weak.shared.u64 [r], ts_evtid;\n\t" \
               "}}" ::: "memory"); \
  asm volatile("{{\n\t" \
               ".reg .b32 r;\n\t" \
               ".reg .b64 p;\n\t" \
               "{rank_instruction}\n\t" \
               "mov.b64 p, %0;\n\t" \
               "mad.lo.u32 r, r, 0x1000000, 0x28;\n\t" \
               "st.volatile.shared.u64 [r], p;\n\t" \
               "pmevent.mask " #ID ";\n\t" \
               "}}" :: "l"(__tl_iket_payload_u64) : "memory"); \
}} while (0)
"""
    else:
        event = rf"""
#define TL_IKET_EVENT(ID, ...) \
  asm volatile("{{\n\t" \
               ".reg .b32 r, t;\n\t" \
               "{rank_instruction}\n\t" \
               "mov.u32 t, %globaltimer_lo;\n\t" \
               "or.b32 t, t, " #ID ";\n\t" \
               "mad.lo.u32 r, r, 0x1000000, 0x20;\n\t" \
               "st.weak.shared.u32 [r], t;\n\t" \
               "pmevent.mask " #ID ";\n\t" \
               "}}" ::: "memory")
"""
        if not runtime_payloads or instrument_method == int(InstrumentMethod.UTRACE_EVENT):
            return (
                event
                + r"""
#define TL_IKET_EVENT_PAYLOAD_U32(ID, VALUE, ...) TL_IKET_EVENT(ID)
#define TL_IKET_EVENT_PAYLOAD_F32(ID, VALUE, ...) TL_IKET_EVENT(ID)
#define TL_IKET_EVENT_PAYLOAD_F16(ID, VALUE, ...) TL_IKET_EVENT(ID)
#define TL_IKET_EVENT_PAYLOAD_BF16(ID, VALUE, ...) TL_IKET_EVENT(ID)
#define TL_IKET_EVENT_PAYLOAD_U64(ID, VALUE, ...) TL_IKET_EVENT(ID)
#define TL_IKET_EVENT_PAYLOAD_F64(ID, VALUE, ...) TL_IKET_EVENT(ID)
#define TL_IKET_EVENT_PAYLOAD_PTR(ID, VALUE, ...) TL_IKET_EVENT(ID)
"""
            )
        payload = rf"""
#define TL_IKET_EVENT_PAYLOAD_U32(ID, VALUE, ...) do {{ \
  unsigned int __tl_iket_payload_u32 = (unsigned int)(VALUE); \
  /* Separate 32-bit stores avoid an STS.64 record rejected by IKET NativeDump. */ \
  asm volatile("{{\n\t" \
               ".reg .b32 r, t;\n\t" \
               "{rank_instruction}\n\t" \
               "mov.u32 t, %globaltimer_lo;\n\t" \
               "or.b32 t, t, " #ID ";\n\t" \
               "mad.lo.u32 r, r, 0x1000000, 0x20;\n\t" \
               "st.weak.shared.u32 [r], t;\n\t" \
               "}}" ::: "memory"); \
  asm volatile("{{\n\t" \
               ".reg .b32 r, p;\n\t" \
               "{rank_instruction}\n\t" \
               "mov.u32 p, %0;\n\t" \
               "mad.lo.u32 r, r, 0x1000000, 0x24;\n\t" \
               "st.volatile.shared.u32 [r], p;\n\t" \
               "pmevent.mask " #ID ";\n\t" \
               "}}" :: "r"(__tl_iket_payload_u32) : "memory"); \
}} while (0)

#define TL_IKET_EVENT_PAYLOAD_U64(ID, VALUE, ...) do {{ \
  unsigned long long __tl_iket_payload_u64 = (unsigned long long)(VALUE); \
  asm volatile("{{\n\t" \
               ".reg .b32 r, t;\n\t" \
               "{rank_instruction}\n\t" \
               "mov.u32 t, %globaltimer_lo;\n\t" \
               "or.b32 t, t, " #ID ";\n\t" \
               "mad.lo.u32 r, r, 0x1000000, 0x20;\n\t" \
               "st.weak.shared.u32 [r], t;\n\t" \
               "}}" ::: "memory"); \
  asm volatile("{{\n\t" \
               ".reg .b32 r;\n\t" \
               ".reg .b64 p;\n\t" \
               "{rank_instruction}\n\t" \
               "mov.b64 p, %0;\n\t" \
               "mad.lo.u32 r, r, 0x1000000, 0x28;\n\t" \
               "st.volatile.shared.u64 [r], p;\n\t" \
               "pmevent.mask " #ID ";\n\t" \
               "}}" :: "l"(__tl_iket_payload_u64) : "memory"); \
}} while (0)
"""

    type_wrappers = r"""
#define TL_IKET_EVENT_PAYLOAD_F32(ID, VALUE, ...) do { \
  union { float f; unsigned int u; } __tl_iket_payload_f32_bits; \
  __tl_iket_payload_f32_bits.f = (float)(VALUE); \
  TL_IKET_EVENT_PAYLOAD_U32(ID, __tl_iket_payload_f32_bits.u); \
} while (0)

#define TL_IKET_EVENT_PAYLOAD_F16(ID, VALUE, ...) do { \
  union { half h; unsigned short u; } __tl_iket_payload_f16_bits; \
  __tl_iket_payload_f16_bits.h = (half)(VALUE); \
  TL_IKET_EVENT_PAYLOAD_U32(ID, (unsigned int)__tl_iket_payload_f16_bits.u); \
} while (0)

#define TL_IKET_EVENT_PAYLOAD_BF16(ID, VALUE, ...) do { \
  union { __nv_bfloat16 b; unsigned short u; } __tl_iket_payload_bf16_bits; \
  __tl_iket_payload_bf16_bits.b = (__nv_bfloat16)(VALUE); \
  TL_IKET_EVENT_PAYLOAD_U32(ID, (unsigned int)__tl_iket_payload_bf16_bits.u); \
} while (0)

#define TL_IKET_EVENT_PAYLOAD_F64(ID, VALUE, ...) do { \
  union { double d; unsigned long long u; } __tl_iket_payload_f64_bits; \
  __tl_iket_payload_f64_bits.d = (double)(VALUE); \
  TL_IKET_EVENT_PAYLOAD_U64(ID, __tl_iket_payload_f64_bits.u); \
} while (0)

#define TL_IKET_EVENT_PAYLOAD_PTR(ID, VALUE, ...) \
  TL_IKET_EVENT_PAYLOAD_U64(ID, (unsigned long long)(VALUE))
"""
    return event + payload + type_wrappers


def inject_iket_cuda(
    code: str,
    target: Any,
    *,
    runtime_payloads: bool,
    instrument_method: str | int | InstrumentMethod | None = "auto",
) -> str:
    """Inject IKET declarations and event macros into generated CUDA source."""
    if _INJECTION_MARKER in code:
        return code
    code, events, effective_method = _canonicalize_cuda_events(code, instrument_method=instrument_method)
    if not events:
        return code

    insert_at = code.find("#include <tl_templates")
    if insert_at < 0:
        insert_at = code.find("#include")
    if insert_at < 0:
        insert_at = 0

    prefix = (
        f"// {_INJECTION_MARKER}\n"
        "// Experimental IKET metadata emitted by tilelang.tools.cuda.iket.\n"
        + _metadata_decls(
            events,
            runtime_payloads=runtime_payloads,
            instrument_method=effective_method,
        )
        + "\n"
        + _event_macros(
            target,
            runtime_payloads=runtime_payloads,
            instrument_method=effective_method,
        )
        + "\n"
    )
    return code[:insert_at] + prefix + code[insert_at:]
