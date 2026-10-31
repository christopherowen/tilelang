from __future__ import annotations

import re

import pytest
import tvm_ffi

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.cache import _dispatch_map
from tilelang.env import CacheState
from tilelang.tools.cuda import iket
import tilelang.testing


_CUDA_POSTPROC = "tilelang_callback_cuda_postproc"
_IKET_SOURCE_MARKER = "// __tilelang_iket_tool__"
_SM90_TARGET = {"kind": "cuda", "arch": "sm_90"}


def _remove_cuda_postproc() -> None:
    if tvm_ffi.get_global_func(_CUDA_POSTPROC, allow_missing=True) is not None:
        tvm_ffi.remove_global_func(_CUDA_POSTPROC)


@pytest.fixture(autouse=True)
def _isolated_iket_state():
    # IKET owns process-global frontend and TVM callback state. Leave any callback
    # installed by the surrounding test process exactly as it was on entry.
    for _ in range(16):
        if not iket.is_enabled():
            break
        iket.disable(restore=True)

    previous = tvm_ffi.get_global_func(_CUDA_POSTPROC, allow_missing=True)
    _remove_cuda_postproc()
    iket.reset()
    iket.disable_runtime_payloads()
    iket.set_instrument_method(None)

    yield

    for _ in range(16):
        if not iket.is_enabled():
            break
        iket.disable(restore=False)
    iket.reset()
    iket.disable_runtime_payloads()
    iket.set_instrument_method(None)
    _remove_cuda_postproc()
    if previous is not None:
        tvm_ffi.register_global_func(_CUDA_POSTPROC, f=previous, override=True)


def _make_mark_kernel(event_name: str):
    @T.prim_func
    def kernel(A: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32):
            iket.mark(event_name)
            A[0] = A[0]

    return kernel


def _make_range_kernel():
    @T.prim_func
    def kernel(A: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32), iket.range("compute_phase"):
            iket.mark("compute_tick")
            A[0] = A[0]

    return kernel


def _make_float_payload_kernel():
    @T.prim_func
    def kernel(A: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32):
            iket.mark("float_value", payload=iket.payload(A[0], dtype="float32"))
            A[0] = A[0]

    return kernel


def _lower_cuda_source(func, target=_SM90_TARGET) -> str:
    with tvm.transform.PassContext(), tvm.target.Target(target):
        artifact = tilelang.lower(func, target=target)
    assert artifact.kernel_source is not None
    return artifact.kernel_source


def _cache_key(func) -> str:
    return _dispatch_map["tvm_ffi"]._generate_key(
        func=func,
        out_idx=[0],
        execution_backend="tvm_ffi",
        args=(),
        target=tvm.target.Target(_SM90_TARGET),
    )


def _iket_calls(func) -> list[tvm.tirx.Call]:
    calls = []

    def collect(node):
        if not isinstance(node, tvm.tirx.Call) or not node.args:
            return
        extern_name = node.args[0]
        if isinstance(extern_name, tvm.tirx.StringImm) and extern_name.value.startswith("TL_IKET_EVENT"):
            calls.append(node)

    tvm.tirx.stmt_functor.post_order_visit(func.body, collect)
    return calls


def _event_metadata_bytes(source: str, event_name: str) -> list[int]:
    safe_name = re.sub(r"[^0-9A-Za-z_]", "_", event_name)
    declaration = re.search(
        rf"__iket_evt_decl_{re.escape(safe_name)}_\d+_attrs\[60\]\s*=\s*\{{([^}}]+)\}};",
        source,
    )
    assert declaration is not None
    return [int(value.strip()) for value in declaration.group(1).split(",")]


def _range_metadata_bytes(source: str, range_name: str) -> list[int]:
    safe_name = re.sub(r"[^0-9A-Za-z_]", "_", range_name)
    declaration = re.search(
        rf"__iket_range_decl_{re.escape(safe_name)}_\d+_attrs\[72\]\s*=\s*\{{([^}}]+)\}};",
        source,
    )
    assert declaration is not None
    return [int(value.strip()) for value in declaration.group(1).split(",")]


def _meta_info_bytes(source: str) -> list[int]:
    declaration = re.search(
        r"__iket_meta_info\[48\]\s*=\s*\{([^}]+)\};",
        source,
    )
    assert declaration is not None
    return [int(value.strip()) for value in declaration.group(1).split(",")]


def _u32_le(data: list[int], offset: int) -> int:
    return (
        data[offset]
        | (data[offset + 1] << 8)
        | (data[offset + 2] << 16)
        | (data[offset + 3] << 24)
    )


def test_iket_is_a_cuda_tool_not_a_language_namespace():
    assert not hasattr(T, "iket")
    assert iket.__name__ == "tilelang.tools.cuda.iket"


@tilelang.testing.requires_cuda
def test_event_metadata_is_preserved_in_tir_and_injected_source():
    with iket.session():
        kernel = _make_range_kernel()
        source = _lower_cuda_source(kernel)

    calls = _iket_calls(kernel)
    assert len(calls) == 3
    assert sum(isinstance(arg, tvm.tirx.StringImm) for call in calls for arg in call.args[1:]) >= 2
    assert _IKET_SOURCE_MARKER in source
    assert "__iket_meta_info" in source
    assert "__iket_evt_decl_compute_phase" in source
    assert "__iket_evt_decl_compute_tick" in source
    assert "__iket_range_decl_compute_phase" in source


@tilelang.testing.requires_cuda
def test_float_payload_macro_uses_distinct_temporaries():
    with iket.session(runtime_payloads=True):
        source = _lower_cuda_source(_make_float_payload_kernel())

    u32_macro = re.search(
        r"#define\s+TL_IKET_EVENT_PAYLOAD_U32\b.*?unsigned\s+int\s+(\w+)\s*=",
        source,
        flags=re.DOTALL,
    )
    f32_macro = re.search(
        r"#define\s+TL_IKET_EVENT_PAYLOAD_F32\b.*?union\s*\{.*?\}\s*(\w+)\s*;",
        source,
        flags=re.DOTALL,
    )

    assert u32_macro is not None
    assert f32_macro is not None
    assert u32_macro.group(1) != f32_macro.group(1)
    assert "TL_IKET_EVENT_PAYLOAD_F32" in source


@tilelang.testing.requires_cuda
def test_prebuilt_payload_primfunc_enables_runtime_payloads_during_lowering():
    assert not iket.runtime_payloads_enabled()
    kernel = _make_float_payload_kernel()
    calls = _iket_calls(kernel)
    assert len(calls) == 1
    assert calls[0].args[0].value == "TL_IKET_EVENT_PAYLOAD_F32"
    assert isinstance(calls[0].args[-1], tvm.tirx.StringImm)

    with iket.session(runtime_payloads=True):
        source = _lower_cuda_source(kernel)

    assert re.search(
        r"#define\s+TL_IKET_EVENT_PAYLOAD_F32\([^\n]+\)\s+do\s+\{",
        source,
    )
    assert "st.volatile.shared.u32 [r], p;" in source
    assert _event_metadata_bytes(source, "float_value")[12:16] == [13, 0, 0, 0]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("arch", ["sm_80", "sm_86", "sm_89"])
def test_cluster_rank_register_is_gated_by_cuda_architecture(arch):
    with iket.session():
        kernel = _make_mark_kernel("architecture_gate")
        pre_hopper_source = _lower_cuda_source(kernel, {"kind": "cuda", "arch": arch})
        sm90_source = _lower_cuda_source(kernel, _SM90_TARGET)

    assert _IKET_SOURCE_MARKER in pre_hopper_source
    assert "%cluster_ctarank" not in pre_hopper_source
    assert "mov.u32 r, 0;" in pre_hopper_source
    assert "%cluster_ctarank" in sm90_source


@tilelang.testing.requires_cuda
def test_prebuilt_primfunc_keeps_metadata_when_session_resets_registry():
    kernel = _make_mark_kernel("parsed_before_session")
    calls = _iket_calls(kernel)
    assert len(calls) == 1
    assert isinstance(calls[0].args[-1], tvm.tirx.StringImm)

    # session() resets process-local state by default. The generated source must
    # therefore recover its metadata from the already parsed PrimFunc.
    with iket.session():
        source = _lower_cuda_source(kernel)

    assert _IKET_SOURCE_MARKER in source
    assert "__iket_evt_decl_parsed_before_session" in source


def test_event_metadata_changes_kernel_cache_identity():
    first = _make_mark_kernel("cache_name_first")
    iket.reset()
    second = _make_mark_kernel("cache_name_second")

    assert first.script(show_meta=True) != second.script(show_meta=True)
    assert _cache_key(first) != _cache_key(second)


@tilelang.testing.requires_cuda
def test_independently_built_primfuncs_get_module_wide_event_ids():
    first = _make_mark_kernel("module_first").with_attr("global_symbol", "first")
    iket.reset()
    second = _make_mark_kernel("module_second").with_attr("global_symbol", "second")
    module = tvm.IRModule({"first": first, "second": second})

    with iket.session():
        source = _lower_cuda_source(module)

    assert "__iket_evt_decl_module_first_1_attrs" in source
    assert "__iket_evt_decl_module_second_2_attrs" in source
    assert re.search(r'TL_IKET_EVENT\(1, "__tl_iket_v1_', source)
    assert re.search(r'TL_IKET_EVENT\(2, "__tl_iket_v1_', source)


def test_event_name_limit_is_checked_during_primfunc_construction():
    with pytest.raises(ValueError, match="at most 32 UTF-8 bytes"):
        _make_mark_kernel("x" * 33)

    with pytest.raises(ValueError, match="at most 32 UTF-8 bytes"):
        _make_mark_kernel("界" * 11)


@tilelang.testing.requires_cuda
def test_nested_session_keeps_outer_callback_active():
    with iket.session():
        with iket.session():
            pass

        assert iket.is_enabled()
        source = _lower_cuda_source(_make_mark_kernel("after_inner_session"))
        assert "__iket_evt_decl_after_inner_session" in source

    assert not iket.is_enabled()
    assert tvm_ffi.get_global_func(_CUDA_POSTPROC, allow_missing=True) is None


def test_same_session_object_cannot_be_nested():
    previous_cache_state = CacheState.is_enabled()
    CacheState.enable()
    try:
        current = iket.session(runtime_payloads=True)
        with current:
            with pytest.raises(RuntimeError, match="cannot be entered"), current:
                pass
            assert iket.runtime_payloads_enabled()
            assert not CacheState.is_enabled()

        assert not iket.runtime_payloads_enabled()
        assert CacheState.is_enabled()
    finally:
        if previous_cache_state:
            CacheState.enable()
        else:
            CacheState.disable()


@pytest.mark.parametrize("initially_enabled", [True, False])
def test_nested_session_disables_and_restores_cache_state(initially_enabled):
    previous = CacheState.is_enabled()
    try:
        if initially_enabled:
            CacheState.enable()
        else:
            CacheState.disable()

        with iket.session():
            assert not CacheState.is_enabled()
            with iket.session():
                assert not CacheState.is_enabled()
            assert not CacheState.is_enabled()

        assert CacheState.is_enabled() is initially_enabled
    finally:
        if previous:
            CacheState.enable()
        else:
            CacheState.disable()


def test_session_restores_an_absent_callback():
    assert tvm_ffi.get_global_func(_CUDA_POSTPROC, allow_missing=True) is None

    with iket.session():
        assert tvm_ffi.get_global_func(_CUDA_POSTPROC, allow_missing=True) is not None

    assert tvm_ffi.get_global_func(_CUDA_POSTPROC, allow_missing=True) is None

    # An absent callback must not be restored as an identity callback: callers
    # must still be able to use override=False after the session exits.
    tvm_ffi.register_global_func(_CUDA_POSTPROC, f=lambda code, _target: code, override=False)


@tilelang.testing.requires_cuda
def test_session_restores_a_previous_callback():
    def previous(code, _target):
        return "// previous callback\n" + code

    tvm_ffi.register_global_func(_CUDA_POSTPROC, f=previous, override=False)

    with iket.session():
        source = _lower_cuda_source(_make_mark_kernel("with_previous_callback"))
        assert "__iket_evt_decl_with_previous_callback" in source

    restored = tvm_ffi.get_global_func(_CUDA_POSTPROC)
    assert restored("kernel source", tvm.target.Target(_SM90_TARGET)).startswith("// previous callback")


def test_failed_non_overriding_session_restores_host_state(tmp_path):
    def previous(code, _target):
        return code

    tvm_ffi.register_global_func(_CUDA_POSTPROC, f=previous, override=False)
    original_cache_state = CacheState.is_enabled()
    original_runtime_payloads = iket.runtime_payloads_enabled()

    with (
        pytest.raises(RuntimeError),
        iket.session(
            override=False,
            output_dir=tmp_path / "unused",
            runtime_payloads=not original_runtime_payloads,
        ),
    ):
        pass

    assert CacheState.is_enabled() == original_cache_state
    assert iket.runtime_payloads_enabled() == original_runtime_payloads
    restored = tvm_ffi.get_global_func(_CUDA_POSTPROC)
    assert restored("kernel source", tvm.target.Target(_SM90_TARGET)) == "kernel source"


@pytest.mark.parametrize(
    ("dtype_alias", "expected_dtype", "expected_id"),
    [
        ("i8", "int8", int(iket.PayloadType.I8)),
        ("uint8", "uint8", int(iket.PayloadType.UI8)),
        ("i16", "int16", int(iket.PayloadType.I16)),
        ("u16", "uint16", int(iket.PayloadType.UI16)),
        ("int32", "int32", int(iket.PayloadType.I32)),
        ("uint32", "uint32", int(iket.PayloadType.UI32)),
        ("i64", "int64", int(iket.PayloadType.I64)),
        ("u64", "uint64", int(iket.PayloadType.UI64)),
        ("fp4", "float4", int(iket.PayloadType.FP4)),
        ("float8_e4m3fn", "float8", int(iket.PayloadType.FP8)),
        ("fp16", "float16", int(iket.PayloadType.FP16)),
        ("bf16", "bfloat16", int(iket.PayloadType.BF16)),
        ("float32", "float32", int(iket.PayloadType.FP32)),
        ("fp64", "float64", int(iket.PayloadType.FP64)),
        ("ptr", "pointer", int(iket.PayloadType.POINTER)),
    ],
)
def test_extended_payload_dtypes_and_iket_ids(dtype_alias, expected_dtype, expected_id):
    spec = iket.payload(1, dtype=dtype_alias)
    assert spec.dtype == expected_dtype
    assert spec.iket_id == expected_id


@tilelang.testing.requires_cuda
def test_extended_payload_macros_and_64bit_stores():
    @T.prim_func
    def kernel(
        A: T.Tensor((1,), T.float16),
        B: T.Tensor((1,), T.bfloat16),
        C: T.Tensor((1,), T.int64),
        D: T.Tensor((1,), T.float64),
    ):
        with T.Kernel(1, threads=32):
            iket.mark("val_f16", payload=iket.payload(A[0], dtype="float16"))
            iket.mark("val_bf16", payload=iket.payload(B[0], dtype="bfloat16"))
            iket.mark("val_i64", payload=iket.payload(C[0], dtype="int64"))
            iket.mark("val_f64", payload=iket.payload(D[0], dtype="float64"))
            iket.mark("val_ptr", payload=iket.payload(C[0], dtype="pointer"))
            A[0] = A[0]

    with iket.session(runtime_payloads=True):
        source = _lower_cuda_source(kernel)

    assert _u32_le(_event_metadata_bytes(source, "val_f16"), 12) == int(iket.PayloadType.FP16)
    assert _u32_le(_event_metadata_bytes(source, "val_bf16"), 12) == int(iket.PayloadType.BF16)
    assert _u32_le(_event_metadata_bytes(source, "val_i64"), 12) == int(iket.PayloadType.I64)
    assert _u32_le(_event_metadata_bytes(source, "val_f64"), 12) == int(iket.PayloadType.FP64)
    assert _u32_le(_event_metadata_bytes(source, "val_ptr"), 12) == int(iket.PayloadType.POINTER)
    assert "TL_IKET_EVENT_PAYLOAD_F16" in source
    assert "TL_IKET_EVENT_PAYLOAD_BF16" in source
    assert "TL_IKET_EVENT_PAYLOAD_U64" in source
    assert "TL_IKET_EVENT_PAYLOAD_F64" in source
    assert "TL_IKET_EVENT_PAYLOAD_PTR" in source
    assert "st.volatile.shared.u64 [r], p;" in source


@tilelang.testing.requires_cuda
def test_start_end_range_single_event_and_non_lifo_interleaving():
    @T.prim_func
    def kernel(A: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32):
            iket.range_start("stage_a")
            iket.range_start("stage_b")
            A[0] = A[0]
            iket.range_end("stage_a")
            iket.range_end("stage_b")

    with iket.session():
        source = _lower_cuda_source(kernel)

    evt_a = _event_metadata_bytes(source, "stage_a")
    evt_b = _event_metadata_bytes(source, "stage_b")
    rng_a = _range_metadata_bytes(source, "stage_a")
    rng_b = _range_metadata_bytes(source, "stage_b")

    assert _u32_le(evt_a, 16) == int(iket.EventPos.RANGE_START_END)
    assert _u32_le(evt_b, 16) == int(iket.EventPos.RANGE_START_END)
    assert _u32_le(rng_a, 16) == int(iket.RangeType.START_END)
    assert _u32_le(rng_b, 16) == int(iket.RangeType.START_END)
    assert _u32_le(rng_a, 20) == int(iket.EventPairMode.NAME_ONLY)
    assert "__iket_evt_decl_stage_a_end" not in source


@tilelang.testing.requires_cuda
def test_start_end_range_with_dual_payloads_scope_pair_mode_and_color():
    @T.prim_func
    def kernel(A: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=32):
            with iket.range(
                "pipe_stage",
                payload=iket.payload(A[0], dtype="int32"),
                end_payload=iket.payload(A[0], dtype="int32"),
                scope="cta",
                pair_mode="name_and_payload",
                color="cyan",
            ):
                A[0] = A[0]

    with iket.session(runtime_payloads=True):
        source = _lower_cuda_source(kernel)

    evt_start = _event_metadata_bytes(source, "pipe_stage")
    evt_end = _event_metadata_bytes(source, "pipe_stage:end")
    rng = _range_metadata_bytes(source, "pipe_stage")

    assert _u32_le(evt_start, 16) == int(iket.EventPos.RANGE_START)
    assert _u32_le(evt_end, 16) == int(iket.EventPos.RANGE_END)
    assert _u32_le(evt_start, 20) == _u32_le(evt_end, 20) == _u32_le(rng, 8)
    assert _u32_le(evt_start, 12) == int(iket.PayloadType.I32)
    assert _u32_le(evt_end, 12) == int(iket.PayloadType.I32)
    assert _u32_le(rng, 4) == int(iket.RangeScope.INTER_WARP_IN_CTA)
    assert _u32_le(rng, 12) == int(iket.Colors.CYAN)
    assert _u32_le(rng, 16) == int(iket.RangeType.START_END)
    assert _u32_le(rng, 20) == int(iket.EventPairMode.NAME_AND_PAYLOAD)

    # Without runtime_payloads=True, NAME_AND_PAYLOAD automatically falls back to NAME_ONLY
    with iket.session(runtime_payloads=False):
        source_no_payload = _lower_cuda_source(kernel)
    rng_no_payload = _range_metadata_bytes(source_no_payload, "pipe_stage")
    assert _u32_le(rng_no_payload, 20) == int(iket.EventPairMode.NAME_ONLY)


@tilelang.testing.requires_cuda
def test_multi_event_range_with_range_step():
    @T.prim_func
    def kernel(A: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=32):
            with iket.range("tma_transfer"):
                iket.range_step("tma_transfer", "tma_arrive", payload=iket.payload(A[0], dtype="int32"))
                A[0] = A[0]

    with iket.session(runtime_payloads=True):
        source = _lower_cuda_source(kernel)

    evt_start = _event_metadata_bytes(source, "tma_transfer")
    evt_step = _event_metadata_bytes(source, "tma_arrive")
    evt_end = _event_metadata_bytes(source, "tma_transfer:end")
    rng = _range_metadata_bytes(source, "tma_transfer")

    assert _u32_le(rng, 16) == int(iket.RangeType.MULTI_EVENT)
    assert _u32_le(rng, 20) == int(iket.EventPairMode.NAME_ONLY)
    assert _u32_le(evt_start, 16) == int(iket.EventPos.RANGE_START)
    assert _u32_le(evt_step, 16) == int(iket.EventPos.RANGE_INTERNAL)
    assert _u32_le(evt_end, 16) == int(iket.EventPos.RANGE_END)
    assert _u32_le(evt_start, 20) == _u32_le(evt_step, 20) == _u32_le(evt_end, 20)


@tilelang.testing.requires_cuda
def test_extended_native_dump_auto_and_explicit():
    def _emit_many_marks(count: int) -> None:
        for idx in range(count):
            T.evaluate(iket.mark(f"evt_{idx:02d}"))

    @T.prim_func
    def many_events_kernel(A: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=32):
            _emit_many_marks(35)
            A[0] = A[0]

    with iket.session():
        source = _lower_cuda_source(many_events_kernel)

    meta = _meta_info_bytes(source)
    assert _u32_le(meta, 8) == 35
    assert _u32_le(meta, 12) == 4095
    assert _u32_le(meta, 32) == 64 + 34

    first_evt = _event_metadata_bytes(source, "evt_00")
    last_evt = _event_metadata_bytes(source, "evt_34")
    assert _u32_le(first_evt, 4) == 64
    assert _u32_le(first_evt, 8) == int(iket.InstrumentMethod.EXTENDED_NATIVE_DUMP)
    assert _u32_le(last_evt, 4) == 98
    assert "st.weak.shared.u64 [r], ts_evtid;" in source

    with (
        pytest.raises(ValueError, match="supports at most 30 events"),
        iket.session(instrument_method="native_dump"),
    ):
        _lower_cuda_source(many_events_kernel)

    with iket.session(instrument_method="extended", runtime_payloads=True):
        explicit_source = _lower_cuda_source(_make_float_payload_kernel())
    explicit_evt = _event_metadata_bytes(explicit_source, "float_value")
    assert _u32_le(explicit_evt, 4) == 64
    assert _u32_le(explicit_evt, 8) == int(iket.InstrumentMethod.EXTENDED_NATIVE_DUMP)
    assert "mad.lo.u32 r, r, 0x1000000, 0x28;" in explicit_source


def test_profile_command_extended_options():
    default_cmd = iket.profile_command(["python", "run.py"], directory="/tmp/iket_out")
    assert default_cmd == (
        "python -m iket.cli.main --output-dir /tmp/iket_out --clobber "
        "profile --postprocess all -- python run.py"
    )

    extended_cmd = iket.profile_command(
        ["python", "run.py"],
        directory="/tmp/iket_out",
        postprocess="json",
        context_buffer_size=256,
        log_level="info",
        working_dir="/tmp/iket_work",
        max_ts_cnt_per_warp=8192,
        use_tracker_dir="/tmp/iket_tracker",
        use_config="/tmp/iket_cfg.yaml",
        config_version="v1",
        warp_trace=True,
        keep=False,
    )
    assert "--context-buffer-size 256" in extended_cmd
    assert "--log-level info" in extended_cmd
    assert "--working-dir /tmp/iket_work" in extended_cmd
    assert "profile --postprocess json" in extended_cmd
    assert "--max-ts-cnt-per-warp 8192" in extended_cmd
    assert "--use-tracker-dir /tmp/iket_tracker" in extended_cmd
    assert "--use-config /tmp/iket_cfg.yaml" in extended_cmd
    assert "--config-version v1" in extended_cmd
    assert "--no-keep" in extended_cmd
    assert "--warp-trace" in extended_cmd
