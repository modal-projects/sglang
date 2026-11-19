"""PTX helpers shared by the Blackwell fused multi-LoRA kernels (CuTe DSL inline asm)."""

import cutlass
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm


def _asm(ret, args, text, cons):
    return llvm.inline_asm(ret, args, text, cons, has_side_effects=True,
                           is_align_stack=False, asm_dialect=0)


def atomic_add_acq_rel(ptr, value):
    return cutlass.Int32(_asm(ir.IntegerType.get_signless(32),
                              [ptr.toint().ir_value(), value.ir_value()],
                              "atom.acq_rel.gpu.global.add.u32 $0,[$1],$2;",
                              "=r,l,r,~{memory}"))


def dsmem_load1(smem_ptr, rank):
    i32, f32 = ir.IntegerType.get_signless(32), ir.F32Type.get()
    remote = _asm(i32, [smem_ptr.toint().ir_value(), rank.ir_value()],
                  "mapa.shared::cluster.u32 $0,$1,$2;", "=r,r,r")
    return cutlass.Float32(_asm(f32, [remote], "ld.shared::cluster.f32 $0,[$1];",
                                "=f,r,~{memory}"))


SPIN_LIMIT_CYCLES = 1 << 32  # ~2 s: only a lost producer or a deadlock spins this long


def wait_count(ptr, wanted, equal=False):
    """Spin (ld.acquire.gpu) until the monotonic counter at ``ptr`` reaches ``wanted`` (or,
    with ``equal``, equals it). The comparison is on the signed difference, so it stays
    correct when the int32 counter and ``wanted`` wrap. Traps after SPIN_LIMIT_CYCLES instead
    of continuing with stale data. Call from one lane."""
    test = "setp.eq.s32 %spp, %spd, 0;" if equal else "setp.ge.s32 %spp, %spd, 0;"
    llvm.inline_asm(
        None,
        [ptr.toint().ir_value(), cutlass.Int32(wanted).ir_value()],
        "{\n"
        ".reg .pred %spp;\n.reg .b32 %spv, %spd;\n.reg .b64 %spt0, %spt1;\n"
        "mov.u64 %spt0, %clock64;\n"
        "SPIN${:uid}:\n"
        "ld.acquire.gpu.global.u32 %spv, [$0];\n"
        "sub.s32 %spd, %spv, $1;\n"
        + test + "\n"
        "@%spp bra DONE${:uid};\n"
        "mov.u64 %spt1, %clock64;\n"
        "sub.u64 %spt1, %spt1, %spt0;\n"
        f"setp.gt.u64 %spp, %spt1, {SPIN_LIMIT_CYCLES};\n"
        "@%spp trap;\n"
        "bra SPIN${:uid};\n"
        "DONE${:uid}:\n"
        "}",
        "l,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=0,
    )


def global_ns():
    return cutlass.Int64(
        llvm.inline_asm(
            ir.IntegerType.get_signless(64),
            [],
            "mov.u64 $0,%globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=0,
        )
    )


def warp_sum(x):
    """Butterfly sum over the 32 lanes (warp-uniform call sites only)."""
    body = "{.reg .f32 v,t; mov.f32 v,$1; "
    for mask in (16, 8, 4, 2, 1):
        body += f"shfl.sync.bfly.b32 t,v,{mask},31,-1; add.rn.f32 v,v,t; "
    body += "mov.f32 $0,v;}"
    return cutlass.Float32(llvm.inline_asm(ir.F32Type.get(), [x.ir_value()], body, "=f,f",
                                           has_side_effects=False, is_align_stack=False,
                                           asm_dialect=0))
