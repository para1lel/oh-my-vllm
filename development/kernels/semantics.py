"""Development HIR for complete operator behavior, independent of kernel tiling.

These representative shapes describe arithmetic and returned state. Existing
project tests remain authoritative for strides, ragged metadata, aliasing, graph
replay, all supported dtypes, and numerical tolerances. Static analysis describes
this HIR, not the generated CUDA kernel's actual traffic or execution time.
"""

from tilefoundry import DType, func, module
from tilefoundry.dsl import Tensor, Topology, tf
from tilefoundry.target import CudaTarget

INFINITY = float("inf")
Q_SCALE = 128**-0.5
ATTENTION_SCALE = 256**-0.5


# A single logical analysis unit, not the TileLang kernel's actual CTA placement.
@module(
    entry="silu",
    target=CudaTarget("nvidia.b200_sxm"),
    topologies=(Topology("cta", 1),),
)
class Operators:
    @func
    def silu(x: Tensor[(2, 512), DType.bf16]):
        gate = tf.cast(x[:, :256], "f32")
        activated = tf.cast(tf.silu(gate), "bf16")
        return tf.cast(tf.cast(activated, "f32") * tf.cast(x[:, 256:], "f32"), "bf16")

    @func
    def gates(
        ba: Tensor[(2, 96), DType.bf16],
        log: Tensor[(48,), DType.f32],
        bias: Tensor[(48,), DType.f32],
    ):
        b = tf.cast(ba[:, :48], "f32")
        a = tf.cast(ba[:, 48:], "f32") + bias
        return -tf.exp(log) * tf.softplus(a), tf.sigmoid(b)

    @func
    def norm(x: Tensor[(2, 5120), DType.bf16], weight: Tensor[(5120,), DType.f32]):
        values = tf.cast(x, "f32")
        mean = tf.reduce(tf.square(values), (-1,), True, "mean")
        return tf.cast(values * tf.rsqrt(mean + 1e-6) * weight, "bf16")

    @func
    def add_norm(
        x: Tensor[(2, 5120), DType.bf16],
        residual: Tensor[(2, 5120), DType.bf16],
        weight: Tensor[(5120,), DType.f32],
    ):
        summed = tf.cast(tf.cast(x, "f32") + tf.cast(residual, "f32"), "bf16")
        return summed, norm(summed, weight)  # noqa: F821

    @func
    def gated_norm(
        x: Tensor[(2, 128), DType.bf16],
        weight: Tensor[(128,), DType.f32],
        gate: Tensor[(2, 128), DType.bf16],
    ):
        values = tf.cast(x, "f32")
        mean = tf.reduce(tf.square(values), (-1,), True, "mean")
        return tf.cast(
            values * tf.rsqrt(mean + 1e-6) * weight * tf.silu(tf.cast(gate, "f32")),
            "bf16",
        )

    @func
    def rope(x: Tensor[(2, 24, 256), DType.bf16], positions: Tensor[(2,), DType.i64]):
        exponents = tf.cast(tf.arange(Tensor[(32,), DType.i64]), "f32") * (-2.0 / 64.0)
        freq = tf.exp(exponents * 16.11809565095832)
        angle = tf.reshape(tf.cast(positions, "f32"), (2, 1, 1)) * tf.reshape(
            freq, (1, 1, 32)
        )
        left = tf.cast(x[:, :, :32], "f32")
        right = tf.cast(x[:, :, 32:64], "f32")
        a = tf.cast(left * tf.cos(angle) - right * tf.sin(angle), "bf16")
        b = tf.cast(right * tf.cos(angle) + left * tf.sin(angle), "bf16")
        return tf.concat([a, b, x[:, :, 64:]], axis=2)

    @func
    def norm_rope(
        x: Tensor[(2, 24, 256), DType.bf16],
        weight: Tensor[(256,), DType.f32],
        positions: Tensor[(2,), DType.i64],
    ):
        # TileFoundry has no f64 dtype. This HIR expresses the logical operator
        # with f32 phase arithmetic, not production's precise phase reduction.
        # Its cost/check results cannot validate maximum-context phase accuracy;
        # use the independent FP64 reference for that numerical boundary.
        values = tf.cast(x, "f32")
        mean = tf.reduce(tf.square(values), (-1,), True, "mean")
        normalized = tf.cast(values * tf.rsqrt(mean + 1e-6) * weight, "bf16")
        return rope(normalized, positions)  # noqa: F821

    @func
    def quant(x: Tensor[(2, 256), DType.bf16]):
        values = tf.reshape(tf.cast(x, "f32"), (2, 2, 128))
        maximum = tf.reduce(values, (-1,), True, "abs_max")
        scale = tf.clamp(maximum, 1e-10, INFINITY) / 448.0
        data = tf.cast(tf.clamp(values / scale, -448.0, 448.0), "fp8e4m3")
        return tf.reshape(data, (2, 256)), tf.reshape(scale, (2, 2))

    @func
    def silu_quant(x: Tensor[(2, 512), DType.bf16]):
        return quant(silu(x))  # noqa: F821

    @func
    def qk(q: Tensor[(2, 2, 128), DType.bf16], k: Tensor[(2, 2, 128), DType.bf16]):
        qf = tf.cast(q, "f32")
        kf = tf.cast(k, "f32")
        qs = tf.reduce(tf.square(qf), (-1,), True, "sum")
        ks = tf.reduce(tf.square(kf), (-1,), True, "sum")
        return tf.cast(qf * tf.rsqrt(qs + 1e-6), "bf16"), tf.cast(
            kf * tf.rsqrt(ks + 1e-6), "bf16"
        )

    @func
    def delta_step(
        q: Tensor[(1, 2, 128), DType.bf16],
        k: Tensor[(1, 2, 128), DType.bf16],
        v: Tensor[(1, 6, 128), DType.bf16],
        g: Tensor[(1, 6), DType.f32],
        beta: Tensor[(1, 6), DType.f32],
        state: Tensor[(6, 128, 128), DType.f32],
    ):
        qf = tf.cast(q, "f32")
        kf = tf.cast(k, "f32")
        qn = (
            qf * tf.rsqrt(tf.reduce(tf.square(qf), (-1,), True, "sum") + 1e-6) * Q_SCALE
        )
        kn = kf * tf.rsqrt(tf.reduce(tf.square(kf), (-1,), True, "sum") + 1e-6)
        qh = tf.reshape(tf.repeat_interleave(qn, 3, axis=1), (6, 1, 128))
        kh = tf.reshape(tf.repeat_interleave(kn, 3, axis=1), (6, 1, 128))
        decayed = state * tf.reshape(tf.exp(g), (6, 1, 1))
        recalled = tf.reduce(decayed * kh, (-1,), False, "sum")
        residual = (tf.reshape(tf.cast(v, "f32"), (6, 128)) - recalled) * tf.reshape(
            beta, (6, 1)
        )
        updated = decayed + tf.reshape(residual, (6, 128, 1)) * kh
        output = tf.cast(tf.reduce(updated * qh, (-1,), False, "sum"), "bf16")
        return tf.reshape(output, (1, 6, 128)), updated

    @func
    def recurrent(
        q: Tensor[(2, 2, 128), DType.bf16],
        k: Tensor[(2, 2, 128), DType.bf16],
        v: Tensor[(2, 6, 128), DType.bf16],
        g: Tensor[(2, 6), DType.f32],
        beta: Tensor[(2, 6), DType.f32],
        state: Tensor[(6, 128, 128), DType.f32],
    ):
        o0, s0 = delta_step(  # noqa: F821
            q[:1, :, :], k[:1, :, :], v[:1, :, :], g[:1, :], beta[:1, :], state
        )
        o1, s1 = delta_step(  # noqa: F821
            q[1:, :, :], k[1:, :, :], v[1:, :, :], g[1:, :], beta[1:, :], s0
        )
        return tf.concat([o0, o1], axis=0), tf.concat(
            [tf.reshape(s0, (1, 6, 128, 128)), tf.reshape(s1, (1, 6, 128, 128))], axis=0
        )

    @func
    def convolution(
        x: Tensor[(2, 128), DType.bf16],
        weight: Tensor[(128, 4), DType.bf16],
        state: Tensor[(128, 3), DType.bf16],
    ):
        window0 = tf.concat([state, tf.transpose(x[:1, :], (1, 0))], axis=1)
        s0 = window0[:, 1:]
        window1 = tf.concat([s0, tf.transpose(x[1:, :], (1, 0))], axis=1)
        s1 = window1[:, 1:]
        y0 = tf.cast(
            tf.silu(
                tf.reduce(
                    tf.cast(window0, "f32") * tf.cast(weight, "f32"),
                    (-1,),
                    False,
                    "sum",
                )
            ),
            "bf16",
        )
        y1 = tf.cast(
            tf.silu(
                tf.reduce(
                    tf.cast(window1, "f32") * tf.cast(weight, "f32"),
                    (-1,),
                    False,
                    "sum",
                )
            ),
            "bf16",
        )
        return tf.concat(
            [tf.reshape(y0, (1, 128)), tf.reshape(y1, (1, 128))], axis=0
        ), tf.concat([tf.reshape(s0, (1, 128, 3)), tf.reshape(s1, (1, 128, 3))], axis=0)

    @func
    def append(
        k: Tensor[(783, 4, 256), DType.bf16],
        v: Tensor[(783, 4, 256), DType.bf16],
        new_k: Tensor[(2, 4, 256), DType.bf16],
        new_v: Tensor[(2, 4, 256), DType.bf16],
    ):
        return tf.concat([k, new_k], axis=0), tf.concat([v, new_v], axis=0)

    @func
    def attention(
        q: Tensor[(1, 24, 256), DType.bf16],
        k: Tensor[(784, 4, 256), DType.bf16],
        v: Tensor[(784, 4, 256), DType.bf16],
    ):
        qh = tf.transpose(tf.cast(q, "f32"), (1, 0, 2))
        kh = tf.transpose(tf.cast(tf.repeat_interleave(k, 6, axis=1), "f32"), (1, 2, 0))
        vh = tf.transpose(tf.cast(tf.repeat_interleave(v, 6, axis=1), "f32"), (1, 0, 2))
        logits = tf.matmul(qh, kh) * ATTENTION_SCALE
        probability = tf.softmax(logits, axis=-1)
        output = tf.matmul(tf.cast(tf.cast(probability, "bf16"), "f32"), vh)
        return tf.cast(tf.transpose(output, (1, 0, 2)), "bf16")
