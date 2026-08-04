FROM lmsysorg/sglang:dev AS base


RUN apt-get update \
    && apt-get install -y --no-install-recommends strace \
    && rm -rf /var/lib/apt/lists/*

COPY docker/flashinfer_jit_cache /opt/flashinfer-jit-cache

RUN set -eux; \
    artifact=/root/.cache/flashinfer/0.6.14/90a/cached_ops/fused_moe_90/fused_moe_90.so; \
    expected=ee5a6a87a13a271c0418e07ea76fd4cba5cbee1af2d7dab53890caacb9254e14; \
    test -f "${artifact}"; \
    echo "${expected}  ${artifact}" | sha256sum --check -; \
    mkdir -p /opt/flashinfer-jit-cache/flashinfer_jit_cache/data/fused_moe_90 \
        /opt/flashinfer-wheels; \
    cp "${artifact}" \
        /opt/flashinfer-jit-cache/flashinfer_jit_cache/data/fused_moe_90/fused_moe_90.so; \
    python -m pip wheel --no-build-isolation --no-deps \
        --wheel-dir /opt/flashinfer-wheels /opt/flashinfer-jit-cache; \
    test -f \
        /opt/flashinfer-wheels/flashinfer_jit_cache-0.6.14+cu130.omni1-py3-none-any.whl; \
    rm -rf /opt/flashinfer-jit-cache


# install UCX
RUN git clone https://github.com/openucx/ucx.git \
    && cd ucx \
    && git checkout v1.20.x \
    && ./autogen.sh \
    && ./contrib/configure-release-mt  \
    --enable-shared                    \
    --disable-static                   \
    --disable-doxygen-doc              \
    --enable-optimizations             \
    --enable-cma                       \
    --enable-devel-headers             \
    --with-cuda=/usr/local/cuda        \
    --with-verbs                       \
    --with-dm                          \
    --prefix=/usr/local                \
    && make -j \
    && make -j install-strip \
    && ldconfig
