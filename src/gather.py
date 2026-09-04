# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import restride_dim

logger = logging.getLogger(__name__)
UB_SIZE_BYTES = 192 * 1024
# wt-2026-09-04-fix: add MAX_RANK for static-signature stride-aware kernel (issue #5746)
MAX_RANK = 5


def compute_base_offset(shape, strides, dim):
    idx = torch.arange(int(torch.prod(torch.tensor(shape))), device="cpu")
    coord = torch.empty((len(shape), idx.numel()), dtype=torch.long, device="cpu")
    for i in reversed(range(len(shape))):
        coord[i] = idx % shape[i]
        idx = idx // shape[i]

    offset = torch.zeros_like(coord[0])
    for i in range(len(shape)):
        if i != dim:
            offset += coord[i] * strides[i]
    return offset


@libentry()
@triton.heuristics({"BLOCK_SIZE": lambda args: 1024})
@triton.jit
def _gather_flat_kernel_fixed(
    inp,
    index,
    out,
    base_offset,
    inp_dim_stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    cur_index = tl.load(index + offset, mask=mask, other=0)
    base = tl.load(base_offset + offset, mask=mask, other=0)

    inp_offset = base + cur_index * inp_dim_stride

    val = tl.load(inp + inp_offset, mask=mask, other=0)
    tl.store(out + offset, val, mask=mask)


def gather_flat_fixed(inp: torch.Tensor, dim: int, index: torch.Tensor, out=None):
    logger.debug("GEMS_ASCEND GATHER")

    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)

    N = index.numel()
    dim_stride = inp.stride(dim)
    inp_strided = restride_dim(inp, dim, index.shape)
    if dim == -1:
        dim = inp_strided.dim() - 1
    base_offset = compute_base_offset(index.shape, inp_strided.stride(), dim).to(
        torch.int64
    )
    base_offset = base_offset.npu()
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    _gather_flat_kernel_fixed[grid](
        inp_strided,
        index,
        out,
        base_offset,
        dim_stride,
        N,
    )
    return out


# wt-2026-09-04-fix: stride-aware gather kernel for non-contiguous index (issue #5746)
# Computes index/out/inp offsets per dimension as sum(coord_i * stride_i) instead of
# linearizing index reads. Missing dims padded by host with shape=1 / stride=0.
@libentry()
@triton.heuristics({"BLOCK_SIZE": lambda args: 1024})
@triton.jit
def _gather_strided_kernel(
    inp,
    index,
    out,
    idx_shape0,
    idx_shape1,
    idx_shape2,
    idx_shape3,
    idx_shape4,
    idx_stride0,
    idx_stride1,
    idx_stride2,
    idx_stride3,
    idx_stride4,
    out_stride0,
    out_stride1,
    out_stride2,
    out_stride3,
    out_stride4,
    inp_stride0,
    inp_stride1,
    inp_stride2,
    inp_stride3,
    inp_stride4,
    dim,
    dim_stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offset < N

    # Decompose the flat offset into per-dim coordinates over the index
    # shape. Missing dims are padded by the host with shape=1 / stride=0,
    # so their coordinate is always 0 and contributes nothing.
    cur = offset
    coord0 = cur % idx_shape0
    cur = cur // idx_shape0
    coord1 = cur % idx_shape1
    cur = cur // idx_shape1
    coord2 = cur % idx_shape2
    cur = cur // idx_shape2
    coord3 = cur % idx_shape3
    cur = cur // idx_shape3
    coord4 = cur % idx_shape4

    index_offset = (
        coord0 * idx_stride0
        + coord1 * idx_stride1
        + coord2 * idx_stride2
        + coord3 * idx_stride3
        + coord4 * idx_stride4
    )
    out_offset = (
        coord0 * out_stride0
        + coord1 * out_stride1
        + coord2 * out_stride2
        + coord3 * out_stride3
        + coord4 * out_stride4
    )
    # inp strides come from restride_dim(inp, dim, index.shape): the stride
    # along `dim` is zeroed, so all dims contribute uniformly here and the
    # gather index itself is applied separately via dim_stride below.
    base = (
        coord0 * inp_stride0
        + coord1 * inp_stride1
        + coord2 * inp_stride2
        + coord3 * inp_stride3
        + coord4 * inp_stride4
    )

    cur_index = tl.load(index + index_offset, mask=mask, other=0)
    inp_offset = base + cur_index * dim_stride
    val = tl.load(inp + inp_offset, mask=mask, other=0)
    tl.store(out + out_offset, val, mask=mask)


# wt-2026-09-04-fix: host wrapper for _gather_strided_kernel (issue #5746)
def gather_strided(inp: torch.Tensor, dim: int, index: torch.Tensor, out=None):
    logger.debug("GEMS_ASCEND GATHER (strided index)")
    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)

    N = index.numel()
    # dim_stride must be taken from the ORIGINAL inp: restride_dim zeroes
    # the stride along dim below.
    dim_stride = inp.stride(dim)
    inp_strided = restride_dim(inp, dim, index.shape)

    def pad_shape(xs):
        return list(xs) + [1] * (MAX_RANK - len(xs))

    def pad_stride(xs):
        return list(xs) + [0] * (MAX_RANK - len(xs))

    idx_shapes = pad_shape(index.shape)
    idx_strides = pad_stride(index.stride())
    out_strides = pad_stride(out.stride())
    inp_strides = pad_stride(inp_strided.stride())

    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    _gather_strided_kernel[grid](
        inp_strided,
        index,
        out,
        idx_shapes[0],
        idx_shapes[1],
        idx_shapes[2],
        idx_shapes[3],
        idx_shapes[4],
        idx_strides[0],
        idx_strides[1],
        idx_strides[2],
        idx_strides[3],
        idx_strides[4],
        out_strides[0],
        out_strides[1],
        out_strides[2],
        out_strides[3],
        out_strides[4],
        inp_strides[0],
        inp_strides[1],
        inp_strides[2],
        inp_strides[3],
        inp_strides[4],
        dim,
        dim_stride,
        N,
    )
    return out


# wt-2026-09-04-fix: dispatch non-contiguous index to stride-aware kernel (issue #5746)
# contiguous -> original flat fast path (unchanged); non-contiguous rank<=MAX_RANK ->
# gather_strided; rank>MAX_RANK -> contiguous() fallback then flat path.
def gather(inp, dim, index, out=None, sparse_grad=False):
    logger.debug("GEMS_ASCEND GATHER")
    if inp.ndim != index.ndim:
        raise IndexError(
            f"self and index must have the same number of dimensions, "
            f"got self.ndim = {inp.ndim} and index.ndim = {index.ndim}"
        )
    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)

    dim = dim % inp.dim()
    if index.is_contiguous() or index.ndim > MAX_RANK:
        # flat fast path; for over-sized ranks force contiguity first
        # (rank > MAX_RANK cannot be expressed in the static kernel signature)
        if not index.is_contiguous():
            index = index.contiguous()
        return gather_flat_fixed(inp, dim, index, out)
    return gather_strided(inp, dim, index, out)


def gather_backward(grad, self, dim, index, sparse_grad):
    logger.debug("GEMS_ASCEND GATHER_BACKWARD")
    from .scatter import scatter_

    result = grad.new_zeros(self.shape)
    return scatter_(result, dim, index, grad, reduce="add")
