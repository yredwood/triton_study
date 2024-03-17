
import sys

import torch

import triton
import triton.language as tl
import transformer_engine.pytorch as te
from transformer_engine.pytorch.attention import apply_rotary_pos_emb

from layer import RotaryPositionEmbedding

head_dim = 1
seq_len = 2048
batch_size = 1


@triton.jit
def rope_bw_kernel(
        freq_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr
    ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    freq = tl.load(freq_ptr + offsets, mask=mask)
    
    cos = tl.cos(freq)
    sin = tl.sin(freq)

    sin_sign = tl.where(tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE//2, 1., -1.)
    x_grad = cos + sin_sign * sin
    tl.store(output_ptr + offsets, x_grad, mask=mask)


@triton.jit
def rope_fw_kernel(
        x_ptr,
        freq_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr
    ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    freq = tl.load(freq_ptr + offsets, mask=mask)

    cos = tl.cos(freq)
    sin = tl.sin(freq)

    # rotate x by index
    offset_from_middle = block_start + (BLOCK_SIZE//2 + tl.arange(0, BLOCK_SIZE)) % BLOCK_SIZE
    mask = offset_from_middle < n_elements
    x_rot = tl.load(x_ptr + offset_from_middle, mask=mask) # maybe cache hit?
    x_rot = tl.where(tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE//2, -x_rot, x_rot)
    output = x * cos + x_rot * sin
    tl.store(output_ptr + offsets, output, mask=mask)


def _fw_rope(x, freq):
    output = torch.empty_like(x)
    assert x.is_cuda and freq.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    rope_fw_kernel[grid](x, freq, output, n_elements, BLOCK_SIZE=1024)
    return output

def _bw_rope(freq):
    output = torch.empty_like(freq)
    assert freq.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    rope_bw_kernel[grid](freq, output, n_elements, BLOCK_SIZE=1024)
    return output

class TritonRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, freq):
        y = _fw_rope(x, freq)
        ctx.save_for_backward(freq)
        return y

    @staticmethod
    def backward(ctx, grad):
       freq, = ctx.saved_tensors
       x_grad = _bw_rope(freq)
       return grad * x_grad, None


def get_fw_bw(fn, x, freq):
    x = torch.nn.Parameter(x.detach(), requires_grad=True)
    out = fn(x, freq)
    loss = out.sum()
    loss.backward()
    return out.squeeze(), x.grad.squeeze()


def verify():
    hidden_dim = 1024
    rope = RotaryPositionEmbedding(hidden_dim).to("cuda")
    x = torch.rand(seq_len, batch_size, head_dim, hidden_dim, device="cuda")
    freq = rope(seq_len)

    torch_fn = lambda x, freq: apply_rotary_pos_emb(x, freq)
    torch_out, torch_grad = get_fw_bw(torch_fn, x, freq)

    te_fn = lambda x, freq: apply_rotary_pos_emb(x, freq, fused=True)
    te_out, te_grad = get_fw_bw(te_fn, x, freq)

    triton_fn = lambda x, freq: TritonRope.apply(x, freq)
    triton_out, triton_grad = get_fw_bw(triton_fn, x, freq)

    # print(f"{torch_out=}")
    # print(f"{te_out=}")
    # print(f"{triton_out=}")
    assert torch.allclose(torch_out, te_out, atol=1e-6)
    assert torch.allclose(torch_out, triton_out, atol=1e-6)
    assert torch.allclose(torch_grad, triton_grad, atol=1e-6)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['hidden_dim'],
        x_vals=[2**i for i in range(10, 18, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['torch', 'te', 'triton'],
        line_names=['torch', 'transformer_engine', 'triton'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='time consumed (ms)',
        plot_name='rope-fw-bw-performance',
        args={},
    ))
def benchmark(hidden_dim, provider):
    x = torch.rand(seq_len, batch_size, head_dim, hidden_dim, device='cuda')
    rope = RotaryPositionEmbedding(hidden_dim).to("cuda")
    freq = rope(seq_len)
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        bench_fn = lambda x, freq: apply_rotary_pos_emb(x, freq)
    if provider == 'te':
        bench_fn = lambda x, freq: apply_rotary_pos_emb(x, freq, fused=True)
    if provider == 'triton':
        bench_fn = lambda x, freq: TritonRope.apply(x, freq)

    def fn():
        return get_fw_bw(bench_fn, x, freq)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    return ms, max_ms, min_ms



if __name__ == "__main__":
    torch.manual_seed(0)
    verify()
    benchmark.run(print_data=True, save_path='./')
