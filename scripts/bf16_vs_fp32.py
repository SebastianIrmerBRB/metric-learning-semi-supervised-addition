#!/usr/bin/env python3

import argparse
import platform
import time

import torch


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as file:
            for line in file:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass

    return platform.processor() or platform.machine()


def cuda_has_native_bf16() -> bool:
    if not torch.cuda.is_available():
        return False

    try:
        # Newer PyTorch versions can distinguish native support from emulation.
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        # Compatibility with older PyTorch releases.
        return torch.cuda.is_bf16_supported()


def benchmark_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    warmup: int,
    iterations: int,
) -> tuple[float, torch.Tensor]:
    output = torch.empty(
        (a.shape[0], b.shape[1]),
        device=a.device,
        dtype=a.dtype,
    )

    for _ in range(warmup):
        torch.mm(a, b, out=output)

    if a.device.type == "cuda":
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iterations):
            torch.mm(a, b, out=output)
        end.record()

        end.synchronize()
        milliseconds = start.elapsed_time(end) / iterations
    else:
        start_time = time.perf_counter()

        for _ in range(iterations):
            torch.mm(a, b, out=output)

        milliseconds = (
            (time.perf_counter() - start_time) * 1000 / iterations
        )

    return milliseconds, output


def benchmark_device(
    device: torch.device,
    size: int,
    warmup: int,
    iterations: int,
    run_bf16: bool = True,
) -> None:
    print(f"\n{'=' * 68}")
    print(f"Device: {device}")
    print(f"Matrix: {size} x {size}")
    print(f"Iterations: {iterations}")

    generator = torch.Generator(device=device)
    generator.manual_seed(1234)

    a_fp32 = torch.randn(
        size,
        size,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    b_fp32 = torch.randn(
        size,
        size,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )

    fp32_ms, fp32_result = benchmark_mm(
        a_fp32,
        b_fp32,
        warmup,
        iterations,
    )

    operations = 2 * size**3
    fp32_tflops = operations / (fp32_ms / 1000) / 1e12

    print(f"\nFP32: {fp32_ms:9.3f} ms   {fp32_tflops:8.2f} TFLOP/s")

    if not run_bf16:
        print("BF16: skipped because native BF16 support was not detected")
        return

    try:
        a_bf16 = a_fp32.to(torch.bfloat16)
        b_bf16 = b_fp32.to(torch.bfloat16)

        bf16_ms, bf16_result = benchmark_mm(
            a_bf16,
            b_bf16,
            warmup,
            iterations,
        )

        bf16_tflops = operations / (bf16_ms / 1000) / 1e12
        speedup = fp32_ms / bf16_ms

        difference = bf16_result.float() - fp32_result
        relative_l2 = (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(fp32_result)
        ).item()
        mean_absolute_error = difference.abs().mean().item()
        maximum_absolute_error = difference.abs().max().item()

        print(f"BF16: {bf16_ms:9.3f} ms   {bf16_tflops:8.2f} TFLOP/s")
        print(f"\nBF16 speedup:        {speedup:9.2f}x")
        print(f"Relative L2 error:   {relative_l2:9.4e}")
        print(f"Mean absolute error: {mean_absolute_error:9.4e}")
        print(f"Max absolute error:  {maximum_absolute_error:9.4e}")

    except (RuntimeError, NotImplementedError) as error:
        print(f"BF16 benchmark failed: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch FP32 versus BF16 matrix multiplication."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=4096,
        help="Square matrix dimension; default: 4096",
    )
    parser.add_argument(
        "--cpu-iters",
        type=int,
        default=5,
        help="CPU benchmark iterations; default: 5",
    )
    parser.add_argument(
        "--gpu-iters",
        type=int,
        default=50,
        help="GPU benchmark iterations; default: 50",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warm-up iterations; default: 3",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="CPU thread count; 0 keeps the PyTorch default",
    )
    args = parser.parse_args()

    if args.size <= 0:
        parser.error("--size must be greater than zero")

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    # Ensure FP32 matrix multiplication uses the highest internal precision.
    torch.set_float32_matmul_precision("highest")

    print(f"PyTorch:        {torch.__version__}")
    print(f"CPU:            {cpu_name()}")
    print(f"CPU capability: {torch.backends.cpu.get_cpu_capability()}")
    print(f"CPU threads:    {torch.get_num_threads()}")

    with torch.inference_mode():
        # Attempting the operation is the most useful CPU BF16 test because
        # support and performance depend on the CPU and PyTorch backend.
        benchmark_device(
            device=torch.device("cpu"),
            size=args.size,
            warmup=args.warmup,
            iterations=args.cpu_iters,
            run_bf16=True,
        )

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            bf16_supported = cuda_has_native_bf16()

            print(f"\nCUDA runtime:   {torch.version.cuda}")
            print(f"GPU:            {properties.name}")
            print(
                "Compute capability: "
                f"{properties.major}.{properties.minor}"
            )
            print(f"Native BF16:    {bf16_supported}")

            benchmark_device(
                device=torch.device("cuda:0"),
                size=args.size,
                warmup=args.warmup,
                iterations=args.gpu_iters,
                run_bf16=bf16_supported,
            )
        else:
            print("\nCUDA GPU not detected by PyTorch.")


if __name__ == "__main__":
    main()