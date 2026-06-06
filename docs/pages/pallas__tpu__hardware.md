- [](../../index.html)
- [Pallas: a JAX kernel language](../index.html)
- [Pallas TPU](index.html)
- TPU Hardware Reference

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .ipynb](../../_sources/pallas/tpu/hardware.ipynb "Download source file")
-  .pdf

# TPU Hardware Reference

# TPU Hardware Reference[\#](#tpu-hardware-reference "Link to this heading")

When writing custom Pallas kernels, it is essential to understand the hardware capabilities and constraints of the target TPU generation.

Note: all specs listed in the table below are **per TensorCore** (a single physical chip may contain multiple TensorCores, e.g., dual-core configurations in v4 or v5p).

Show code cell source

Hide code cell source

    from IPython.display import HTML, display
    from jax.experimental.pallas import tpu as pltpu

    headers = [
        "Version", "Generation", "TensorCores/Chip", "VMEM Capacity", "CMEM Capacity",
        "SMEM Capacity", "HBM Capacity", "HBM BW", "BF16 Peak", "FP8 Peak", "INT8 Peak", "INT4 Peak", "SparseCore"
    ]

    html_lines = []
    html_lines.append("<style>")
    html_lines.append("  .bd-article {")
    html_lines.append("    width: 1300px !important;")
    html_lines.append("  }")
    html_lines.append("  .tpu-spec-table {")
    html_lines.append("    font-family: 'Google Sans', Arial, sans-serif;")
    html_lines.append("    border-collapse: collapse;")
    html_lines.append("    width: 100%;")
    html_lines.append("    margin: 20px 0;")
    html_lines.append("    font-size: 14px;")
    html_lines.append("  }")
    html_lines.append("  .tpu-spec-table th {")
    html_lines.append("    background-color: #3c4043;")
    html_lines.append("    color: white;")
    html_lines.append("    text-align: left;")
    html_lines.append("    padding: 12px 16px;")
    html_lines.append("    font-weight: 500;")
    html_lines.append("    border: 1px solid #dadce0;")
    html_lines.append("  }")
    html_lines.append("  .tpu-spec-table td {")
    html_lines.append("    padding: 12px 16px;")
    html_lines.append("    border: 1px solid #dadce0;")
    html_lines.append("    color: #3c4043;")
    html_lines.append("  }")
    html_lines.append("  .tpu-spec-table tr:nth-child(even) {")
    html_lines.append("    background-color: #f8f9fa;")
    html_lines.append("  }")
    html_lines.append("  .tpu-spec-table tr:hover {")
    html_lines.append("    background-color: #f1f3f4;")
    html_lines.append("  }")
    html_lines.append("</style>")
    html_lines.append("<table class='tpu-spec-table'>")
    html_lines.append("  <thead>")
    html_lines.append("    <tr>")
    for h in headers:
      html_lines.append(f"      <th>{h}</th>")
    html_lines.append("    </tr>")
    html_lines.append("  </thead>")
    html_lines.append("  <tbody>")

    for cv in pltpu.ChipVersion:
      if cv == pltpu.ChipVersion.TPU_7:
        continue  # Skip TPU 7 as it is redundant with 7x

      # Get per-TensorCore specs (num_cores=1)
      info = pltpu.get_tpu_info_for_chip(cv, 1)

      sc = info.sparse_core
      sc_str = "No"
      if sc is not None:
        sc_str = f"Yes ({sc.num_cores} SCs, {sc.num_subcores} subcores, {sc.vmem_capacity_bytes // 1024} KiB VMEM)"

      row = [
          cv.value.upper(),
          f"TPU v{info.generation}" if info.generation < 7 else f"TPU {info.generation}",
          str(cv.num_physical_tensor_cores_per_chip),
          f"{info.vmem_capacity_bytes // (1024 * 1024)} MiB",
          f"{info.cmem_capacity_bytes // 1024} KiB" if info.cmem_capacity_bytes > 0 else "N/A",
          f"{info.smem_capacity_bytes // 1024} KiB",
          f"{info.hbm_capacity_bytes // 1000000000} GB",
          f"{info.mem_bw_bytes_per_second / 1e9:.1f} GB/s" if info.mem_bw_bytes_per_second > 0 else "N/A",
          f"{int(round(info.bf16_ops_per_second / 1e12))} TFLOPs/s" if info.bf16_ops_per_second > 0 else "N/A",
          f"{int(round(info.fp8_ops_per_second / 1e12))} TFLOPs/s" if info.fp8_ops_per_second > 0 else "N/A",
          f"{int(round(info.int8_ops_per_second / 1e12))} TOPs/s" if info.int8_ops_per_second > 0 else "N/A",
          f"{int(round(info.int4_ops_per_second / 1e12))} TOPs/s" if info.int4_ops_per_second > 0 else "N/A",
          sc_str,
      ]

      html_lines.append("    <tr>")
      for cell in row:
        html_lines.append(f"      <td>{cell}</td>")
      html_lines.append("    </tr>")

    html_lines.append("  </tbody>")
    html_lines.append("</table>")

    display(HTML("\n".join(html_lines)))

| Version | Generation | TensorCores/Chip | VMEM Capacity | CMEM Capacity | SMEM Capacity | HBM Capacity | HBM BW | BF16 Peak | FP8 Peak | INT8 Peak | INT4 Peak | SparseCore |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
| V2 | TPU v2 | 2 | 16 MiB | N/A | 16 KiB | 8 GB | 358.0 GB/s | 23 TFLOPs/s | N/A | N/A | N/A | No |
| V3 | TPU v3 | 2 | 16 MiB | N/A | 16 KiB | 17 GB | 412.5 GB/s | 70 TFLOPs/s | N/A | N/A | N/A | No |
| V4I | TPU v4 | 1 | 16 MiB | 130859 KiB | 1024 KiB | 8 GB | 614.0 GB/s | 137 TFLOPs/s | N/A | N/A | N/A | No |
| V4 | TPU v4 | 2 | 16 MiB | 65429 KiB | 1024 KiB | 17 GB | 615.0 GB/s | 138 TFLOPs/s | N/A | N/A | N/A | No |
| V5E | TPU v5 | 1 | 128 MiB | N/A | 1024 KiB | 17 GB | 820.0 GB/s | 197 TFLOPs/s | N/A | 394 TOPs/s | 788 TOPs/s | No |
| V5P | TPU v5 | 2 | 64 MiB | N/A | 1024 KiB | 51 GB | 1230.0 GB/s | 230 TFLOPs/s | N/A | 459 TOPs/s | 920 TOPs/s | Yes (4 SCs, 16 subcores, 512 KiB VMEM) |
| V6E | TPU v6 | 1 | 128 MiB | N/A | 1024 KiB | 34 GB | 1640.0 GB/s | 920 TFLOPs/s | 920 TFLOPs/s | 1840 TOPs/s | 3680 TOPs/s | Yes (2 SCs, 16 subcores, 256 KiB VMEM) |
| 7X | TPU 7 | 2 | 64 MiB | N/A | 1024 KiB | 103 GB | 3700.0 GB/s | 1155 TFLOPs/s | 2300 TFLOPs/s | N/A | N/A | Yes (2 SCs, 16 subcores, 512 KiB VMEM) |
| 8I | TPU 8 | 2 | 192 MiB | N/A | 1024 KiB | 154 GB | 4300.0 GB/s | 550 TFLOPs/s | 4404 TFLOPs/s | N/A | N/A | No |

[](prng.html "previous page")

previous

Pseudo-Random Number Generation

[](../gpu/index.html "next page")

next

Pallas:Mosaic GPU

By The JAX authors

© Copyright 2024, The JAX Authors.\
