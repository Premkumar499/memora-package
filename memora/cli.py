"""Command line interface for Memora using Typer and Rich."""

from __future__ import annotations

import json
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from memora import __version__
from memora.hardware.gpu_detector import GPUDetector
from memora.utils.common import format_bytes

app = typer.Typer(help="Memora: PyTorch Memory-Aware Optimization CLI")
console = Console()


@app.command()
def hardware() -> None:
    """Detect and display system hardware, CPU RAM, and GPU memory."""
    report = GPUDetector.get_hardware_report()

    console.print(Panel.fit(
        f"[bold cyan]Memora Hardware Discovery[/bold cyan]\n"
        f"OS: {report.platform_system} ({report.platform_release})\n"
        f"Python: {report.python_version} | PyTorch: {report.torch_version}\n"
        f"CUDA Available: [{'green' if report.cuda_available else 'yellow'}]{report.cuda_available}[/]"
        + (f" (v{report.cuda_version})" if report.cuda_version else "")
        + f"\nSystem RAM: {report.to_dict()['system_ram_available_str']} free / {report.to_dict()['system_ram_total_str']} total",
        title="System Environment",
    ))

    table = Table(title="Detected Compute Devices", header_style="bold magenta")
    table.add_column("Index", style="cyan")
    table.add_column("Type", style="bold")
    table.add_column("Device Name", style="white")
    table.add_column("Total Memory", style="green")
    table.add_column("Compute Cap.", style="yellow")
    table.add_column("FP16 / BF16", style="blue")

    for dev in report.devices:
        half_support = f"FP16: {dev.supports_fp16} | BF16: {dev.supports_bf16}"
        table.add_row(
            str(dev.device_index),
            dev.device_type.upper(),
            dev.name,
            format_bytes(dev.total_memory),
            dev.compute_capability or "N/A",
            half_support,
        )

    console.print(table)


@app.command()
def report(file_path: str = typer.Argument(..., help="Path to Memora report JSON file")) -> None:
    """Read and format a Memora training run report."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        console.print(Panel(
            f"[bold]Run ID:[/bold] {data.get('run_id')}\n"
            f"[bold]Timestamp:[/bold] {data.get('timestamp')}\n"
            f"[bold]Device:[/bold] {data.get('hardware_info', {}).get('default_device', {}).get('name')}",
            title="[bold green]Report Summary[/bold green]",
        ))

        opts = data.get("optimization_decisions", {})
        opt_table = Table(title="Optimization Decisions", header_style="bold cyan")
        opt_table.add_column("Parameter", style="bold")
        opt_table.add_column("Value", style="green")

        for k, v in opts.items():
            if k != "reasons":
                opt_table.add_row(k, str(v))
        console.print(opt_table)

        prof = data.get("profiler_summary", {})
        if prof:
            prof_table = Table(title="Performance Metrics", header_style="bold yellow")
            prof_table.add_column("Metric", style="bold")
            prof_table.add_column("Value", style="cyan")
            for k, v in prof.items():
                prof_table.add_row(k, str(v))
            console.print(prof_table)

    except Exception as e:
        console.print(f"[bold red]Error loading report:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Show Memora version."""
    import torch
    console.print(f"Memora version: [bold green]{__version__}[/bold green]")
    console.print(f"PyTorch version: [bold cyan]{torch.__version__}[/bold cyan]")


if __name__ == "__main__":
    app()
