import shlex
from typing import List, Optional

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from vertex_job_runner.job import run_custom_training_job
from vertex_job_runner.settings import Settings

app = typer.Typer(
    help="Vertex AI Job Runner CLI",
    pretty_exceptions_enable=True,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=True,
)
console = Console()


def get_help(field_name: str) -> str:
    """Helper to get help text from Settings model."""
    field = Settings.model_fields.get(field_name)
    return str(field.description) if field and field.description else ""


@app.command()
def run(
    machine_type: Optional[str] = typer.Option(None, help=get_help("machine_type")),
    accelerator_type: Optional[str] = typer.Option(None, help=get_help("accelerator_type")),
    accelerator_count: Optional[int] = typer.Option(None, help=get_help("accelerator_count")),
    args: Optional[str] = typer.Option(
        None,
        help=get_help("args") + " (space-separated strings, e.g., --param1=val1 param2)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show config without execution"),
) -> None:
    """Run a Vertex AI custom training job."""

    cli_args = {
        "machine_type": machine_type,
        "accelerator_type": accelerator_type,
        "accelerator_count": accelerator_count,
        "args": shlex.split(args) if args is not None else None,
    }
    cli_overrides: dict[str, str | int | List[str]] = {
        k: v for k, v in cli_args.items() if v is not None
    }

    # Initialize settings with CLI overrides
    try:
        settings = Settings(**cli_overrides)  # type: ignore
    except ValidationError as e:
        console.print(f"[bold red]Configuration Error:[/bold red]\n{e}")
        raise typer.Exit(code=1)

    table = Table(title="🚀 Custom-Training Job Configuration", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="magenta")
    for key, value in settings.model_dump().items():
        table.add_row(key, str(value))
    console.print(table)

    if dry_run:
        console.print("[yellow]Dry run mode. Exiting without execution.[/yellow]")
        return

    with console.status("[bold green]Submitting job to Vertex AI...[/bold green]"):
        job_name = run_custom_training_job(settings)
        console.print(f"Job submitted! Job: [bold]{job_name}[/bold]")


if __name__ == "__main__":
    app()
