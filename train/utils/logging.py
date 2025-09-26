from rich.console import Console
from rich.table import Table

console = Console()


def banner(text: str) -> None:
    console.rule(f"[bold cyan]{text}")


def log_kv(**kwargs) -> None:
    table = Table(show_header=False, box=None, pad_edge=False)
    for key, value in kwargs.items():
        table.add_row(f"[bold]{key}", str(value))
    console.print(table)
