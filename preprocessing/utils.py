from rich.console import Console
from rich.progress import (
    MofNCompleteColumn,
    TimeElapsedColumn,
    Progress,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    Column,
)

console = Console()


def create_progress_bar(console) -> Progress:
    """
    Create a rich progress bar with a spinner and time remaining.
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(table_column=Column(justify="right", width=10)),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        transient=True,
        console=console,
        speed_estimate_period=30,
    )
    return progress


progress = create_progress_bar(console)
