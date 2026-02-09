from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from nemo_subtitle_generator.transcriber import ModelType, Transcriber

app = typer.Typer(
    name="gen-subtitle",
    help="Subtitle generation with NVIDIA NeMo",
    add_completion=False,
)
console = Console()
MODEL_ALIASES = {
    "parakeet": ModelType.PARAKEET,
    "canary": ModelType.CANARY,
    ModelType.PARAKEET.value.lower(): ModelType.PARAKEET,
    ModelType.CANARY.value.lower(): ModelType.CANARY,
}


def parse_model(model_value: str) -> ModelType:
    model_key = model_value.strip().lower()
    model = MODEL_ALIASES.get(model_key)
    if model is None:
        supported = ", ".join(sorted(MODEL_ALIASES.keys()))
        raise ValueError(f"Unsupported model '{model_value}'. Supported values: {supported}")
    return model


@app.command()
def transcribe(
    file: Annotated[Path, typer.Argument(help="Audio or video file to transcribe")],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model to use (parakeet|canary or full model ID)",
        ),
    ] = "parakeet",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output SRT file path"),
    ] = None,
    source_lang: Annotated[
        str,
        typer.Option("--source-lang", "-s", help="Source language (for Canary)"),
    ] = "auto",
    target_lang: Annotated[
        Optional[str],
        typer.Option("--target-lang", "-t", help="Target language for translation (Canary only)"),
    ] = None,
    text_only: Annotated[
        bool,
        typer.Option("--text-only", help="Output plain text instead of SRT"),
    ] = False,
    local_attention: Annotated[
        bool,
        typer.Option("--local-attention", help="Use local attention for long files (Parakeet, reduces VRAM)"),
    ] = False,
) -> None:
    """Transcribe an audio or video file to SRT subtitles."""
    if not file.exists():
        console.print(f"[red]Error: file not found: {file}[/red]")
        raise typer.Exit(1)

    try:
        parsed_model = parse_model(model)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold blue]Gen Subtitle[/bold blue]\n"
        f"File: [green]{file.name}[/green]\n"
        f"Model: [yellow]{parsed_model.value}[/yellow]"
    ))

    transcriber = Transcriber(model_type=parsed_model, local_attention=local_attention)
    try:
        transcriber.validate_options(source_lang=source_lang, target_lang=target_lang)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Loading model...", total=None)
        transcriber.load_model()

    if text_only:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Transcribing...", total=None)
            text = transcriber.transcribe(
                file,
                source_lang=source_lang,
                target_lang=target_lang,
            )

        console.print("\n[bold]Transcription:[/bold]")
        console.print(Panel(text))
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Generating subtitles...", total=None)
            srt_path = transcriber.transcribe_to_srt(
                file,
                output_path=output,
                source_lang=source_lang,
                target_lang=target_lang,
            )

        console.print(f"\n[green]OK[/green] Subtitles saved: [bold]{srt_path}[/bold]")


@app.command()
def batch(
    directory: Annotated[Path, typer.Argument(help="Directory to scan")],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model to use (parakeet|canary or full model ID)",
        ),
    ] = "parakeet",
    max_depth: Annotated[
        int,
        typer.Option("--max-depth", "-d", help="Maximum recursion depth"),
    ] = 3,
    source_lang: Annotated[
        str,
        typer.Option("--source-lang", "-s", help="Source language (for Canary)"),
    ] = "auto",
    target_lang: Annotated[
        Optional[str],
        typer.Option("--target-lang", "-t", help="Target language for translation (Canary only)"),
    ] = None,
    local_attention: Annotated[
        bool,
        typer.Option("--local-attention", help="Use local attention for long files (Parakeet, reduces VRAM)"),
    ] = False,
) -> None:
    """Transcribe all audio/video files in a directory recursively."""
    if not directory.exists():
        console.print(f"[red]Error: directory not found: {directory}[/red]")
        raise typer.Exit(1)

    if not directory.is_dir():
        console.print(f"[red]Error: {directory} is not a directory[/red]")
        raise typer.Exit(1)

    try:
        parsed_model = parse_model(model)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    transcriber = Transcriber(model_type=parsed_model, local_attention=local_attention)
    try:
        transcriber.validate_options(source_lang=source_lang, target_lang=target_lang)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    def find_media_files(path: Path, current_depth: int) -> list[Path]:
        files = []
        if current_depth > max_depth:
            return files
        try:
            for item in path.iterdir():
                if item.is_file() and transcriber.is_supported(item):
                    files.append(item)
                elif item.is_dir():
                    files.extend(find_media_files(item, current_depth + 1))
        except PermissionError:
            console.print(f"[yellow]Warning: permission denied for {path}[/yellow]")
        return files

    console.print(Panel.fit(
        f"[bold blue]Gen Subtitle - Batch[/bold blue]\n"
        f"Directory: [green]{directory}[/green]\n"
        f"Max depth: [yellow]{max_depth}[/yellow]\n"
        f"Model: [yellow]{parsed_model.value}[/yellow]"
    ))

    console.print("\n[bold]Scanning for media files...[/bold]")
    media_files = find_media_files(directory, 0)

    if not media_files:
        console.print("[yellow]No audio/video files found.[/yellow]")
        raise typer.Exit(0)

    console.print(f"[green]{len(media_files)}[/green] file(s) found\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Loading model...", total=None)
        transcriber.load_model()

    success_count = 0
    error_count = 0

    for i, file in enumerate(media_files, 1):
        output_path = file.with_suffix(".srt")
        console.print(f"\n[bold][{i}/{len(media_files)}][/bold] {file.name}")

        if output_path.exists():
            console.print(f"  [yellow]skipped: {output_path.name} already exists[/yellow]")
            continue

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("  Transcribing...", total=None)
                transcriber.transcribe_to_srt(
                    file,
                    output_path=output_path,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
            console.print(f"  [green]OK[/green] {output_path.name}")
            success_count += 1
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            error_count += 1

    console.print(f"\n[bold]Done.[/bold]")
    console.print(f"  [green]{success_count}[/green] succeeded")
    if error_count:
        console.print(f"  [red]{error_count}[/red] failed")


@app.command()
def models() -> None:
    """List available models."""
    console.print("\n[bold]Available models:[/bold]\n")

    console.print("[yellow]1. Parakeet TDT 0.6B v3[/yellow]")
    console.print("   ID: nvidia/parakeet-tdt-0.6b-v3")
    console.print("   CLI key: parakeet")
    console.print("   25 European languages, auto-detect")
    console.print("   Fast, 600M parameters")
    console.print()

    console.print("[yellow]2. Canary 1B v2[/yellow]")
    console.print("   ID: nvidia/canary-1b-v2")
    console.print("   CLI key: canary")
    console.print("   25 European languages + translation")
    console.print("   Requires explicit --source-lang (no auto mode)")
    console.print("   1B parameters")
    console.print()


@app.command()
def gui() -> None:
    """Launch the Gradio web interface."""
    from nemo_subtitle_generator.gui import launch_gui
    launch_gui()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
