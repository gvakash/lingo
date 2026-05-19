#!/usr/bin/env python3
"""
cli.py — Command-line interface for batch processing
Usage:
    python cli.py process audio.wav
    python cli.py batch ./audio_dir/ --language hi
    python cli.py evaluate audio.wav --reference "expected transcription text"
    python cli.py serve
"""
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.syntax import Syntax
from loguru import logger
import json

app = typer.Typer(
    name="lingo-pipeline",
    help="🎙️ Lingo Speech Transcription & Evaluation Pipeline",
    add_completion=False,
)
console = Console()


@app.command()
def process(
    audio_path: Path = typer.Argument(..., help="Path to audio file"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language code (hi/en/ta/...)"),
    no_diarize: bool = typer.Option(False, "--no-diarize", help="Skip diarization"),
    no_emotion: bool = typer.Option(False, "--no-emotion", help="Skip emotion tagging"),
    no_denoise: bool = typer.Option(False, "--no-denoise", help="Skip noise reduction"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o"),
    reference: Optional[str] = typer.Option(None, "--reference", "-r", help="Reference text for WER"),
    speakers: Optional[int] = typer.Option(None, "--speakers", "-s", help="Number of speakers"),
):
    """Process a single audio file through the full pipeline."""
    from app.core.pipeline import SpeechPipeline, PipelineConfig

    if not audio_path.exists():
        console.print(f"[red]✗ File not found: {audio_path}[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold cyan]🎙️ Lingo Speech Pipeline[/bold cyan]\n"
        f"File: [yellow]{audio_path.name}[/yellow]\n"
        f"Lang: {language or 'auto-detect'} | "
        f"Diarize: {not no_diarize} | Emotion: {not no_emotion}",
        title="Processing"
    ))

    config = PipelineConfig(
        language=language,
        run_diarization=not no_diarize,
        run_emotion=not no_emotion,
        denoise=not no_denoise,
        num_speakers=speakers,
        reference_text=reference,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running pipeline...", total=None)
        pl = SpeechPipeline()
        result = pl.run(audio_path, config, output_dir)
        progress.update(task, completed=True)

    # Display results
    _display_result(result)


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Directory of audio files"),
    language: Optional[str] = typer.Option(None, "--language", "-l"),
    pattern: str = typer.Option("*.wav,*.mp3,*.m4a,*.flac", "--pattern"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o"),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel workers"),
):
    """Batch process a directory of audio files."""
    from app.core.pipeline import SpeechPipeline, PipelineConfig

    patterns = pattern.split(",")
    files = []
    for p in patterns:
        files.extend(input_dir.glob(p.strip()))

    if not files:
        console.print(f"[red]No audio files found in {input_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Found {len(files)} files to process[/cyan]")

    pl = SpeechPipeline()
    config = PipelineConfig(language=language)
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=len(files))
        for f in files:
            result = pl.run(f, config, output_dir)
            results.append(result)
            progress.advance(task)

    # Summary table
    table = Table(title="Batch Processing Summary", show_header=True)
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Language")
    table.add_column("Duration")
    table.add_column("Speakers")
    table.add_column("Time")

    for r in results:
        tx = r.transcription or {}
        d = r.diarization or {}
        meta = r.audio_metadata or {}
        status_color = "green" if r.status == "success" else "yellow" if r.status == "partial" else "red"
        table.add_row(
            Path(r.input_file).name,
            f"[{status_color}]{r.status}[/{status_color}]",
            tx.get("language", "?"),
            f"{meta.get('duration_secs', 0):.1f}s",
            str(d.get("num_speakers", "?")),
            f"{r.total_time_secs:.1f}s",
        )

    console.print(table)


@app.command()
def evaluate(
    audio_path: Path = typer.Argument(...),
    reference: str = typer.Argument(..., help="Reference transcription text"),
    model: str = typer.Option("base", "--model", "-m"),
    language: Optional[str] = typer.Option(None, "--language", "-l"),
):
    """Evaluate ASR quality (WER/CER) against a reference transcription."""
    from app.core.pipeline import SpeechPipeline, PipelineConfig

    config = PipelineConfig(
        language=language,
        run_diarization=False,
        run_emotion=False,
        reference_text=reference,
    )

    pl = SpeechPipeline()
    result = pl.run(audio_path, config)
    tx = result.transcription or {}

    console.print(Panel(
        f"[bold]Reference:[/bold]  {reference}\n"
        f"[bold]Hypothesis:[/bold] {tx.get('text', 'N/A')}\n\n"
        f"[bold cyan]WER:[/bold cyan]  {tx.get('wer', 'N/A')}\n"
        f"[bold cyan]CER:[/bold cyan]  {tx.get('cer', 'N/A')}\n"
        f"[bold]Language:[/bold]   {tx.get('language', '?')}\n"
        f"[bold]RTF:[/bold]        {tx.get('rtf', '?')}x",
        title="📊 Evaluation Results",
        border_style="cyan"
    ))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
):
    """Start the FastAPI server."""
    import uvicorn
    console.print(Panel(
        f"[bold cyan]🚀 Starting Lingo Speech Pipeline API[/bold cyan]\n"
        f"Host: {host}:{port}\n"
        f"Docs: http://localhost:{port}/docs",
        title="API Server"
    ))
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


def _display_result(result):
    """Pretty-print pipeline result."""
    tx = result.transcription or {}
    d = result.diarization or {}
    e = result.emotion or {}
    meta = result.audio_metadata or {}

    status_color = "green" if result.status == "success" else "yellow"

    console.print(Panel(
        f"[bold]Status:[/bold]  [{status_color}]{result.status.upper()}[/{status_color}]\n"
        f"[bold]Job ID:[/bold]  {result.job_id}\n"
        f"[bold]Total Time:[/bold] {result.total_time_secs}s",
        title="✅ Pipeline Complete"
    ))

    if tx:
        console.print("\n[bold cyan]📝 Transcription[/bold cyan]")
        console.print(f"  Language:  {tx.get('language', '?')}")
        console.print(f"  RTF:       {tx.get('rtf', '?')}x")
        if tx.get('wer') is not None:
            console.print(f"  WER:       [bold]{tx['wer']:.4f}[/bold]")
        console.print(f"\n[italic]{tx.get('text', '')}[/italic]")

    if d:
        console.print("\n[bold cyan]🎤 Diarization[/bold cyan]")
        console.print(f"  Speakers:  {d.get('num_speakers', '?')}")
        for speaker, dur in (d.get('speaker_durations') or {}).items():
            console.print(f"  {speaker}:  {dur:.1f}s")

    if e:
        console.print("\n[bold cyan]😊 Emotion[/bold cyan]")
        console.print(f"  Dominant:  {e.get('dominant_emotion', '?')}")
        console.print(f"  Confidence: {e.get('mean_confidence', 0):.2f}")

    if result.output_files:
        console.print("\n[bold]📁 Output Files[/bold]")
        for name, path in result.output_files.items():
            console.print(f"  {name}: [dim]{path}[/dim]")


if __name__ == "__main__":
    app()
