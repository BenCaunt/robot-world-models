from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from robot_world_models.catalog import (
    CatalogError,
    build_catalog,
    catalog_summary,
    repository_root,
    validate_repository,
)
from robot_world_models.devices import device_report
from robot_world_models.runpod import make_draft_plan
from robot_world_models.warmhub import WarmHubCLI, WarmHubError

app = typer.Typer(no_args_is_help=True, help="WarmHub-first robot world-model recipes.")
catalog_app = typer.Typer(no_args_is_help=True, help="Validate and build the capability catalog.")
warmhub_app = typer.Typer(no_args_is_help=True, help="Read from both WarmHub registries.")
runpod_app = typer.Typer(
    no_args_is_help=True,
    help="Plan bounded remote compute; no writes in v0.1.",
)
app.add_typer(catalog_app, name="catalog")
app.add_typer(warmhub_app, name="warmhub")
app.add_typer(runpod_app, name="runpod")


def _emit(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@catalog_app.command("validate")
def validate_catalog_command() -> None:
    """Validate every manifest and cross-reference."""
    try:
        loaded = validate_repository()
    except CatalogError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    counts: dict[str, int] = {}
    for _, manifest in loaded:
        counts[manifest.kind] = counts.get(manifest.kind, 0) + 1
    _emit({"status": "ok", "counts": counts})


@catalog_app.command("build")
def build_catalog_command(
    check: Annotated[
        bool,
        typer.Option(help="Fail when generated files are stale; do not write."),
    ] = False,
) -> None:
    """Generate catalog/catalog.json and JSON Schemas."""
    try:
        paths = build_catalog(check=check)
    except CatalogError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    _emit(
        {
            "status": "current" if check else "written",
            "paths": [str(path.relative_to(repository_root())) for path in paths],
        }
    )


@catalog_app.command("list")
def list_catalog_command() -> None:
    """List validated manifests."""
    _emit(catalog_summary())


@warmhub_app.command("discover")
def discover_command(
    query: Annotated[str, typer.Argument(help="Robot, task, dataset, or modality query.")],
    output: Annotated[Path | None, typer.Option(help="Optional JSON receipt path.")] = None,
) -> None:
    """Search both first-class WarmHub registries."""
    try:
        result = WarmHubCLI.from_environment().discover(query)
    except WarmHubError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _emit(result)


@app.command("device")
def device_command() -> None:
    """Report the local PyTorch device selection."""
    _emit(device_report())


@app.command("train")
def train_command(
    recipe: Annotated[str, typer.Argument(help="Recipe manifest ID.")],
    run_dir: Annotated[Path, typer.Option(help="Artifact and receipt directory.")],
    max_steps: Annotated[
        int | None,
        typer.Option(min=1, help="Optional bounded override for debugging."),
    ] = None,
    smoke_test_steps: Annotated[
        int | None,
        typer.Option(min=1, help="Optional smoke-test override."),
    ] = None,
    max_episodes: Annotated[
        int | None,
        typer.Option(min=3, help="Optional episode-subset override."),
    ] = None,
) -> None:
    """Run a local, receipt-producing training proof from a reviewed recipe."""
    from robot_world_models.training import run_recipe

    try:
        result = run_recipe(
            recipe_id=recipe,
            run_dir=run_dir,
            max_steps=max_steps,
            smoke_test_steps=smoke_test_steps,
            max_episodes=max_episodes,
        )
    except Exception as error:
        typer.echo(f"{type(error).__name__}: {error}", err=True)
        raise typer.Exit(code=1) from error
    _emit(result)


@runpod_app.command("plan")
def runpod_plan_command(
    max_hourly_usd: Annotated[float, typer.Option(min=0.01)],
    max_runtime_minutes: Annotated[int, typer.Option(min=1)],
    preferred_gpu: Annotated[str, typer.Option()] = "NVIDIA GeForce RTX 5090",
) -> None:
    """Render a non-mutating budget ceiling; live pricing and approval remain required."""
    plan = make_draft_plan(
        preferred_gpu=preferred_gpu,
        max_hourly_usd=max_hourly_usd,
        max_runtime_minutes=max_runtime_minutes,
    )
    _emit(plan.model_dump(mode="json"))


if __name__ == "__main__":
    app()
