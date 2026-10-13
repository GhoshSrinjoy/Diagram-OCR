from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import download

ALL_TOKENS = {"a", "all", "*", "#"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show formatted dataset sizes from datasets.yaml. By default this is interactive: "
            "it lists datasets, lets you select one or more with commas, then resolves sizes."
        )
    )
    parser.add_argument("--manifest", type=Path, default=download.DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset key, repo_id, number, or comma-separated list. Repeatable.",
    )
    parser.add_argument("--all", action="store_true", help="Select every enabled dataset in the manifest.")
    parser.add_argument("--stage", default=None,
                        help="Optional stage filter, for example small_high_priority.")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Use YAML expected sizes only; no network call.")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--include-gated", action="store_true")
    parser.add_argument("--allow-outside-data-dir", action="store_true")
    parser.add_argument("--token", default=None, help="Hugging Face token. Defaults to HF_TOKEN.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file to load before reading HF_TOKEN. Defaults include src/.env.",
    )
    parser.add_argument("--revision", default=None, help="Override revision for every selected dataset.")
    return parser.parse_args()


def active_configs(
    configs: list[download.DatasetConfig], args: argparse.Namespace
) -> list[download.DatasetConfig]:
    result: list[download.DatasetConfig] = []
    for config in configs:
        if args.stage and config.stage != args.stage:
            continue
        if not config.enabled and not args.include_disabled:
            continue
        if config.gated and not args.include_gated:
            continue
        result.append(config)
    return result


def split_selection_tokens(values: list[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        for token in value.split(","):
            cleaned = token.strip()
            if cleaned:
                tokens.append(cleaned)
    return tokens


def expand_range(token: str) -> list[int] | None:
    if "-" not in token:
        return None
    start_text, end_text = [part.strip() for part in token.split("-", 1)]
    if not start_text.isdigit() or not end_text.isdigit():
        return None
    start = int(start_text)
    end = int(end_text)
    if start > end:
        start, end = end, start
    return list(range(start, end + 1))


def select_configs(
    configs: list[download.DatasetConfig],
    selection_tokens: list[str],
) -> list[download.DatasetConfig]:
    if not selection_tokens or any(token.lower() in ALL_TOKENS for token in selection_tokens):
        return configs

    by_key = {config.key.lower(): config for config in configs}
    by_repo = {config.repo_id.lower(): config for config in configs}
    selected: list[download.DatasetConfig] = []
    selected_keys: set[str] = set()

    for token in selection_tokens:
        lowered = token.lower()
        indexes = expand_range(lowered)
        if indexes is not None:
            for index in indexes:
                add_by_index(configs, selected, selected_keys, index)
            continue
        if lowered.isdigit():
            add_by_index(configs, selected, selected_keys, int(lowered))
            continue

        config = by_key.get(lowered) or by_repo.get(lowered)
        if config is None:
            available = ", ".join(config.key for config in configs)
            raise SystemExit(f"Unknown dataset selection {token!r}. Available keys: {available}")
        add_config(selected, selected_keys, config)

    if not selected:
        raise SystemExit("No datasets selected.")
    return selected


def add_by_index(
    configs: list[download.DatasetConfig],
    selected: list[download.DatasetConfig],
    selected_keys: set[str],
    index: int,
) -> None:
    if index < 1 or index > len(configs):
        raise SystemExit(f"Dataset number out of range: {index}")
    add_config(selected, selected_keys, configs[index - 1])


def add_config(
    selected: list[download.DatasetConfig],
    selected_keys: set[str],
    config: download.DatasetConfig,
) -> None:
    if config.key in selected_keys:
        return
    selected_keys.add(config.key)
    selected.append(config)


def show_available_table(deps: dict[str, Any], configs: list[download.DatasetConfig]) -> None:
    console = deps["Console"]()
    table = deps["Table"](title="Datasets In Manifest")
    table.add_column("#", justify="right")
    table.add_column("Key")
    table.add_column("Repo")
    table.add_column("Category")
    table.add_column("Expected", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Stage")

    for index, config in enumerate(configs, start=1):
        table.add_row(
            str(index),
            config.key,
            config.repo_id,
            config.category,
            download.format_size(deps, config.expected_size_bytes),
            download.format_size(deps, config.max_allowed_bytes),
            config.stage,
        )
    console.print(table)


def prompt_for_selection() -> list[str]:
    raw = input("Select datasets by number/key/repo_id, comma-separated, range like 1-3, or A for all: ")
    return split_selection_tokens([raw])


def diff_text(deps: dict[str, Any], expected: int | None, actual: int | None) -> str:
    if expected is None or actual is None:
        return "unknown"
    diff = actual - expected
    if diff == 0:
        return "0 B"
    sign = "+" if diff > 0 else "-"
    return f"{sign}{download.format_size(deps, abs(diff))}"


def status_text(actual: int | None, max_allowed: int | None) -> str:
    if actual is None or max_allowed is None:
        return "n/a"
    if actual > max_allowed:
        return "over limit"
    return "ok"


def show_manifest_sizes(deps: dict[str, Any], configs: list[download.DatasetConfig]) -> None:
    console = deps["Console"]()
    table = deps["Table"](title="Manifest Dataset Sizes")
    table.add_column("Key")
    table.add_column("Repo")
    table.add_column("Category")
    table.add_column("Expected", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Enabled")

    total_expected = 0
    has_unknown = False
    for config in configs:
        if config.expected_size_bytes is None:
            has_unknown = True
        else:
            total_expected += config.expected_size_bytes
        table.add_row(
            config.key,
            config.repo_id,
            config.category,
            download.format_size(deps, config.expected_size_bytes),
            download.format_size(deps, config.max_allowed_bytes),
            "yes" if config.enabled else "no",
        )

    table.caption = (
        "Total expected size: unknown"
        if has_unknown
        else f"Total expected size: {download.format_size(deps, total_expected)}"
    )
    console.print(table)


def show_remote_sizes(deps: dict[str, Any], plans: list[download.DatasetPlan]) -> None:
    console = deps["Console"]()
    table = deps["Table"](title="Resolved Hugging Face Dataset Sizes")
    table.add_column("Key")
    table.add_column("Repo")
    table.add_column("Revision")
    table.add_column("Files", justify="right")
    table.add_column("Expected", justify="right")
    table.add_column("Remote Size", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Status")

    total_actual = 0
    total_expected = 0
    unknown_actual = False
    unknown_expected = False

    for plan in plans:
        actual = plan.total_size
        expected = plan.config.expected_size_bytes
        if actual is None:
            unknown_actual = True
        else:
            total_actual += actual
        if expected is None:
            unknown_expected = True
        else:
            total_expected += expected

        table.add_row(
            plan.config.key,
            plan.config.repo_id,
            plan.resolved_revision[:12],
            str(len(plan.files)),
            download.format_size(deps, expected),
            download.format_size(deps, actual),
            diff_text(deps, expected, actual),
            download.format_size(deps, plan.config.max_allowed_bytes),
            status_text(actual, plan.config.max_allowed_bytes),
        )

    actual_text = "unknown" if unknown_actual else download.format_size(deps, total_actual)
    expected_text = "unknown" if unknown_expected else download.format_size(deps, total_expected)
    table.caption = f"Total expected: {expected_text} | Total remote: {actual_text}"
    console.print(table)


def resolve_output_dir(raw_manifest: dict[str, Any], args: argparse.Namespace) -> Path:
    output_args = SimpleNamespace(
        output_dir=args.output_dir,
        allow_outside_data_dir=args.allow_outside_data_dir,
    )
    return download.resolve_output_dir(raw_manifest, output_args)


def main() -> int:
    args = parse_args()
    needs_remote = not args.manifest_only
    deps = download.load_runtime_deps(require_hf=needs_remote, require_requests=False)
    download.load_dotenv(args, deps)
    raw_manifest = download.read_manifest(args.manifest, deps["yaml"])
    configs = active_configs(download.build_dataset_configs(raw_manifest), args)
    if not configs:
        raise SystemExit("No datasets available after filters.")

    show_available_table(deps, configs)

    selection_tokens = split_selection_tokens(args.dataset)
    if args.all:
        selection_tokens = ["all"]
    if not selection_tokens:
        selection_tokens = prompt_for_selection()

    selected = select_configs(configs, selection_tokens)

    if args.manifest_only:
        show_manifest_sizes(deps, selected)
        return 0

    output_dir = resolve_output_dir(raw_manifest, args)
    token = download.resolve_hf_token(args)
    plans = download.resolve_plans(deps, selected, output_dir, token, args.revision)
    show_remote_sizes(deps, plans)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
