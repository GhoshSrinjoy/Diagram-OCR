"""Live terminal dashboard for training (rich): progress, metrics, eval, GPU, data mix, log tail.

The trainer updates a `State` object; `Dashboard` renders it ~4x per second. Log records go to
both the run's train.log (via logging) and the on-screen tail.
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

try:
    import pynvml

    pynvml.nvmlInit()
    _GPU = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # pragma: no cover - no NVML
    pynvml, _GPU = None, None

SPARK = "▁▂▃▄▅▆▇█"


@dataclass
class State:
    run_name: str = ""
    model_name: str = ""
    phase: str = "starting"  # starting | training | evaluating | saving | done | stopped
    step: int = 0
    start_step: int = 0  # step this process started from (after a resume)
    total_steps: int = 1
    epoch: int = 1
    epochs: int = 1
    started: float = field(default_factory=time.time)
    step_started: float = field(default_factory=time.time)
    loss: Optional[float] = None
    loss_ema: Optional[float] = None
    loss_hist: collections.deque = field(default_factory=lambda: collections.deque(maxlen=120))
    lr: float = 0.0
    grad_norm: Optional[float] = None
    tokens_per_s: float = 0.0
    samples_per_s: float = 0.0
    samples_seen: int = 0
    eval: dict = field(default_factory=dict)  # latest eval metrics
    eval_step: Optional[int] = None
    best: Optional[float] = None
    mix: dict = field(default_factory=dict)
    note: str = ""
    extra: dict = field(default_factory=dict)  # extra train rows, e.g. RL reward stats


def _spark(values: list[float], width: int = 48) -> str:
    vals = list(values)[-width:]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return "".join(SPARK[min(len(SPARK) - 1, int((v - lo) / span * (len(SPARK) - 1)))] for v in vals)


def _fmt_td(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _fmt(v: Optional[float], spec: str = ".4f") -> str:
    return "-" if v is None else format(v, spec)


class TailHandler(logging.Handler):
    def __init__(self, maxlen: int = 400):
        super().__init__()
        self.lines: collections.deque = collections.deque(maxlen=maxlen)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append((record.levelno, self.format(record)))


class Dashboard:
    def __init__(self, state: State, tail: TailHandler, enabled: bool = True):
        self.state, self.tail, self.enabled = state, tail, enabled
        self.console = Console()
        self.live: Optional[Live] = None

    def __enter__(self) -> Dashboard:
        if self.enabled:
            self.console.set_window_title(f"Tadpole training - {self.state.run_name}")
            self.live = Live(self.render(), console=self.console, refresh_per_second=4, screen=True,
                             get_renderable=self.render)
            self.live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self.live is not None:
            self.live.__exit__(*exc)

    # ------------------------------------------------------------------ panels
    def _header(self) -> Panel:
        s = self.state
        elapsed = time.time() - s.started
        frac = min(1.0, s.step / max(1, s.total_steps))
        done = s.step - s.start_step  # only steps run by this process count towards the rate
        rate = elapsed / done if done > 0 else 0.0
        eta = rate * (s.total_steps - s.step) if done > 0 else 0.0
        top = Text.assemble(
            ("Tadpole  ", "bold cyan"), (s.run_name, "bold"), ("  ·  ", "dim"), (s.model_name, ""),
            ("  ·  ", "dim"), (s.phase.upper(), _phase_style(s.phase)),
        )
        line = Text.assemble(
            (f" step {s.step:,}/{s.total_steps:,} ", "bold"), (f"({frac * 100:5.1f}%)", ""),
            (f"   epoch {s.epoch}/{s.epochs}", ""), (f"   elapsed {_fmt_td(elapsed)}", "dim"),
            (f"   ETA {_fmt_td(eta)}" if s.step else "   ETA -", "bold green"),
            (f"   {s.samples_seen:,} samples", "dim"),
        )
        return Panel(Group(top, ProgressBar(total=1.0, completed=frac, width=None), line), padding=(0, 1))

    def _train(self) -> Panel:
        s = self.state
        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim"), t.add_column(justify="right")
        t.add_row("loss", _fmt(s.loss))
        t.add_row("loss (ema)", Text(_fmt(s.loss_ema), style="bold"))
        t.add_row("lr", _fmt(s.lr, ".2e"))
        t.add_row("grad norm", _fmt(s.grad_norm, ".3f"))
        t.add_row("tokens / s", f"{s.tokens_per_s:,.0f}")
        t.add_row("samples / s", f"{s.samples_per_s:.2f}")
        for name, value in s.extra.items():
            t.add_row(name, str(value))
        spark = Text(_spark(list(s.loss_hist)), style="cyan")
        return Panel(Group(t, spark), title="train", border_style="cyan")

    def _eval(self) -> Panel:
        s, e = self.state, self.state.eval
        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim"), t.add_column(justify="right")
        rows = [("val loss", e.get("val_loss"), ".4f"), ("node F1", e.get("node_f1"), ".3f"),
                ("edge F1", e.get("edge_f1"), ".3f"), ("edge+label F1", e.get("edge_label_f1"), ".3f"),
                ("kind acc", e.get("kind_acc"), ".3f"), ("valid output", e.get("valid_rate"), ".1%"),
                ("exact match", e.get("exact_rate"), ".1%")]
        for name, v, spec in rows:
            t.add_row(name, _fmt(v, spec))
        title = f"eval @ step {s.eval_step:,}" if s.eval_step is not None else "eval (pending)"
        best = Text(f"best edge F1 {_fmt(s.best, '.3f')}", style="green") if s.best is not None else Text("")
        return Panel(Group(t, best), title=title, border_style="magenta")

    def _system(self) -> Panel:
        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim"), t.add_column(justify="right")
        if _GPU is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(_GPU).gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU)
                temp = pynvml.nvmlDeviceGetTemperature(_GPU, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(_GPU) / 1000
                t.add_row("GPU util", f"{util}%")
                t.add_row("VRAM", f"{mem.used / 2**30:.1f} / {mem.total / 2**30:.0f} GB")
                t.add_row("GPU temp", f"{temp} °C")
                t.add_row("power", f"{power:.0f} W")
            except Exception:
                t.add_row("GPU", "n/a")
        vm = psutil.virtual_memory()
        t.add_row("RAM", f"{vm.used / 2**30:.1f} / {vm.total / 2**30:.0f} GB")
        t.add_row("CPU", f"{psutil.cpu_percent():.0f}%")
        return Panel(t, title="system", border_style="yellow")

    def _mix(self) -> Panel:
        total = sum(self.state.mix.values()) or 1
        parts = [f"{k} {v / total:.0%}" for k, v in self.state.mix.items()]
        return Panel(Text("  ·  ".join(parts) or "-", overflow="fold"), title="data mix (this epoch)",
                     border_style="blue")

    def _log(self, height: int) -> Panel:
        lines = list(self.tail.lines)[-max(1, height - 2):]
        text = Text()
        for level, msg in lines:
            style = "red" if level >= logging.ERROR else "yellow" if level >= logging.WARNING else ""
            text.append(msg + "\n", style=style)
        return Panel(text, title="log", border_style="white")

    def render(self) -> Layout:
        height = self.console.size.height
        layout = Layout()
        layout.split_column(
            Layout(self._header(), size=5),
            Layout(name="mid", size=11),
            Layout(self._mix(), size=3),
            Layout(name="log"),
        )
        layout["mid"].split_row(Layout(self._train()), Layout(self._eval()), Layout(self._system()))
        layout["log"].update(self._log(max(4, height - 19)))
        return layout


def _phase_style(phase: str) -> str:
    return {"training": "bold green", "evaluating": "bold magenta", "saving": "bold yellow",
            "done": "bold cyan", "stopped": "bold red"}.get(phase, "bold")
