#!/usr/bin/env python3
"""
attractor_farm.py — Live progress tracker for the WEXAC circuit scan.

Usage:
    python3 attractor_farm.py          # refresh every 60s
    python3 attractor_farm.py --fast   # refresh every 15s
"""
import subprocess, sys, time, re, random
from datetime import datetime, timedelta
from collections import deque

from rich import box
from rich.align  import Align
from rich.console import Console
from rich.layout  import Layout
from rich.live    import Live
from rich.panel   import Panel
from rich.table   import Table
from rich.text    import Text
from rich.columns import Columns

# ── config ────────────────────────────────────────────────────────────────────
SSH      = ["ssh", "-i", "/Users/Avimayo/.ssh/id_rsa",
            "-o", "ConnectTimeout=12", "-o", "StrictHostKeyChecking=no",
            "avimayo@access1.wexac.weizmann.ac.il"]
REFRESH  = 15 if "--fast" in sys.argv else 60
TOTAL_RETRY = 479    # circS3: remaining chunks resubmitted at %400, email suppressed
TOTAL_SCAN  = 2560   # full scan target (files on disk)

LOGO = r"""
   ___  ____  ____  ____  ___   ___  ____  __  ____     ____  ___   ____  __  __
  / _ |/_  /_/_  /_/ __/ / _ | / __/_/_  /_/ \/  __\   / __/ / _ | / __/ /  \/  \
 / __ | / /_  / /_/ __/ / , _// /__  / /_/ /   / _/   / __/ / , _// __/ / /\/\ /
/_/ |_|/___/ /___/___/ /_/|_| \___/ /___/_/   /_/    /_/   /_/|_| \___/ /_/  /_/
"""

TAGLINE = "🌾  Growing stable attractors since June 2026  🌾"

MESSAGES = [
    "🔬  NSolve is doing its best… Jacobians are nervous.",
    "⚗️   Steady states are playing hide-and-seek in parameter space.",
    "👻  Ghost attractors haunting the 3-D faces of the hypercube.",
    "📐  Most eigenvalues are negative.  Nature is predictable (mostly).",
    "🎲  Ten thousand random seeds walked into a bar…",
    "🧬  B→M is refusing to find hierarchy.  Classic feedback.",
    "🌀  Spiralling toward the fixed points of existence.",
    "⚡  25.6 million parameter samples.  That's commitment.",
    "🏔️   Traversing the energy landscape one sector at a time.",
    "💀  RIP state 1101. Never observed experimentally.",
    "🕵️   Semistable states — the dark matter of this circuit.",
    "🎯  Canonical hierarchy: {0000, 1000, 1100, 1111}.  Beauty.",
    "🧮  The Jacobian has entered the chat.",
    "🦠  B is desperately trying to invade.  Eigenvalue says no.",
    "📊  Forward edges destroy hierarchy.  Backward edges restore it.",
    "⏳  Patience is a virtue.  So is 4 GB of RAM per job.",
    "🔭  Looking for attractors in a 4-D landscape, one chunk at a time.",
    "🎓  Each circuit is a universe.  256 universes. 10 chunks each.",
    "💡  The answer is probably 42, but let's check all 256 circuits first.",
    "🍵  Good time for a coffee.  The cluster will handle it.",
]

# ── SSH fetch ─────────────────────────────────────────────────────────────────

REMOTE_CMD = r"""
{ bjobs -J 'circS' 2>/dev/null; bjobs -J 'circS3' 2>/dev/null; } | awk 'NR>1 {print $3}' | sort | uniq -c;
echo 'SEP_S';
ls ~/circuit_hpc/results_v2/*_r0_5_stable.csv 2>/dev/null | wc -l;
echo 'SEP_F';
ls ~/circuit_hpc/results_v2/*_r3_5_stable.csv 2>/dev/null | wc -l;
"""

def parse_counts(block):
    run = pend = 0
    for line in block.splitlines():
        m = re.match(r'\s*(\d+)\s+(\w+)', line)
        if m:
            n, stat = int(m.group(1)), m.group(2)
            if stat == "RUN":    run  += n
            elif stat == "PEND": pend += n
    return run, pend

def fetch():
    try:
        out = subprocess.check_output(
            SSH + [REMOTE_CMD], stderr=subprocess.DEVNULL, timeout=25
        ).decode()
    except Exception as e:
        return None, str(e)

    parts = re.split(r'SEP_[A-Z]+', out)
    if len(parts) < 3:
        return None, "unexpected output"

    r_run, r_pend = parse_counts(parts[0])

    try:
        done_r05 = int(parts[1].strip())
        done_r35 = int(parts[2].strip())
    except ValueError:
        done_r05 = done_r35 = 0

    return {
        "r_run":    r_run,
        "r_pend":   r_pend,
        "done_r05": done_r05,
        "done_r35": done_r35,
        "ts":       datetime.now(),
    }, None

# ── rendering helpers ─────────────────────────────────────────────────────────

def bar(done, total, width=38):
    if total == 0:
        frac = 1.0
    else:
        frac = min(done / total, 1.0)
    filled = int(frac * width)
    empty  = width - filled
    pct    = frac * 100

    if frac >= 1.0:
        color = "bright_green"
        fill  = "█" * width
    elif frac > 0.6:
        color = "green"
        fill  = "█" * filled + "▓" + "░" * max(empty - 1, 0)
    elif frac > 0.25:
        color = "yellow"
        fill  = "█" * filled + "▒" + "░" * max(empty - 1, 0)
    else:
        color = "red"
        fill  = "█" * filled + "░" * empty

    bar_str  = f"[{color}]{fill}[/{color}]"
    pct_str  = f"[bold]{pct:5.1f}%[/bold]"
    cnt_str  = f"[dim]{done:4d}/{total}[/dim]"
    return f"{bar_str} {pct_str}  {cnt_str}"

def status_icon(run, pend, done, total):
    if done >= total:   return "[bright_green bold]✅ DONE![/]"
    if run  > 0:        return "[yellow bold]⚡ RUNNING[/]"
    if pend > 0:        return "[cyan bold]⏳ QUEUED[/]"
    return "[dim]❓ UNKNOWN[/]"

def eta_string(done, total, history):
    """history = deque of (timestamp, done_count)"""
    if done >= total:
        return "[bright_green]Finished![/]"
    if len(history) < 2:
        return "[dim]calculating…[/]"
    t0, d0 = history[0]
    t1, d1 = history[-1]
    dt = (t1 - t0).total_seconds()
    dd = d1 - d0
    if dt < 1 or dd <= 0:
        return "[dim]calculating…[/]"
    rate = dd / dt          # chunks per second
    remaining = total - done
    secs = remaining / rate
    eta  = timedelta(seconds=int(secs))
    return f"[cyan]~{eta}[/]"

# ── layout builder ────────────────────────────────────────────────────────────

def build_screen(st, history, msg, countdown):
    # ---- header ----
    header = Panel(
        Align.center(
            Text(LOGO, style="bold bright_cyan", justify="center") +
            Text("\n" + TAGLINE, style="bold yellow", justify="center")
        ),
        box=box.DOUBLE_EDGE,
        style="bright_cyan",
        padding=(0, 2),
    )

    if st is None:
        body = Panel("[red bold]⚠  Could not reach WEXAC — is VPN on?[/]",
                     title="CONNECTION ERROR", border_style="red")
        footer = Panel(f"[dim]Retrying in {countdown}s…[/]", border_style="dim")
        return Columns([header, body, footer], equal=False)

    # ---- derived counts ----
    a_done  = TOTAL_A  - st["a_run"]  - st["a_pend"]
    b_done  = TOTAL_B  - st["b_run"]  - st["b_pend"]
    bm_done = TOTAL_BM - st["bm_run"] - st["bm_pend"]
    a_done  = max(a_done,  0)
    b_done  = max(b_done,  0)
    bm_done = max(bm_done, 0)

    main_done  = st["done_r05"]
    main_total = TOTAL_A + TOTAL_B   # 2560
    main_run   = st["a_run"] + st["b_run"]
    main_pend  = st["a_pend"] + st["b_pend"]

    # ---- field progress table ----
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1),
              expand=True, border_style="dim")
    t.add_column("label", style="bold", width=32)
    t.add_column("bar",   ratio=1)
    t.add_column("icon",  width=14)

    t.add_row(
        "[bright_white]🌾 FIELD A[/]  [dim]circuits  1-128[/]",
        bar(a_done, TOTAL_A),
        status_icon(st["a_run"], st["a_pend"], a_done, TOTAL_A),
    )
    t.add_row(
        "[bright_white]🌾 FIELD B[/]  [dim]circuits 129-256[/]",
        bar(b_done, TOTAL_B),
        status_icon(st["b_run"], st["b_pend"], b_done, TOTAL_B),
    )
    t.add_row("", "", "")
    t.add_row(
        "[bold magenta]🎯 B→M FOCUSED[/]  [dim]range [3,5][/]",
        bar(bm_done, TOTAL_BM),
        status_icon(st["bm_run"], st["bm_pend"], bm_done, TOTAL_BM),
    )
    t.add_row("", "", "")
    t.add_row(
        "[bold bright_white]📦 TOTAL MAIN SCAN[/]",
        bar(main_done, main_total),
        status_icon(main_run, main_pend, main_done, main_total),
    )

    fields_panel = Panel(t, title="[bold]🚜  FIELD STATUS[/]",
                         border_style="green", padding=(1, 2))

    # ---- stats table ----
    s = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 1),
              border_style="dim")
    s.add_column("",      style="bold", width=20)
    s.add_column("Main",  justify="right", width=8)
    s.add_column("B→M",   justify="right", width=8)

    s.add_row("[yellow]⚡ Running[/]",
              f"[yellow]{main_run}[/]",
              f"[yellow]{st['bm_run']}[/]")
    s.add_row("[cyan]⏳ Pending[/]",
              f"[cyan]{main_pend}[/]",
              f"[cyan]{st['bm_pend']}[/]")
    s.add_row("[bright_green]✅ Done[/]",
              f"[bright_green]{main_done}[/]",
              f"[bright_green]{bm_done}[/]")
    s.add_row("[dim]━━━━━━━━━━━━━━[/]", "[dim]━━━━━━━[/]", "[dim]━━━━━━━[/]")
    s.add_row("Total chunks",
              f"[bold]{main_total}[/]",
              f"[bold]{TOTAL_BM}[/]")
    s.add_row("Samples/chunk",
              "[bold]10 000[/]", "[bold]10 000[/]")
    s.add_row("Total samples",
              f"[bold]{main_done * 10000:,}[/]",
              f"[bold]{bm_done * 10000:,}[/]")

    harvest_eta = eta_string(main_done, main_total, history)
    s.add_row("[dim]━━━━━━━━━━━━━━[/]", "[dim]━━━━━━━[/]", "[dim]━━━━━━━[/]")
    s.add_row("⏱  ETA", harvest_eta, "")

    stats_panel = Panel(s, title="[bold]📊  STATS[/]",
                        border_style="blue", padding=(1, 1))

    # ---- results panel ----
    r = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), expand=True)
    r.add_column("label", style="bold")
    r.add_column("val",   justify="right")

    stable_files = st["done_r05"] + st["done_r35"]
    r.add_row("🌱 Stable CSVs written",    f"[bright_green]{stable_files * 2}[/]")
    r.add_row("👻 Semistable CSVs written", f"[magenta]{stable_files * 2}[/]")
    r.add_row("🔬 [0,5] chunks harvested",  f"[green]{st['done_r05']}[/]  / 2560")
    r.add_row("🎯 [3,5] chunks harvested",  f"[magenta]{st['done_r35']}[/]  / 10")

    results_panel = Panel(r, title="[bold]🌽  HARVEST[/]",
                          border_style="magenta", padding=(1, 1))

    # ---- message + footer ----
    ts  = st["ts"].strftime("%H:%M:%S")
    footer_txt = (
        f"[dim]💬  {msg}[/]\n\n"
        f"[dim]🕐  Last updated: [bold]{ts}[/]  •  "
        f"Next refresh in [bold]{countdown}s[/][/]"
    )
    footer = Panel(Align.center(footer_txt),
                   border_style="dim", padding=(0, 2))

    right = Layout(name="right")
    right.split_column(
        Layout(stats_panel,   name="stats"),
        Layout(results_panel, name="harvest"),
    )
    inner = Layout(name="inner")
    inner.split_row(
        Layout(fields_panel, name="fields", ratio=3),
        right,
    )
    return Layout(Panel(inner, box=box.MINIMAL, border_style="dim")), header, footer

# ── main loop ─────────────────────────────────────────────────────────────────

def main():
    console  = Console()
    history  = deque(maxlen=20)   # (timestamp, done_count) for ETA
    msg_idx  = 0
    state    = None
    error    = None

    console.print(f"\n[bold bright_cyan]🌾  ATTRACTOR FARM  —  connecting to WEXAC…[/]\n")

    with Live(console=console, refresh_per_second=2, screen=True) as live:
        next_fetch = 0

        while True:
            now = time.time()

            if now >= next_fetch:
                state, error = fetch()
                if state:
                    history.append((state["ts"], state["done_r05"]))
                msg_idx    = (msg_idx + random.randint(1, 4)) % len(MESSAGES)
                next_fetch = now + REFRESH

            countdown = max(0, int(next_fetch - time.time()))

            # ── assemble display ──
            outer = Layout()
            outer.split_column(
                Layout(name="header", size=8),
                Layout(name="body",   ratio=1),
                Layout(name="footer", size=5),
            )

            # header
            outer["header"].update(
                Panel(
                    Align.center(
                        Text(TAGLINE + "\n", style="bold yellow", justify="center")
                    ),
                    title="[bold bright_cyan]🌾  ATTRACTOR FARM  🌾[/]",
                    subtitle=f"[dim]WEXAC · {datetime.now().strftime('%Y-%m-%d')}[/]",
                    box=box.DOUBLE_EDGE,
                    border_style="bright_cyan",
                )
            )

            if state is None:
                outer["body"].update(
                    Panel(
                        Align.center(
                            Text("\n⚠  Could not reach WEXAC\n"
                                 "    Check VPN or SSH key.\n\n"
                                 f"Error: {error}", style="bold red", justify="center")
                        ),
                        border_style="red",
                    )
                )
            else:
                r_done     = max(TOTAL_RETRY - state["r_run"] - state["r_pend"], 0)
                main_done  = state["done_r05"]
                main_run   = state["r_run"]
                main_pend  = state["r_pend"]

                # -- fields --
                ft = Table(box=box.SIMPLE, show_header=False,
                           padding=(0, 1), expand=True)
                ft.add_column("label", style="bold", width=44)
                ft.add_column("bar",   ratio=1)
                ft.add_column("icon",  width=16)

                ft.add_row(
                    "⚡ [bold yellow]circS3 RETRY[/]  [dim]479 chunks · %400 · stable-only · 5k samples[/]",
                    bar(r_done,   TOTAL_RETRY),
                    status_icon(state["r_run"], state["r_pend"], r_done, TOTAL_RETRY),
                )
                ft.add_row("", "", "")
                ft.add_row(
                    "[bold bright_white]📦 TOTAL  [dim](files on disk)[/][/]",
                    bar(main_done, TOTAL_SCAN),
                    status_icon(main_run, main_pend, main_done, TOTAL_SCAN),
                )

                fields_panel = Panel(ft,
                    title="[bold]🚜  FIELD STATUS[/]",
                    border_style="green", padding=(1, 2))

                # -- stats --
                st2 = Table(box=box.SIMPLE_HEAD, show_header=False,
                            padding=(0, 1), border_style="dim")
                st2.add_column("label", style="bold", width=24)
                st2.add_column("val",   justify="right", width=12)

                st2.add_row("[yellow]⚡ Running[/]",   f"[yellow]{state['r_run']}[/]")
                st2.add_row("[cyan]⏳ Pending[/]",    f"[cyan]{state['r_pend']}[/]")
                st2.add_row("[bright_green]✅ Retry done[/]", f"[bright_green]{r_done}[/] / {TOTAL_RETRY}")
                st2.add_row("[dim]──────────────────────[/]", "[dim]──────────[/]")
                st2.add_row("Files on disk",  f"[bold]{main_done}[/] / {TOTAL_SCAN}")
                st2.add_row("Samples (est.)", f"[bold]{main_done * 7500:,}[/]")
                st2.add_row("[dim]──────────────────────[/]", "[dim]──────────[/]")
                st2.add_row("⏱  ETA",
                            eta_string(main_done, TOTAL_SCAN, history), "")

                stats_panel = Panel(st2,
                    title="[bold]📊  STATS[/]",
                    border_style="blue", padding=(1, 1))

                # -- harvest --
                ht = Table(box=box.SIMPLE, show_header=False,
                           padding=(0, 1), expand=True)
                ht.add_column("label", style="bold")
                ht.add_column("val",   justify="right")

                ht.add_row("🌱 [0,5] stable chunks",
                           f"[bright_green]{state['done_r05']:4d}[/] / {TOTAL_SCAN}")
                ht.add_row("🎯 [3,5] B→M chunks",
                           f"[bright_green]{state['done_r35']:4d}[/] / 10")
                ht.add_row("[dim]────────────────────[/]", "[dim]──────[/]")
                ht.add_row("👻 Semistable CSVs",
                           f"[magenta]~1669[/] circuits")

                harvest_panel = Panel(ht,
                    title="[bold]🌽  HARVEST[/]",
                    border_style="magenta", padding=(1, 1))

                right_col = Layout(name="right_col")
                right_col.split_column(
                    Layout(stats_panel,   name="stats"),
                    Layout(harvest_panel, name="harvest"),
                )
                body_layout = Layout()
                body_layout.split_row(
                    Layout(fields_panel, name="fields", ratio=3),
                    right_col,
                )
                outer["body"].update(body_layout)

            # footer
            footer_txt = (
                f"[italic dim]{MESSAGES[msg_idx]}[/]\n\n"
                f"[dim]🕐  {state['ts'].strftime('%H:%M:%S') if state else '??:??:??'}"
                f"  •  next refresh in [bold]{countdown}s[/][/]"
            )
            outer["footer"].update(
                Panel(Align.center(footer_txt),
                      border_style="dim", padding=(0, 2))
            )

            live.update(outer)
            time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        Console().print("\n[bold yellow]👋  Harvest paused. Come back soon![/]\n")
