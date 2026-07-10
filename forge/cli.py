"""`forge` — one front door for the erewhon code agents.

A Typer app built from :data:`forge.registry.REGISTRY`. Each verb is a thin passthrough: it
forwards the remaining argv to the agent's ``main(argv)`` in-process, so each agent keeps its own
argparse parser (no arg definitions are duplicated here). ``forge research --help`` is delegated to
the agent's own parser via ``add_help_option=False``.
"""

from __future__ import annotations

import typer

from forge.registry import REGISTRY, AgentCommand

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Erewhon code agents — one front door. Run `forge <verb> --help` for an agent's options.",
)

# Extra args (the agent's flags + positionals) flow through untouched to its argparse parser.
_PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}


def _register(cmd: AgentCommand) -> None:
    # `cmd` is bound per _register() call, so the closure needs no late-bind guard. And Typer turns
    # any extra function parameter into a CLI option, so `ctx` must be the only parameter here.
    @app.command(
        name=cmd.name,
        help=cmd.summary,
        add_help_option=False,  # let the agent's own argparse render `forge <verb> --help`
        context_settings=_PASSTHROUGH,
    )
    def _verb(ctx: typer.Context) -> None:
        raise typer.Exit(cmd.load_main()(ctx.args) or 0)


for _cmd in REGISTRY:
    if _cmd.exposes_cli:
        _register(_cmd)


def main() -> None:
    """Console-script entry point (``[project.scripts] forge``)."""
    app()


if __name__ == "__main__":
    main()
