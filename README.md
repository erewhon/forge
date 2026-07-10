# forge

Front door for the erewhon code agents — a `forge` CLI and a `forge-mcp` MCP server
wrapping a set of coding-focused agents: code review ensembles, parallel model
comparison, research harnesses, dependency bumping, testing loops, and the
architect/worker coding pipeline.

## Install

```sh
uv sync
```

Optional [Nous](https://github.com/erewhon/nous) notebook integration (task
tracking, review sinks):

```sh
uv sync --extra nous
```

## Usage

```sh
forge --help       # CLI verbs
forge-mcp          # MCP server (stdio)
```

## License

Apache-2.0
