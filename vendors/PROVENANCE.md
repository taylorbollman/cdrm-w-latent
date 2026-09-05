# Vendor sources

The old roadmap vendors directories were empty. These populated sources were
copied from the local jepa-gpt-rl checkout before its deletion, without `.git`
metadata or submodule remote configuration. Original licenses remain included.

- `apex`: `064d6d5b4b0f417ae21c522b504e6da544fb265c`
- `vllm`: `9898f94abe005f675e0a7f40b0ae891afb5cf992`
- `flash-attention`: `0409f9adcbdebff6cc19eb95f370d40e896980bc`

## Recurrent transformer

`recurrent-transformer` is a Git submodule, with history and Apache-2.0 license
retained. Its origin is `https://github.com/taylorbollman/recurrent-transformer.git`,
forked from `geniucos/recurrent-transformer` at initial revision
`a21b42d2bc292edb86ed1b62cee4bcab809a9d21`. The parent repository records the exact
revision through the submodule pointer.
