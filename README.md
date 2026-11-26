# bb-plugin-media-inspector

Read what a media file actually is, instead of guessing at it.

## Tools

### `media_inspect(path)`

Images: format, pixel size, colour type, software, and whether the file carries
an embedded ComfyUI graph. Audio: codec, sample rate, channels, bit depth, bit
rate, duration, and a Whisper-readiness verdict (16 kHz mono or not).

### `comfy_graph(path, raw?)`

Reads the ComfyUI API-format graph out of a generated PNG: node count and
types, checkpoints, VAEs, LoRAs, sampler/scheduler, steps, cfg, seed and output
size. `raw: true` also returns the prompt JSON.

## Requirements

- `exiftool` on PATH (or set its absolute path in settings)
- `ffprobe` on PATH (or set its absolute path in settings)

## Install

```sh
npm install
npm run bundle
bb plugin install . --yes
bb plugin reload media-inspector
```

## Development

```sh
npm run build      # tsc typecheck
npm test           # parsers + live exiftool/ffprobe runs on generated files
npm run bundle     # bb plugin build → dist/
```

The live tests generate their own fixtures with ffmpeg, so they prove the
parsers against real binary output rather than hand-written JSON. They skip
cleanly when the binaries are absent.
