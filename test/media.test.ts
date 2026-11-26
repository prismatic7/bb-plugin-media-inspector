// Media-inspector tests: the parsers, plus real exiftool/ffprobe runs against
// files generated on the spot when those binaries are installed.
//
// Run: node --test test/media.test.ts
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { audioReport, imageReport, renderReport, summariseGraph } from "../server.ts";

const has = (bin: string, args: string[]) => {
  try {
    execFileSync(bin, args, { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
};

const PROMPT = JSON.stringify({
  "3": { class_type: "KSampler", inputs: { seed: 12345, steps: 28, cfg: 6.5, sampler_name: "euler", scheduler: "normal" } },
  "4": { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: "flux2-klein.safetensors" } },
  "5": { class_type: "LoraLoader", inputs: { lora_name: "grain.safetensors" } },
  "6": { class_type: "EmptyLatentImage", inputs: { width: 1024, height: 1024 } },
});

test("summariseGraph reads the recipe out of an API-format graph", () => {
  const g = summariseGraph(PROMPT);
  assert.ok(g);
  assert.equal(g.nodes, 4);
  assert.deepEqual(g.models, ["flux2-klein.safetensors"]);
  assert.deepEqual(g.loras, ["grain.safetensors"]);
  assert.equal(g.sampling.sampler, "euler");
  assert.equal(g.sampling.steps, 28);
  assert.equal(g.sampling.cfg, 6.5);
  assert.equal(g.sampling.seed, 12345);
  assert.deepEqual(g.size, { width: 1024, height: 1024 });
});

test("summariseGraph rejects a non-graph chunk", () => {
  assert.equal(summariseGraph("{\"hello\":true}"), null);
  assert.equal(summariseGraph("not json"), null);
});

test("imageReport detects the embedded ComfyUI chunks", () => {
  const r = imageReport({ SourceFile: "a.png", FileType: "PNG", ImageWidth: 1024, ImageHeight: 768, Prompt: "{}", Workflow: "{}" }, "a.png");
  assert.equal(r.embeddedGraph, "prompt+workflow");
  assert.equal(r.width, 1024);
  const none = imageReport({ SourceFile: "b.png", FileType: "PNG" });
  assert.equal(none.embeddedGraph, null);
});

test("audioReport checks Whisper readiness", () => {
  const mono16 = audioReport({ streams: [{ codec_type: "audio", codec_name: "pcm_s16le", sample_rate: "16000", channels: 1, channel_layout: "mono" }], format: { duration: "3.5" } });
  assert.equal(mono16.whisperReady, true);
  const stereo48 = audioReport({ streams: [{ codec_type: "audio", codec_name: "pcm_s24le", sample_rate: "48000", channels: 2 }], format: { duration: "3.5" } });
  assert.equal(stereo48.whisperReady, false);
  assert.equal(stereo48.whisperNotes.length, 2);
});

test("renderReport drops empty rows", () => {
  const out = renderReport("file", [["format", "PNG"], ["software", null], ["size", "10×10"]]);
  assert.equal(out, "file\nformat         PNG\nsize           10×10");
});

test("live: ffprobe on a generated WAV", (t) => {
  if (!has("ffmpeg", ["-version"]) || !has("ffprobe", ["-version"])) return t.skip("ffmpeg/ffprobe missing");
  const dir = mkdtempSync(join(tmpdir(), "media-"));
  const wav = join(dir, "tone.wav");
  execFileSync("ffmpeg", ["-v", "quiet", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-ar", "48000", "-ac", "2", wav]);
  const probe = JSON.parse(execFileSync("ffprobe", ["-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", wav], { encoding: "utf8" }));
  const a = audioReport(probe);
  assert.equal(a.sampleRate, 48000);
  assert.equal(a.channels, 2);
  assert.equal(a.whisperReady, false);
  assert.ok(a.durationSec && a.durationSec > 0.5);
});

test("live: exiftool on a generated PNG reports no ComfyUI graph", (t) => {
  if (!has("ffmpeg", ["-version"]) || !has("exiftool", ["-ver"])) return t.skip("ffmpeg/exiftool missing");
  const dir = mkdtempSync(join(tmpdir(), "media-"));
  const png = join(dir, "plain.png");
  execFileSync("ffmpeg", ["-v", "quiet", "-f", "lavfi", "-i", "color=c=black:s=64x48", "-frames:v", "1", png]);
  const raw = JSON.parse(execFileSync("exiftool", ["-j", png], { encoding: "utf8" }))[0];
  const r = imageReport(raw, png);
  assert.equal(r.width, 64);
  assert.equal(r.height, 48);
  assert.equal(r.embeddedGraph, null);
});
