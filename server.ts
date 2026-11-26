// bb-plugin-media-inspector — read what a media file actually is.
//
// Two questions this answers that a coding agent otherwise guesses at:
//
//   1. What generated this image? A ComfyUI PNG carries its whole prompt
//      graph in PNG text chunks. exiftool surfaces them; comfy_graph() reads
//      the graph back out (models, sampler, steps, cfg, seed, size).
//   2. Is this audio usable for speech-to-text? ffprobe knows; whisper wants
//      16 kHz mono. audioReport() checks the file against that.
//
// exiftool and ffprobe are called as external binaries. No third-party code is
// copied. The approach follows the agy-media-inspector plugin on this machine;
// the implementation is new TypeScript.
import { type BbPluginApi } from "@get-bb/plugin-sdk";
import { z } from "zod";
import { execFile as execFileCb } from "node:child_process";
import { promisify } from "node:util";
import { extname, isAbsolute, resolve } from "node:path";
import { existsSync } from "node:fs";

const execFileAsync = promisify(execFileCb);

const IMAGE_EXTS = [".png", ".webp", ".jpg", ".jpeg", ".tif", ".tiff"];
const AUDIO_EXTS = [".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".ogg", ".opus", ".caf"];

// ─── Pure parsers (tested without touching disk) ──────────────────────

export interface ImageReport {
  file: string;
  format: string;
  width: number | null;
  height: number | null;
  colourType: string | null;
  software: string | null;
  embeddedGraph: "prompt+workflow" | "prompt" | "workflow" | null;
  textKeys: string[];
}

/** Summarise `exiftool -j` output for one file. */
export function imageReport(meta: Record<string, any>, file = ""): ImageReport {
  const keys = Object.keys(meta ?? {});
  const lower = keys.map((k) => k.toLowerCase());
  const graphKeys: string[] = [];
  for (const [k, v] of Object.entries(meta ?? {})) {
    if (typeof v !== "string" || !v.trim().startsWith("{")) continue;
    const n = k.toLowerCase();
    if (n === "prompt" || n === "workflow" || n === "comfyui_prompt" || n === "comfyui_workflow") {
      graphKeys.push(n);
    }
  }
  const embeddedGraph =
    graphKeys.includes("prompt") && graphKeys.includes("workflow")
      ? "prompt+workflow"
      : graphKeys.includes("prompt")
        ? "prompt"
        : graphKeys.includes("workflow")
          ? "workflow"
          : null;
  return {
    file: file || (meta?.SourceFile ?? ""),
    format: meta?.FileType ?? meta?.FileTypeExtension ?? "unknown",
    width: meta?.ImageWidth ?? meta?.ImageSize?.split?.("x")?.[0] ?? null,
    height: meta?.ImageHeight ?? meta?.ImageSize?.split?.("x")?.[1] ?? null,
    colourType: meta?.ColorType ?? meta?.BitsPerSample ?? null,
    software: meta?.Software ?? meta?.CreatorTool ?? null,
    embeddedGraph,
    textKeys: keys.filter((k) => lower[lower.indexOf(k.toLowerCase())] !== "").filter((k) =>
      /^(prompt|workflow|parameters|description|comment|software|creator|parameters|xmp)/i.test(k),
    ),
  };
}

/** ComfyUI prompt graph → the facts worth reporting. */
export interface GraphSummary {
  nodes: number;
  nodeTypes: Array<{ type: string; count: number }>;
  models: string[];
  sampling: {
    sampler: string | null;
    scheduler: string | null;
    steps: number | null;
    cfg: number | null;
    seed: number | null;
  };
  size: { width: number | null; height: number | null };
  loras: string[];
}

/** Parse the "prompt" chunk of a ComfyUI PNG (the API-format graph). */
export function summariseGraph(promptJson: string): GraphSummary | null {
  let graph: any;
  try {
    graph = JSON.parse(promptJson);
  } catch {
    return null;
  }
  if (!graph || typeof graph !== "object") return null;
  const entries = Object.values(graph) as any[];
  if (!entries.length || !entries.every((n) => n && typeof n === "object" && "class_type" in n)) {
    return null;
  }
  const counts = new Map<string, number>();
  const models = new Set<string>();
  const loras = new Set<string>();
  const sampling = { sampler: null as string | null, scheduler: null as string | null, steps: null as number | null, cfg: null as number | null, seed: null as number | null };
  const size = { width: null as number | null, height: null as number | null };

  for (const node of entries) {
    const type = String(node.class_type);
    counts.set(type, (counts.get(type) ?? 0) + 1);
    const inputs = node.inputs ?? {};
    for (const key of ["ckpt_name", "unet_name", "vae_name", "clip_name"]) {
      if (typeof inputs[key] === "string") models.add(inputs[key]);
    }
    if (typeof inputs.lora_name === "string") loras.add(inputs.lora_name);
    if (typeof inputs.sampler_name === "string") sampling.sampler = inputs.sampler_name;
    if (typeof inputs.scheduler === "string") sampling.scheduler = inputs.scheduler;
    if (typeof inputs.steps === "number") sampling.steps = inputs.steps;
    if (typeof inputs.cfg === "number") sampling.cfg = inputs.cfg;
    if (typeof inputs.seed === "number") sampling.seed = inputs.seed;
    if (typeof inputs.noise_seed === "number" && sampling.seed === null) sampling.seed = inputs.noise_seed;
    if (typeof inputs.width === "number") size.width = inputs.width;
    if (typeof inputs.height === "number") size.height = inputs.height;
  }

  return {
    nodes: entries.length,
    nodeTypes: [...counts.entries()].map(([type, count]) => ({ type, count })).sort((a, b) => b.count - a.count),
    models: [...models],
    sampling,
    size,
    loras: [...loras],
  };
}

export interface AudioReport {
  codec: string | null;
  sampleRate: number | null;
  channels: number | null;
  channelLayout: string | null;
  bitDepth: number | null;
  bitRate: number | null;
  durationSec: number | null;
  whisperReady: boolean;
  whisperNotes: string[];
}

/** Summarise `ffprobe -print_format json -show_format -show_streams`. */
export function audioReport(probe: any): AudioReport {
  const stream = (probe?.streams ?? []).find((s: any) => s.codec_type === "audio") ?? {};
  const sampleRate = stream.sample_rate ? Number(stream.sample_rate) : null;
  const channels = typeof stream.channels === "number" ? stream.channels : null;
  const duration = Number(probe?.format?.duration ?? stream.duration ?? 0) || null;
  const notes: string[] = [];
  if (sampleRate !== 16000) notes.push(`resample to 16 kHz (is ${sampleRate ?? "unknown"} Hz)`);
  if (channels !== 1) notes.push(`downmix to mono (is ${channels ?? "unknown"} channel(s))`);
  return {
    codec: stream.codec_name ?? null,
    sampleRate,
    channels,
    channelLayout: stream.channel_layout ?? null,
    bitDepth: stream.bits_per_raw_sample ? Number(stream.bits_per_raw_sample) : (stream.bits_per_sample ? Number(stream.bits_per_sample) : null),
    bitRate: stream.bit_rate ? Number(stream.bit_rate) : (probe?.format?.bit_rate ? Number(probe.format.bit_rate) : null),
    durationSec: duration,
    whisperReady: notes.length === 0,
    whisperNotes: notes.length ? notes : ["16 kHz mono — ready for Whisper"],
  };
}

/** Render a report as compact key/value lines. */
export function renderReport(title: string, rows: Array<[string, unknown]>): string {
  const body = rows
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([k, v]) => `${k.padEnd(14)} ${Array.isArray(v) ? v.join(", ") : String(v)}`)
    .join("\n");
  return `${title}\n${body}`;
}

// ─── Plugin ───────────────────────────────────────────────────────────

export default async function plugin(bb: BbPluginApi) {
  bb.log.info("bb-plugin-media-inspector loaded");

  const settings = bb.settings.define({
    exiftoolPath: { type: "string" as const, label: "exiftool", default: "", description: "Empty = `exiftool` from PATH." },
    ffprobePath: { type: "string" as const, label: "ffprobe", default: "", description: "Empty = `ffprobe` from PATH." },
  });
  let { exiftoolPath, ffprobePath } = await settings.get();
  settings.onChange((next) => {
    exiftoolPath = next.exiftoolPath;
    ffprobePath = next.ffprobePath;
  });

  const exiftool = () => (exiftoolPath.trim() || "exiftool");
  const ffprobe = () => (ffprobePath.trim() || "ffprobe");

  function resolvePath(path: string, projectDir: string | null): string {
    const base = projectDir ?? process.cwd();
    return isAbsolute(path) ? path : resolve(base, path);
  }

  async function exiftoolJson(file: string) {
    const { stdout } = await execFileAsync(exiftool(), ["-j", "-G1", file], { maxBuffer: 32 * 1024 * 1024, timeout: 60_000 });
    const parsed = JSON.parse(stdout);
    const raw = Array.isArray(parsed) ? parsed[0] : parsed;
    // exiftool -G1 prefixes keys with the group ("PNG:Prompt"); strip it back off.
    const out: Record<string, any> = {};
    for (const [k, v] of Object.entries(raw ?? {})) out[k.includes(":") ? k.split(":").slice(1).join(":") : k] = v;
    return out;
  }

  async function ffprobeJson(file: string) {
    const { stdout } = await execFileAsync(
      ffprobe(),
      ["-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", file],
      { maxBuffer: 8 * 1024 * 1024, timeout: 60_000 },
    );
    return JSON.parse(stdout);
  }

  bb.agents.registerTool({
    name: "media_inspect",
    description:
      "Inspect an image or audio file and report what it actually is. Images: format, size, " +
      "colour type, software, and whether the file carries an embedded ComfyUI graph. Audio: " +
      "codec, sample rate, channels, bit depth, duration, and whether it is already usable for " +
      "Whisper (16 kHz mono). Use it before guessing at parameters or re-encoding.",
    parameters: z.object({
      path: z.string().min(1).describe("File path, relative to the project root or absolute."),
    }),
    async execute({ path }, { threadId, projectId }) {
      const dir = await threadCwd(bb, threadId, projectId);
      const file = resolvePath(path, dir);
      if (!existsSync(file)) return `media_inspect: no such file: ${file}`;
      const ext = extname(file).toLowerCase();
      try {
        if (IMAGE_EXTS.includes(ext)) {
          const meta = await exiftoolJson(file);
          const r = imageReport(meta, file);
          return renderReport(`${r.file} (image)`, [
            ["format", r.format],
            ["size", r.width && r.height ? `${r.width}×${r.height}` : null],
            ["colour", r.colourType],
            ["software", r.software],
            ["comfy graph", r.embeddedGraph ?? "none"],
            ["text tags", r.textKeys],
          ]);
        }
        if (AUDIO_EXTS.includes(ext)) {
          const probe = await ffprobeJson(file);
          const a = audioReport(probe);
          return renderReport(`${file} (audio)`, [
            ["codec", a.codec],
            ["sample rate", a.sampleRate ? `${a.sampleRate} Hz` : null],
            ["channels", a.channels ? `${a.channels} (${a.channelLayout})` : null],
            ["bit depth", a.bitDepth],
            ["bit rate", a.bitRate ? `${Math.round(a.bitRate / 1000)} kb/s` : null],
            ["duration", a.durationSec ? `${a.durationSec.toFixed(2)} s` : null],
            ["whisper", a.whisperNotes.join("; ")],
          ]);
        }
        const meta = await exiftoolJson(file);
        const r = imageReport(meta, file);
        return renderReport(`${file} (unrecognised media)`, [["format", r.format], ["text tags", r.textKeys]]);
      } catch (err: any) {
        return `media_inspect failed: ${(err?.stderr || err?.message || String(err)).toString().slice(0, 600)}`;
      }
    },
  });

  bb.agents.registerTool({
    name: "comfy_graph",
    description:
      "Read the ComfyUI graph embedded in a generated PNG: node count and types, checkpoints and " +
      "VAEs, LoRAs, sampler/scheduler, steps, cfg, seed and output size. Use it to explain or " +
      "reproduce an image instead of guessing at the recipe.",
    parameters: z.object({
      path: z.string().min(1).describe("Path to a ComfyUI-generated image."),
      raw: z.boolean().optional().describe("Also return the prompt JSON verbatim (can be large)."),
    }),
    async execute({ path, raw }, { threadId, projectId }) {
      const dir = await threadCwd(bb, threadId, projectId);
      const file = resolvePath(path, dir);
      if (!existsSync(file)) return `comfy_graph: no such file: ${file}`;
      try {
        const meta = await exiftoolJson(file);
        const promptKey = Object.keys(meta).find((k) => k.toLowerCase() === "prompt");
        if (!promptKey || typeof meta[promptKey] !== "string") {
          return `comfy_graph: no ComfyUI prompt chunk in ${file}. Text tags present: ${
            Object.keys(meta).filter((k) => typeof meta[k] === "string" && meta[k].trim().startsWith("{")).join(", ") || "none"
          }`;
        }
        const summary = summariseGraph(meta[promptKey]);
        if (!summary) return "comfy_graph: the prompt chunk is present but is not an API-format graph.";
        const head = renderReport(`${file} (ComfyUI graph)`, [
          ["nodes", summary.nodes],
          ["models", summary.models],
          ["loras", summary.loras],
          ["sampler", [summary.sampling.sampler, summary.sampling.scheduler].filter(Boolean).join(" / ")],
          ["steps", summary.sampling.steps],
          ["cfg", summary.sampling.cfg],
          ["seed", summary.sampling.seed],
          ["size", summary.size.width && summary.size.height ? `${summary.size.width}×${summary.size.height}` : null],
          ["node types", summary.nodeTypes.map((n) => `${n.type}×${n.count}`)],
        ]);
        return raw ? `${head}\n\n--- prompt JSON ---\n${meta[promptKey]}` : head;
      } catch (err: any) {
        return `comfy_graph failed: ${(err?.stderr || err?.message || String(err)).toString().slice(0, 600)}`;
      }
    },
  });

  bb.onDispose(() => {
    bb.log.info("bb-plugin-media-inspector disposed");
  });
}

/** The directory this thread actually works in: its environment's path, not
 *  the project root. A worktree or an attached workspace is a different
 *  directory from the project, and resolving paths against the project puts
 *  files in the wrong tree. */
export async function threadCwd(bb: BbPluginApi, threadId: string, projectId: string): Promise<string> {
  const asPath = (v: any): string | null =>
    v && typeof v === "object" && typeof v.path === "string" && v.path.trim() ? v.path : null;
  try {
    const t: any = await bb.sdk.threads.get({ threadId } as any);
    const thread = t?.thread ?? t;
    const envId = thread?.environmentId;
    if (envId) {
      const e: any = await bb.sdk.environments.get({ environmentId: envId } as any);
      const p = asPath(e?.environment ?? e);
      if (p) return p;
    }
  } catch {
    // fall through to the project
  }
  try {
    const p: any = await bb.sdk.projects.get({ projectId } as any);
    const sources = (p?.project ?? p)?.sources ?? [];
    const src = Array.isArray(sources) ? sources.find((s: any) => s.type === "local_path") : null;
    if (src?.path) return src.path;
  } catch {
    // fall through to the server cwd
  }
  return process.cwd();
}
