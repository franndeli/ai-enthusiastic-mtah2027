// Data guard for pi: blocks any tool call that would bring gated match data into a cloud LLM context.
// Delegates every decision to scripts/agent_guard.py (shared with the Claude Code hooks); fails closed.
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const here = typeof __dirname !== "undefined" ? __dirname : path.dirname(fileURLToPath(import.meta.url));
const GUARD_SCRIPT = path.resolve(here, "../../scripts/agent_guard.py");

function guardReason(toolName: string, toolInput: unknown): string | undefined {
  try {
    execFileSync("python3", [GUARD_SCRIPT], {
      input: JSON.stringify({ tool_name: toolName, tool_input: toolInput ?? {} }),
      stdio: ["pipe", "ignore", "pipe"],
      timeout: 10_000,
    });
    return undefined;
  } catch (error: any) {
    const stderr = error?.stderr?.toString().trim();
    return stderr || `DATA GUARD: guard failed to run (${error?.message ?? error}); blocking as a precaution.`;
  }
}

export default function (pi: ExtensionAPI) {
  pi.on("session_start", async (_event, ctx) => {
    if (ctx.hasUI) ctx.ui.notify("MTAH data guard active: gated match data is blocked from agent tools.", "info");
  });

  pi.on("tool_call", async (event, ctx) => {
    const reason = guardReason(event.toolName, event.input);
    if (!reason) return undefined;
    if (ctx.hasUI) ctx.ui.notify(reason, "warning");
    return { block: true, reason };
  });
}
