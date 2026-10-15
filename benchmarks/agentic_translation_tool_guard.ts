// The accepted soak translates one validated corpus inside its private workspace.
import { resolve, sep } from "node:path";

interface ToolCallEvent {
  toolName: string;
  input: { op?: unknown; path?: unknown };
}

interface ToolResultItem {
  type: string;
  text?: string;
}

interface ToolResultEvent {
  content: ToolResultItem[];
}

interface ToolGuard {
  on(
    name: "tool_call",
    handler: (event: ToolCallEvent) => { block: true; reason: string } | void,
  ): void;
  on(
    name: "tool_result",
    handler: (event: ToolResultEvent) => { content: ToolResultItem[] },
  ): void;
}

export default function (api: ToolGuard) {
  const root = resolve(process.cwd());
  const corpus = resolve(root, "corpus") + sep;
  const specification = resolve(root, "TASK.md");
  api.on("tool_call", (event) => {
    const op = event.input?.op;
    if (event.toolName === "goal") {
      if (typeof op === "string" && ["get", "complete", "drop"].includes(op)) return;
      return { block: true, reason: "The host controls goal creation and resumption." };
    }
    if (event.toolName === "edit") {
      // Pinned OMP edit payloads are hashline/ast operations ({i, input},
      // [PATH#TAG], REM, MV DEST), so a single path check cannot bound them.
      return {
        block: true,
        reason: "The translation workload writes whole files; use the write tool.",
      };
    }
    if (!["read", "grep", "write"].includes(event.toolName))
      return { block: true, reason: "Translate with the read, grep and write tools." };
    const argument = event.input?.path;
    if (event.toolName === "read" && argument === "history://current/full") return;
    if (argument !== undefined && typeof argument !== "string")
      return { block: true, reason: "Use a path in the translation workspace." };
    if (event.toolName === "write" && typeof argument !== "string")
      return { block: true, reason: "The write tool needs a workspace path." };
    const filename =
      typeof argument === "string" ? argument.replace(/:\d+(?:-\d+)?$/, "") : ".";
    const target = resolve(root, filename);
    if (target !== root && !target.startsWith(root + sep))
      return { block: true, reason: "Tools must stay in the translation workspace." };
    if (
      event.toolName === "write" &&
      (target === specification || target.startsWith(corpus))
    )
      return { block: true, reason: "The specification and corpus are read-only." };
  });
  api.on("tool_result", (event) => ({
    content: event.content.map((item) =>
      item.type === "text" && (item.text?.length ?? 0) > 16000
        ? {
            ...item,
            text:
              (item.text ?? "").slice(0, 16000) +
              "\n[Tool output capped. Read a smaller range.]",
          }
        : item,
    ),
  }));
}
