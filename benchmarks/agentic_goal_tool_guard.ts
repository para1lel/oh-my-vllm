// The native goal builds a new CPU project in its private workspace.
import { resolve, sep } from "node:path";

export default function (api: any) {
  const root = resolve(process.cwd());
  const commands = new Set([
    "python -m unittest discover -s tests -q",
    "python -m compileall -q goaldb tests",
    "python -m goaldb.benchmark",
  ]);
  api.on("tool_call", (event: any) => {
    if (event.toolName === "goal" && !["get", "complete", "drop"].includes(event.input.op))
      return { block: true, reason: "The host controls goal creation and resumption." };
    if (event.toolName === "bash") {
      if (!commands.has(event.input.command?.trim()) ||
          (event.input.cwd && resolve(root, event.input.cwd) !== root))
        return { block: true, reason: "Run the documented CPU checks in this workspace." };
      return;
    }
    if (!["read", "grep", "write", "edit"].includes(event.toolName)) return;
    const argument = event.input.path;
    if (event.toolName === "read" && argument === "history://current/full") return;
    if (argument !== undefined && typeof argument !== "string")
      return { block: true, reason: "Use a path in the new goal workspace." };
    const filename = typeof argument === "string" ? argument.replace(/:\d+(?:-\d+)?$/, "") : ".";
    const target = resolve(root, filename);
    if (target !== root && !target.startsWith(root + sep))
      return { block: true, reason: "Tools must stay in the new goal workspace." };
    if (["write", "edit"].includes(event.toolName) && target === resolve(root, "TASK.md"))
      return { block: true, reason: "Keep the task specification unchanged." };
  });
  api.on("tool_result", (event: any) => ({
    content: event.content.map((item: any) => {
      if (item.type !== "text" || item.text.length <= 16000) return item;
      const artifacts = item.text.match(/artifact:\/\/[^\s\]]+/g) || [];
      return { ...item, text: item.text.slice(0, 16000) +
        "\n[Read a smaller range for complete evidence.]\n" + artifacts.join("\n") };
    }),
  }));
}
