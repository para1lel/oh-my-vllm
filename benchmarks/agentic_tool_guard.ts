// Explicitly loaded by the isolated soak clients. No extension discovery.
import { resolve, sep } from "node:path";
import { statSync, readFileSync } from "node:fs";

export default function (api: any) {
  const root = resolve(process.cwd());
  const notes = resolve(root, "notes") + sep;
  api.on("tool_call", (event: any) => {
    const argument = event.input.path;
    if (event.toolName === "read" && argument === "history://current/full") return;
    if (argument !== undefined && typeof argument !== "string")
      return { block: true, reason: "Use a source path in the audit workspace." };
    const selector = typeof argument === "string" ? argument.match(/:(\d+)(?:-(\d+))?$/) : null;
    const filename = typeof argument === "string" ? argument.replace(/:\d+(?:-\d+)?$/, "") : ".";
    const target = resolve(root, filename);
    if (target !== root && !target.startsWith(root + sep))
      return { block: true, reason: "Tools must stay inside the copied audit workspace." };
    if (event.toolName === "write" && !target.startsWith(notes))
      return { block: true, reason: "Write progress notes only under notes/." };
    if (event.toolName === "read") {
      if (selector && Number(selector[2] || selector[1]) - Number(selector[1]) >= 180)
        return { block: true, reason: "Read at most 180 lines at a time." };
      const limit = event.input.limit;
      if (limit !== undefined && (!Number.isInteger(limit) || limit < 1 || limit > 180))
        return { block: true, reason: "Read at most 180 lines at a time." };
      if (!selector && limit === undefined) {
        try {
          if (statSync(target).isFile() && readFileSync(target, "utf8").split("\n").length > 180)
            return { block: true, reason: "Use path:START-END or read offset/limit for a bounded range." };
        } catch {
          // The native read tool reports missing paths as ordinary tool errors.
        }
      }
    }
    if (event.toolName === "find" && typeof event.input.pattern === "string" &&
        (event.input.pattern.startsWith("/") || event.input.pattern.split("/").includes("..")))
      return { block: true, reason: "Find source files within this workspace." };
  });
  api.on("tool_result", (event: any) => ({
    content: event.content.map((item: any) => item.type === "text" && item.text.length > 16000
      ? { ...item, text: item.text.slice(0, 16000) + "\n[Tool output capped. Read a smaller range.]" }
      : item),
  }));
}
