// Drives the translation tool guard clauses without the OMP runtime.
import guard from "../../benchmarks/agentic_translation_tool_guard.ts";

type GuardEvent = { toolName: string; input: Record<string, unknown> };
type Handler = (event: GuardEvent) => unknown;

const handlers: Record<string, Handler[]> = { tool_call: [], tool_result: [] };
const api = {
  on(name: "tool_call" | "tool_result", handler: Handler) {
    handlers[name].push(handler);
  },
};

guard(api);

const toolName = process.argv[2];
const input = JSON.parse(process.argv[3]);
process.stdout.write(
  JSON.stringify(handlers.tool_call[0]({ toolName, input }) ?? null),
);
