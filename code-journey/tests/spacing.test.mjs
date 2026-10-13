import assert from "node:assert/strict";
import test from "node:test";
import { missingDefinitionSpacing } from "../scripts/check-spacing.mjs";

test("source strings do not create definitions and attached comments stay attached", () => {
  const source = "function first() { return 'function fake() {}'; }\n// Next operation.\nexport function second() {}\n";
  const insertions = missingDefinitionSpacing(source);
  assert.equal(insertions.size, 1);
  assert.equal([...insertions.keys()][0], source.indexOf("// Next"));
});

test("imports stay grouped while adjacent class methods need a blank line", () => {
  const source = "import first from 'first';\nimport second from 'second';\n\nclass Worker {\nfirst() {}\nsecond() {}\n}\n";
  const insertions = missingDefinitionSpacing(source);
  assert.equal(insertions.size, 1);
  assert.equal([...insertions.keys()][0], source.indexOf("second()"));
});

test("function-valued class fields and comment interiors do not bypass spacing", () => {
  const source = "class Worker {\nfirst = () => {};\n/* Next\n\nmethod */\nsecond = () => {};\n}\n";
  const insertions = missingDefinitionSpacing(source);
  assert.equal(insertions.size, 1);
  assert.equal([...insertions.keys()][0], source.indexOf("/* Next"));
});

test("TypeScript stripping preserves definition source positions", () => {
  const source = "function first(value: number): number { return value; }\nfunction second(): void {}\n";
  assert.equal([...missingDefinitionSpacing(source, true).keys()][0], source.indexOf("function second"));
});
