import { test } from 'node:test';
import assert from 'node:assert/strict';
import { dedent } from '../scripts/excerpt.mjs';

test('Rust excerpt removes common spaces, retaining inner blocks and blank rows', () => {
  assert.equal(dedent('    fn f() {\n        if ready {\n            run();\n        }\n    }'),
    'fn f() {\n    if ready {\n        run();\n    }\n}');
});
test('Python excerpt retains the relative indentation required by its suite', () => {
  assert.equal(dedent('    def f():\n        if valid:\n            return 1\n    \n        return 0'),
    'def f():\n    if valid:\n        return 1\n\n    return 0');
});
test('tabs and mixed prefixes are removed literally, without changing tab semantics', () => {
  assert.equal(dedent('\t  first\n\t    second'), 'first\n  second');
  assert.equal(dedent(' \tfirst\n  second'), '\tfirst\n second');
});
test('top-level, leading blank and whitespace-only inputs remain safe', () => {
  assert.equal(dedent('\n    one\n      two\n'), '\none\n  two\n');
  assert.equal(dedent('one\n    two'), 'one\n    two');
  assert.equal(dedent('  \n\t'), '  \n\t');
});
