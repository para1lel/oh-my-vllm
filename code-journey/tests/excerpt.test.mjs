import { test } from 'node:test';
import assert from 'node:assert/strict';
import { dedent, extractExcerpt } from '../scripts/excerpt.mjs';

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

test('a Rust field retains its complete documentation and original starting line', () => {
  const source = 'struct Request {\n    pub prompt_len: usize,\n    /// Prompt plus accepted output.\n    ///\n    /// Drafts live in `draft_token_ids`.\n    pub token_ids: Vec<u32>,\n    /// Pending drafts.\n    pub draft_token_ids: Vec<u32>,\n}';
  assert.deepEqual(extractExcerpt(source, {
    begin: '    pub token_ids:', end: '    /// Pending drafts.', language: 'rust',
  }), {
    line: 3,
    originalText: '    /// Prompt plus accepted output.\n    ///\n    /// Drafts live in `draft_token_ids`.\n    pub token_ids: Vec<u32>,',
    text: '/// Prompt plus accepted output.\n///\n/// Drafts live in `draft_token_ids`.\npub token_ids: Vec<u32>,',
  });
});

test('Rust declaration keeps block documentation and multiline attributes together', () => {
  const source = '// Unrelated section.\n\n/** One scheduled request.\n * With its execution metadata.\n */\n#[derive(\n    Debug,\n    Clone\n)]\npub struct ScheduledRequest {\n    pub id: u64,\n}';
  const excerpt = extractExcerpt(source, { begin: 'pub struct ScheduledRequest', language: 'rust' });
  assert.equal(excerpt.line, 3);
  assert.equal(excerpt.text, source.split('\n').slice(2).join('\n'));
});

test('Python declaration keeps decorators and its body docstring', () => {
  const source = 'class Worker:\n    @instrument(\n        "execute",\n    )\n    def execute(self):\n        """Execute one batch.\n\n        Return one result for each request.\n        """\n        return self.results\n';
  const excerpt = extractExcerpt(source, { begin: '    def execute(', language: 'python' });
  assert.equal(excerpt.line, 2);
  assert.equal(excerpt.text, '@instrument(\n    "execute",\n)\ndef execute(self):\n    """Execute one batch.\n\n    Return one result for each request.\n    """\n    return self.results');
});

test('metadata respects neighboring code and missing anchors fail clearly', () => {
  const source = '/// Previous element.\npub struct Previous;\n\npub struct Current;';
  assert.equal(extractExcerpt(source, { begin: 'pub struct Current', language: 'rust' }).text, 'pub struct Current;');
  assert.throws(() => extractExcerpt(source, { begin: 'absent', language: 'rust' }), /Missing source anchor/);
  assert.throws(() => extractExcerpt(source, { begin: 'pub struct Current', end: 'absent', language: 'rust' }), /Missing end anchor/);
});

test('Rust documentation retains blank paragraphs and same-indent body text', () => {
  const source = '/**\nRequest history.\n\nAccepted only.\n*/\npub struct Request;';
  const excerpt = extractExcerpt(source, { begin: 'pub struct Request', language: 'rust' });
  assert.equal(excerpt.line, 1);
  assert.equal(excerpt.text, source);
});

test('multiline metadata keeps internal blanks and quoted delimiters without absorbing adjacent code', () => {
  for (const [language, begin, source] of [
    ['python', 'def execute', '@instrument(\n    "execute [)]",\n\n    capture=True,\n)\ndef execute():\n    return 1'],
    ['rust', 'pub struct Request', '#[cfg(\n    feature = ")]",\n\n)]\npub struct Request;'],
  ]) {
    const excerpt = extractExcerpt(source, { begin, language });
    assert.equal(excerpt.line, 1);
    assert.equal(excerpt.text, source);
  }
  const source = '@instrument(\n    "previous",\n)\ndef previous():\n    consume(\n        1,\n    )\ndef current():\n    return 2';
  assert.equal(extractExcerpt(source, { begin: 'def current', language: 'python' }).line, 8);
});
