# Document rules

Write all English documents for this project in [ASD-STE100 Issue 9](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf).
Use the approved words with their approved meanings and parts of speech.

Use each project [technical term](glossary.md) only with the meaning in the glossary.
Do not use general words as new technical terms.

Put each fact in one location. Use links from other locations.
README gives user information. CONTRIBUTING gives the contribution procedure.

The root AGENTS gives rules for all directories. A child AGENTS gives rules for its directory tree.
Add a child guide only when a directory has local rules.

Keep each English Markdown document at most 2000 English words.
Count the visible text, headings, table text, and inline identifiers.

Do not count fenced code or link destinations.
Keep procedure sentences at most 20 words. Give one action in each procedure sentence.

Keep descriptive sentences at most 25 words. Keep descriptive paragraphs at most 6 sentences.

Supply a full Chinese translation in a same-directory `<stem>.zh.md` file.
Keep facts, conditions, uncertainty, commands, identifiers, numbers, and heading levels the same.

Use the English document as the source. Use Chinese links when the related files exist.
Use halfwidth Chinese punctuation. Put spaces between Chinese text and English words or numbers.

Use source links for implementation details. Do not copy the implementation into prose.
Use a text anchor that occurs only once in the source file for each source line link.

The documentation hook updates the line numbers.
Put host paths and addresses in ignored local configuration.

Use the [nightly reporting policy](deepseek-v41/validation.md) for approved performance documents.
Do not add unrelated test results, raw logs, or delivery histories.

Get a document review from another agent. Examine the word meanings, parts of speech, and translation.
Use the official standard. The automated checks do not examine all STE rules.
Compare the Chinese text with the English text.
