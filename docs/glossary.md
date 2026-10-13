# Project technical glossary

[terms.json](terms.json) is the authoritative source for technical terms, meanings, translations, and registered forms.
Generate the following reference pages from that source:

- [Term reference A](glossary/model.md).
- [Term reference B](glossary/runtime.md).
- [Term reference C](glossary/gpu-service.md).

ASD-STE100 governs grammar and general words. A technical entry permits only its defined project meaning.
Code identifiers, proper product names, and protocol text keep their source forms.

## Maintenance

Add a term only when the approved dictionary cannot express its technical meaning accurately.
Supply a noun or verb, a definition, a Chinese translation, and explicit forms.
Review the term against source and Rule 1.5 or Rule 1.12.
Use general approved wording when it expresses the same action.

```bash
scripts/with-env.sh python scripts/build_glossary.py
scripts/with-env.sh python scripts/build_glossary.py --check
```
