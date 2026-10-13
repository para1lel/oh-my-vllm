use std::collections::BTreeSet;
use std::env;
use std::error::Error;
use std::fs;

use proc_macro2::Span;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{ImplItem, Item, Stmt, TraitItem};

struct DefinitionSpacing<'a> {
    lines: Vec<&'a str>,
    missing: BTreeSet<usize>,
}

impl DefinitionSpacing<'_> {
    fn inspect(&mut self, entries: impl Iterator<Item = (Span, bool)>) {
        let entries: Vec<_> = entries.collect();
        for pair in entries.windows(2) {
            let [(previous, left_definition), (current, right_definition)] = pair else {
                unreachable!()
            };
            if !left_definition && !right_definition {
                continue;
            }
            let end = previous.end().line;
            let mut start = current.start().line - 1;
            while start > end && self.lines[start - 1].trim_start().starts_with("//") {
                start -= 1;
            }
            if start < end
                || self.lines[end..start]
                    .iter()
                    .all(|line| !line.trim().is_empty())
            {
                self.missing.insert(start);
            }
        }
    }
}

fn is_definition(item: &Item) -> bool {
    matches!(
        item,
        Item::Fn(_) | Item::Struct(_) | Item::Enum(_) | Item::Trait(_) | Item::Impl(_)
    ) || matches!(item, Item::Mod(module) if module.content.is_some())
}

impl<'ast> Visit<'ast> for DefinitionSpacing<'_> {
    fn visit_file(&mut self, file: &'ast syn::File) {
        self.inspect(
            file.items
                .iter()
                .map(|item| (item.span(), is_definition(item))),
        );
        visit::visit_file(self, file);
    }

    fn visit_item_mod(&mut self, module: &'ast syn::ItemMod) {
        if let Some((_, items)) = &module.content {
            self.inspect(items.iter().map(|item| (item.span(), is_definition(item))));
        }
        visit::visit_item_mod(self, module);
    }

    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        self.inspect(
            item.items
                .iter()
                .map(|item| (item.span(), matches!(item, ImplItem::Fn(_)))),
        );
        visit::visit_item_impl(self, item);
    }

    fn visit_item_trait(&mut self, item: &'ast syn::ItemTrait) {
        self.inspect(
            item.items
                .iter()
                .map(|item| (item.span(), matches!(item, TraitItem::Fn(_)))),
        );
        visit::visit_item_trait(self, item);
    }

    fn visit_block(&mut self, block: &'ast syn::Block) {
        self.inspect(block.stmts.iter().map(|statement| {
            (
                statement.span(),
                matches!(statement, Stmt::Item(item) if is_definition(item)),
            )
        }));
        visit::visit_block(self, block);
    }
}

fn missing_blank_lines(source: &str) -> Result<BTreeSet<usize>, syn::Error> {
    let syntax = syn::parse_file(source)?;
    let mut spacing = DefinitionSpacing {
        lines: source.lines().collect(),
        missing: BTreeSet::new(),
    };
    spacing.visit_file(&syntax);
    Ok(spacing.missing)
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = env::args().skip(1).collect();
    let fix = arguments
        .first()
        .is_some_and(|argument| argument == "--fix");
    let mut failed = false;
    for path in &arguments[usize::from(fix)..] {
        let source = fs::read_to_string(path)?;
        let missing = missing_blank_lines(&source)?;
        if missing.is_empty() {
            continue;
        }
        if fix {
            let mut lines: Vec<_> = source.lines().map(String::from).collect();
            for line in missing.iter().rev() {
                lines.insert(*line, String::new());
            }
            fs::write(path, lines.join("\n") + "\n")?;
            let remaining = missing_blank_lines(&fs::read_to_string(path)?)?;
            if remaining.is_empty() {
                println!("{path}: definition spacing updated");
                continue;
            }
        }
        failed = true;
        for line in missing {
            println!(
                "{path}:{}: add a blank line before this definition",
                line + 1
            );
        }
    }
    if failed {
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::missing_blank_lines;

    #[test]
    fn attributes_comments_and_raw_strings_keep_their_definition() {
        let source = concat!(
            "fn first() { let text = r#\"fn fake() {}\"#; }\n",
            "/// Second operation.\n#[inline]\nfn second() {}\n"
        );
        assert_eq!(missing_blank_lines(source).unwrap(), [1].into());
    }

    #[test]
    fn module_imports_stay_grouped_and_nested_methods_need_spacing() {
        let source = concat!(
            "mod first;\nmod second;\n\nstruct State;\n\nimpl State {\n",
            "fn first() {}\nfn second() {}\n}\n"
        );
        assert_eq!(missing_blank_lines(source).unwrap(), [7].into());
    }
}
