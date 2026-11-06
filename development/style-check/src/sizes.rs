use std::error::Error;
use std::fs;

use proc_macro2::{Delimiter, Group, Span, TokenStream, TokenTree};
use syn::spanned::Spanned;
use syn::visit::{self, Visit};

use crate::macro_templates;

#[derive(Default)]
struct Functions {
    entries: Vec<String>,
    errors: Vec<String>,
}

fn json_string(value: &str) -> String {
    let mut result = String::from("\"");
    for character in value.chars() {
        match character {
            '\"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            c if c.is_control() => result.push_str(&format!("\\u{:04x}", c as u32)),
            c => result.push(c),
        }
    }
    result.push('\"');
    result
}

impl Functions {
    fn visit_complete(&mut self, tokens: TokenStream) -> bool {
        if let Ok(file) = syn::parse2::<syn::File>(tokens.clone()) {
            self.visit_file(&file);
            return true;
        }
        if let Ok(expression) = syn::parse2::<syn::Expr>(tokens.clone()) {
            self.visit_expr(&expression);
            return true;
        }
        let wrapped = TokenStream::from(TokenTree::Group(Group::new(Delimiter::Brace, tokens)));
        if let Ok(block) = syn::parse2::<syn::Block>(wrapped) {
            self.visit_block(&block);
            return true;
        }
        false
    }

    fn visit_tokens(&mut self, tokens: TokenStream) {
        if self.visit_complete(tokens.clone()) {
            return;
        }
        let tokens: Vec<_> = tokens.into_iter().collect();
        let mut index = 0;
        while index < tokens.len() {
            let token = &tokens[index];
            if let Some(consumed) = self.visit_prefix(&tokens[index..]) {
                index += consumed;
                continue;
            }
            if macro_templates::has_definition(TokenStream::from(token.clone())) {
                self.error(syn::Error::new(
                    token.span(),
                    "cannot parse function or closure in macro arguments",
                ));
            }
            if let TokenTree::Group(group) = token {
                self.visit_tokens(group.stream());
            }
            index += 1;
        }
    }

    fn visit_prefix(&mut self, tokens: &[TokenTree]) -> Option<usize> {
        if let Ok((item, consumed)) = macro_templates::prefix::<syn::Item>(tokens) {
            self.visit_item(&item);
            return Some(consumed);
        }
        if let Ok((expression, consumed)) = macro_templates::prefix::<syn::Expr>(tokens) {
            self.visit_expr(&expression);
            return Some(consumed);
        }
        if let Ok((pattern, consumed)) = macro_templates::pattern_prefix(tokens) {
            self.visit_pat(&pattern);
            return Some(consumed);
        }
        None
    }

    fn error(&mut self, error: syn::Error) {
        self.errors.push(format!(
            "{{\"line\":{},\"message\":{}}}",
            error.span().start().line,
            json_string(&error.to_string())
        ));
    }

    fn visit_templates(&mut self, item: &syn::Macro) {
        match macro_templates::templates(item.tokens.clone()) {
            Ok(templates) => {
                for template in templates {
                    self.visit_tokens(template);
                }
            }
            Err(error) => self.error(error),
        }
    }

    fn record(&mut self, name: &str, kind: &str, span: Span) {
        let start = span.start().line;
        let end = span.end().line;
        self.entries.push(format!(
            "{{\"name\":{},\"kind\":{},\"start\":{start},\"end\":{end},\"lines\":{}}}",
            json_string(name),
            json_string(kind),
            end - start + 1
        ));
    }
}

impl<'ast> Visit<'ast> for Functions {
    fn visit_item(&mut self, item: &'ast syn::Item) {
        if let syn::Item::Verbatim(tokens) = item {
            if macro_templates::has_definition(tokens.clone()) {
                self.error(syn::Error::new_spanned(
                    item,
                    "unsupported function item syntax",
                ));
            }
        } else {
            visit::visit_item(self, item);
        }
    }

    fn visit_expr(&mut self, expression: &'ast syn::Expr) {
        if let syn::Expr::Verbatim(tokens) = expression {
            if macro_templates::has_definition(tokens.clone()) {
                self.error(syn::Error::new_spanned(
                    expression,
                    "unsupported closure syntax",
                ));
            }
        } else {
            visit::visit_expr(self, expression);
        }
    }

    fn visit_macro(&mut self, item: &'ast syn::Macro) {
        if item.path.is_ident("macro_rules") {
            self.visit_templates(item);
        } else {
            self.visit_tokens(item.tokens.clone());
        }
    }

    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.record(&item.sig.ident.to_string(), "function", item.span());
        visit::visit_item_fn(self, item);
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        self.record(&item.sig.ident.to_string(), "method", item.span());
        visit::visit_impl_item_fn(self, item);
    }

    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        if item.default.is_some() {
            self.record(&item.sig.ident.to_string(), "method", item.span());
        }
        visit::visit_trait_item_fn(self, item);
    }

    fn visit_expr_closure(&mut self, item: &'ast syn::ExprClosure) {
        self.record("<closure>", "closure", item.span());
        visit::visit_expr_closure(self, item);
    }
}

pub fn report(paths: &[String]) -> Result<(), Box<dyn Error>> {
    let mut reports = Vec::new();
    for path in paths {
        let source = fs::read_to_string(path)?;
        let mut functions = Functions::default();
        let (entries, errors) = match syn::parse_file(&source) {
            Ok(syntax) => {
                functions.visit_file(&syntax);
                (functions.entries.join(","), functions.errors.join(","))
            }
            Err(error) => (
                String::new(),
                format!(
                    "{{\"line\":{},\"message\":{}}}",
                    error.span().start().line,
                    json_string(&error.to_string())
                ),
            ),
        };
        reports.push(format!(
            "{{\"path\":{},\"functions\":[{entries}],\"errors\":[{errors}]}}",
            json_string(path)
        ));
    }
    println!("[{}]", reports.join(","));
    Ok(())
}
