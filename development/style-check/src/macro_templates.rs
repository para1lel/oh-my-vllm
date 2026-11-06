use proc_macro2::{Group, Ident, TokenStream, TokenTree};
use syn::parse::{Parse, ParseStream, Parser};
use syn::{Error, Token};

struct Rules {
    templates: Vec<TokenStream>,
}

impl Parse for Rules {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut templates = Vec::new();
        while !input.is_empty() {
            let matcher: TokenTree = input.parse()?;
            if !matches!(matcher, TokenTree::Group(_)) {
                return Err(Error::new(matcher.span(), "expected a macro matcher group"));
            }
            input.parse::<Token![=>]>()?;
            let expansion: TokenTree = input.parse()?;
            let TokenTree::Group(group) = expansion else {
                return Err(Error::new(
                    expansion.span(),
                    "expected a macro template group",
                ));
            };
            templates.push(normalize(group.stream())?);
            if !input.is_empty() {
                input.parse::<Token![;]>()?;
            }
        }
        Ok(Self { templates })
    }
}

fn is_punctuation(token: &TokenTree, value: char) -> bool {
    matches!(token, TokenTree::Punct(punctuation) if punctuation.as_char() == value)
}

fn repetition_end(tokens: &[TokenTree], group: usize) -> syn::Result<usize> {
    let first = group + 1;
    for (index, token) in tokens.iter().enumerate().skip(first).take(2) {
        if matches!(token, TokenTree::Punct(p) if matches!(p.as_char(), '*' | '+' | '?')) {
            return Ok(index + 1);
        }
    }
    Err(Error::new(
        tokens[group].span(),
        "unsupported macro repetition",
    ))
}

fn normalized_group(group: &Group) -> syn::Result<Group> {
    let mut normalized = Group::new(group.delimiter(), normalize(group.stream())?);
    normalized.set_span(group.span());
    Ok(normalized)
}

fn normalize(source: TokenStream) -> syn::Result<TokenStream> {
    let tokens: Vec<_> = source.into_iter().collect();
    let mut normalized = TokenStream::new();
    let mut index = 0;
    while index < tokens.len() {
        if is_punctuation(&tokens[index], '$') {
            match tokens.get(index + 1) {
                Some(TokenTree::Ident(name)) => {
                    normalized.extend([TokenTree::Ident(Ident::new(
                        &format!("__macro_{}", name.to_string().trim_start_matches("r#")),
                        name.span(),
                    ))]);
                    index += 2;
                }
                Some(TokenTree::Group(group)) => {
                    let end = repetition_end(&tokens, index + 1)?;
                    normalized.extend(normalize(group.stream())?);
                    index = end;
                }
                _ => {
                    return Err(Error::new(
                        tokens[index].span(),
                        "unsupported macro metavariable",
                    ));
                }
            }
        } else {
            let token = match &tokens[index] {
                TokenTree::Group(group) => TokenTree::Group(normalized_group(group)?),
                token => token.clone(),
            };
            normalized.extend([token]);
            index += 1;
        }
    }
    Ok(normalized)
}

pub fn templates(tokens: TokenStream) -> syn::Result<Vec<TokenStream>> {
    Ok(syn::parse2::<Rules>(tokens)?.templates)
}

pub fn prefix<T: Parse>(tokens: &[TokenTree]) -> syn::Result<(T, usize)> {
    let parser = |input: ParseStream<'_>| {
        let value: T = input.parse()?;
        let remainder: TokenStream = input.parse()?;
        Ok((value, remainder.into_iter().count()))
    };
    let (value, remainder) = parser.parse2(tokens.iter().cloned().collect())?;
    Ok((value, tokens.len() - remainder))
}

pub fn pattern_prefix(tokens: &[TokenTree]) -> syn::Result<(syn::Pat, usize)> {
    let parser = |input: ParseStream<'_>| {
        let value = syn::Pat::parse_multi_with_leading_vert(input)?;
        let remainder: TokenStream = input.parse()?;
        Ok((value, remainder.into_iter().count()))
    };
    let (value, remainder) = parser.parse2(tokens.iter().cloned().collect())?;
    Ok((value, tokens.len() - remainder))
}

pub fn has_definition(tokens: TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        TokenTree::Ident(name) => name == "fn",
        TokenTree::Punct(punctuation) => punctuation.as_char() == '|',
        TokenTree::Group(group) => has_definition(group.stream()),
        TokenTree::Literal(_) => false,
    })
}
