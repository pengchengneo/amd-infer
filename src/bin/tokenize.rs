//! Native Rust tokenizer, reusing the already-downloaded official tokenizer JSON.
use std::{error::Error, fs};
fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tokenizers::parallelism::set_parallelism(false);
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 4 {
        return Err("usage: tokenize encode|decode TOKENIZER_JSON INPUT OUTPUT".into());
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&a[1])?;
    match a[0].as_str() {
        "encode" => {
            let text = fs::read_to_string(&a[2])?;
            let ids = tokenizer.encode(text.as_str(), false)?.get_ids().to_vec();
            fs::write(&a[3], format!("{ids:?}\n"))?;
            println!("encoded {} tokens", ids.len());
        }
        "decode" => {
            let raw = fs::read_to_string(&a[2])?;
            let ids: Result<Vec<u32>, _> = raw
                .trim()
                .trim_matches(['[', ']'])
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse())
                .collect();
            fs::write(&a[3], tokenizer.decode(&ids?, false)?)?;
        }
        _ => return Err("unknown tokenizer command".into()),
    }
    Ok(())
}
