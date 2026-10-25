//! Tokenizer: vocabulary, encoding, decoding and chat-template tokens.
//!
//! Three encoders, chosen by `tokenizer.ggml.model`:
//! - "llama": SentencePiece BPE driven by per-token scores (Gemma 3, Llama 2),
//!   with `<0xHH>` byte fallback;
//! - "gpt2": byte-level BPE driven by merge ranks (Llama 3, Qwen), with a
//!   pre-tokenizer that splits words, numbers and punctuation the way the
//!   llama-bpe regex does (Qwen's splits digits singly);
//! - other: BPE over ▁-for-space codepoints driven by merge ranks (Gemma 4).
//!
//! Decoding renders byte tokens as their byte and control tokens as nothing,
//! so streamed text never shows "<0x0A>" or "<end_of_turn>".

mod unicode;

use self::unicode::{CpClass, classify};
use crate::error::{Result, bail};
use crate::gguf::{GgufFile, GgufType, GgufValue};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::ops::Range;

// GGUF tokenizer.ggml.token_type values.
pub(crate) const TOKEN_NORMAL: i32 = 1;
pub(crate) const TOKEN_UNKNOWN: i32 = 2;
pub(crate) const TOKEN_CONTROL: i32 = 3;
pub(crate) const TOKEN_USER_DEFINED: i32 = 4;
pub(crate) const TOKEN_UNUSED: i32 = 5;
pub(crate) const TOKEN_BYTE: i32 = 6;

/// The encoder, from `tokenizer.ggml.model`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VocabKind {
    /// "llama": SentencePiece scores.
    Spm,
    /// "gpt2": byte-level BPE.
    Gpt2,
    /// ▁-for-space BPE by merge rank.
    Bpe,
}

/// How a byte-level BPE vocabulary cuts text before merging
/// (`tokenizer.ggml.pre`). The regexes differ in their digit runs and in
/// whether combining marks join the letters around them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VocabPre {
    /// "llama-bpe" and anything unrecognised: up to three digits per piece.
    Llama3,
    /// "qwen2": one digit per piece.
    Qwen2,
    /// "qwen35": one digit per piece, and combining marks count as letters.
    Qwen35,
}

/// Which turn format a chat session writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChatFormat {
    /// Unknown: user text goes in as it is.
    None,
    /// `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`
    Gemma,
    /// `<|turn>user\n...<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`
    Gemma4,
    /// `<|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>`
    Llama3,
    /// `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n` (Qwen)
    ChatMl,
}

/// A model's vocabulary: how to encode and decode text, and the tokens its
/// chat format is written with. Token ids are `i32`; -1 means none.
#[derive(Clone, Debug)]
pub(crate) struct Vocab {
    pub kind: VocabKind,
    pub pre: VocabPre,
    /// `tokenizer.ggml.model`, as written in the file.
    pub model: String,

    /// As the file spells them ("▁Paris", "<0x0A>").
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    /// The `TOKEN_*` type of each token (may be empty).
    pub types: Vec<i32>,
    /// The bytes each token renders to; not always whole UTF-8 sequences.
    pub pieces: Vec<Vec<u8>>,

    pub token_to_id: HashMap<String, i32>,
    /// "lhs rhs" -> rank.
    pub merge_rank: HashMap<String, i32>,

    pub bos_id: i32,
    pub eos_id: i32,
    pub unk_id: i32,
    pub pad_id: i32,
    pub add_bos: bool,
    /// End every sequence with EOS (embedding models).
    pub add_eos: bool,
    pub add_space_prefix: bool,
    /// The `<0xHH>` token of each byte, or -1.
    pub byte_ids: [i32; 256],

    /// End of generation: EOS and end-of-turn tokens.
    pub eog_ids: Vec<i32>,

    /// User-defined tokens (function-call tags, runs of spaces or newlines):
    /// always one token wherever their text appears in the input. Sorted by
    /// first byte, then longest first; those starting with byte `b` are
    /// `user_defined[user_defined_at[b]..user_defined_at[b + 1]]`.
    pub user_defined: Vec<(String, i32)>,
    pub user_defined_at: [u32; 257],

    pub chat: ChatFormat,
    /// `<start_of_turn>` | `<|turn>` | `<|start_header_id|>` | `<|im_start|>`
    pub tok_turn_start: i32,
    /// `<end_of_turn>` | `<turn|>` | `<|eot_id|>` | `<|im_end|>`
    pub tok_turn_end: i32,
    /// `<|end_header_id|>` (Llama 3)
    pub tok_header_end: i32,
    /// "\n" (Gemma)
    pub tok_newline: i32,
    /// `<|channel>` (Gemma 4): opens the thought channel.
    pub tok_channel_start: i32,
    /// `<channel|>`
    pub tok_channel_end: i32,
    /// The file's chat template opens each reply with an empty thought
    /// channel (Gemma 4 26B/31B with thinking off; E2B/E4B do not).
    pub empty_thought: bool,
    /// FunctionGemma: the system prompt (with the function declarations) is
    /// a turn of its own, under the "developer" role.
    pub developer_turn: bool,
    /// ChatML, from the file's template: the system prompt it writes when
    /// none is given (Qwen2.5's), and how it opens each reply - with an empty
    /// `<think>` block when thinking can be turned off (Qwen3, 3.5, 3.8), or
    /// with `"<think>\n"` for models that always think.
    pub default_system: String,
    pub reply_open: String,
    /// Thinking. `can_think`: the session's thinking switch lets the model
    /// reason before it answers (Gemma 4, whose template has `<|think|>`;
    /// ChatML templates with an `enable_thinking` switch). `always_thinks`:
    /// it reasons whatever it is told. With thinking on, a ChatML reply opens
    /// with `reply_open_think` (`"<think>\n"`, or nothing for Qwen3, which
    /// opens the block itself), and the system turn starts with
    /// `think_instructions` (Qwen3.8's default reasoning effort); Gemma 4's
    /// system turn starts with `tok_think`.
    pub can_think: bool,
    pub always_thinks: bool,
    pub reply_open_think: String,
    pub think_instructions: String,
    /// `<|think|>` (Gemma 4)
    pub tok_think: i32,
    /// The markers around the model's reasoning, which tell it apart from the
    /// answer: `<think>` and `</think>`, or Gemma 4's thought channel, whose
    /// first line is the channel's name (`thought_label`). -1 without them.
    pub tok_thought_open: i32,
    pub tok_thought_close: i32,
    pub thought_label: bool,
    /// TranslateGemma: language code ("en", "pt-BR") -> English name, from
    /// the chat template. Empty for other models.
    pub languages: HashMap<String, String>,
}

impl Default for Vocab {
    fn default() -> Self {
        Vocab {
            kind: VocabKind::Spm,
            pre: VocabPre::Llama3,
            model: String::new(),
            tokens: Vec::new(),
            scores: Vec::new(),
            types: Vec::new(),
            pieces: Vec::new(),
            token_to_id: HashMap::new(),
            merge_rank: HashMap::new(),
            bos_id: -1,
            eos_id: -1,
            unk_id: -1,
            pad_id: -1,
            add_bos: true,
            add_eos: false,
            add_space_prefix: false,
            byte_ids: [-1; 256],
            eog_ids: Vec::new(),
            user_defined: Vec::new(),
            user_defined_at: [0; 257],
            chat: ChatFormat::None,
            tok_turn_start: -1,
            tok_turn_end: -1,
            tok_header_end: -1,
            tok_newline: -1,
            tok_channel_start: -1,
            tok_channel_end: -1,
            empty_thought: false,
            developer_turn: false,
            default_system: String::new(),
            reply_open: String::new(),
            can_think: false,
            always_thinks: false,
            reply_open_think: String::new(),
            think_instructions: String::new(),
            tok_think: -1,
            tok_thought_open: -1,
            tok_thought_close: -1,
            thought_label: false,
            languages: HashMap::new(),
        }
    }
}

impl Vocab {
    /// Read the `tokenizer.*` metadata.
    pub fn load(f: &GgufFile) -> Result<Vocab> {
        let tokens = match f.get("tokenizer.ggml.tokens").and_then(GgufValue::as_array) {
            Some((GgufType::String, tokens)) if !tokens.is_empty() => tokens,
            _ => bail!(UnsupportedModel, "the file has no tokenizer vocabulary"),
        };
        let n = tokens.len();
        if n > i32::MAX as usize {
            bail!(BadFormat, "the vocabulary is implausibly large");
        }

        let mut v = Vocab {
            tokens: tokens
                .iter()
                .map(|t| t.as_str().unwrap_or_default().to_owned())
                .collect(),
            ..Vocab::default()
        };
        if let Some((_, scores)) = f.get("tokenizer.ggml.scores").and_then(GgufValue::as_array)
            && scores.len() == n
        {
            v.scores = scores.iter().map(|s| s.as_float().unwrap_or(0.0) as f32).collect();
        }
        if let Some((_, types)) = f.get("tokenizer.ggml.token_type").and_then(GgufValue::as_array)
            && types.len() == n
        {
            v.types = types.iter().map(signed_or_zero).collect();
        }

        let special = |key: &str| kv_int(f, key).filter(|&id| id >= 0 && (id as usize) < n).unwrap_or(-1);
        v.bos_id = special("tokenizer.ggml.bos_token_id");
        v.eos_id = special("tokenizer.ggml.eos_token_id");
        v.unk_id = special("tokenizer.ggml.unknown_token_id");
        v.pad_id = special("tokenizer.ggml.padding_token_id");

        if let Some(model) = f.get_str("tokenizer.ggml.model") {
            v.model = model.to_owned();
        }

        if let Some((GgufType::String, merges)) = f.get("tokenizer.ggml.merges").and_then(GgufValue::as_array) {
            v.merge_rank.reserve(merges.len());
            for (i, merge) in merges.iter().enumerate() {
                v.merge_rank
                    .entry(merge.as_str().unwrap_or_default().to_owned())
                    .or_insert(i as i32);
            }
        }

        if v.model == "gpt2" {
            if v.merge_rank.is_empty() {
                bail!(UnsupportedModel, "the byte-level BPE tokenizer has no merges");
            }
            v.kind = VocabKind::Gpt2;
            match f.get_str("tokenizer.ggml.pre") {
                Some("qwen2") => v.pre = VocabPre::Qwen2,
                Some("qwen35") => v.pre = VocabPre::Qwen35,
                _ => {}
            }
        } else {
            v.kind = if v.merge_rank.is_empty() {
                VocabKind::Spm
            } else {
                VocabKind::Bpe
            };
        }

        // SentencePiece Llama models prepend a space unless the file says
        // otherwise; Gemma files always say.
        let arch = f.get_str("general.architecture").unwrap_or_default();
        v.add_space_prefix =
            kv_bool(f, "tokenizer.ggml.add_space_prefix").unwrap_or(v.kind == VocabKind::Spm && arch == "llama");
        v.add_bos = kv_bool(f, "tokenizer.ggml.add_bos_token").unwrap_or(v.add_bos);
        v.add_eos = kv_bool(f, "tokenizer.ggml.add_eos_token").unwrap_or(v.add_eos);

        v.token_to_id.reserve(n);
        for (i, token) in v.tokens.iter().enumerate() {
            v.token_to_id.entry(token.clone()).or_insert(i as i32);
        }

        if !v.types.is_empty() {
            v.user_defined = v
                .tokens
                .iter()
                .zip(&v.types)
                .enumerate()
                .filter(|&(_, (token, &ty))| ty == TOKEN_USER_DEFINED && !token.is_empty())
                .map(|(i, (token, _))| (token.clone(), i as i32))
                .collect();
            v.user_defined.sort_by(|(a, a_id), (b, b_id)| {
                a.as_bytes()[0]
                    .cmp(&b.as_bytes()[0])
                    .then(b.len().cmp(&a.len()))
                    .then(a_id.cmp(b_id))
            });
        }
        let mut j = 0;
        for b in 0..256 {
            v.user_defined_at[b] = j as u32;
            while j < v.user_defined.len() && v.user_defined[j].0.as_bytes()[0] as usize == b {
                j += 1;
            }
        }
        v.user_defined_at[256] = j as u32;

        for (i, token) in v.tokens.iter().enumerate() {
            if let Some(b) = parse_byte_token(token)
                && (v.types.is_empty() || v.types[i] == TOKEN_BYTE)
                && v.byte_ids[b as usize] < 0
            {
                v.byte_ids[b as usize] = i as i32;
            }
        }

        v.pieces = (0..n).map(|i| v.render_piece(i)).collect();

        // Chat turn format, recognised by its control tokens. Gemma 4 spells
        // Gemma's turn markers "<|turn>" and "<turn|>" and adds a system role
        // and a thought channel.
        let sot = v.find_token(&["<start_of_turn>", "<|turn>"]);
        let eot = v.find_token(&["<end_of_turn>", "<turn|>"]);
        let nl = v.find_token(&["\n"]);
        let sh = v.find_token(&["<|start_header_id|>"]);
        let eh = v.find_token(&["<|end_header_id|>"]);
        let eot_id = v.find_token(&["<|eot_id|>"]);
        let im_start = v.find_token(&["<|im_start|>"]);
        let im_end = v.find_token(&["<|im_end|>"]);
        let tpl = f.get_str("tokenizer.chat_template");
        if sot >= 0 && eot >= 0 && nl >= 0 {
            v.chat = if v.tokens[sot as usize] == "<|turn>" {
                ChatFormat::Gemma4
            } else {
                ChatFormat::Gemma
            };
            v.tok_turn_start = sot;
            v.tok_turn_end = eot;
            v.tok_newline = nl;
            v.tok_channel_start = v.find_token(&["<|channel>"]);
            v.tok_channel_end = v.find_token(&["<channel|>"]);
            v.developer_turn = v.chat == ChatFormat::Gemma && v.find_token(&["<start_function_declaration>"]) >= 0;
            // The template is Jinja source, so its newline is the two characters "\n".
            v.empty_thought = v.tok_channel_start >= 0
                && v.tok_channel_end >= 0
                && tpl.is_some_and(|t| t.contains("<|channel>thought\\n<channel|>"));
            if v.chat == ChatFormat::Gemma4 && v.tok_channel_start >= 0 && v.tok_channel_end >= 0 {
                v.tok_thought_open = v.tok_channel_start;
                v.tok_thought_close = v.tok_channel_end;
                v.thought_label = true;
                v.tok_think = v.find_token(&["<|think|>"]);
                v.can_think = v.tok_think >= 0 && tpl.is_some_and(|t| t.contains("<|think|>"));
            }
            if let Some(tpl) = tpl
                && tpl.contains("source_lang_code")
            {
                parse_languages(tpl, &mut v.languages);
            }
        } else if sh >= 0 && eh >= 0 && eot_id >= 0 {
            v.chat = ChatFormat::Llama3;
            v.tok_turn_start = sh;
            v.tok_header_end = eh;
            v.tok_turn_end = eot_id;
        } else if im_start >= 0 && im_end >= 0 {
            v.chat = ChatFormat::ChatMl;
            v.tok_turn_start = im_start;
            v.tok_turn_end = im_end;
            let think_open = v.find_token(&["<think>"]);
            let think_close = v.find_token(&["</think>"]);
            if think_open >= 0 && think_close >= 0 {
                v.tok_thought_open = think_open;
                v.tok_thought_close = think_close;
            }
            if let Some(tpl) = tpl {
                parse_chatml_template(tpl, &mut v);
            }
        }

        // End of generation: EOS, the turn terminator, and the other common
        // end markers when the file marks them as control tokens.
        v.add_eog(v.eos_id);
        v.add_eog(v.tok_turn_end);
        for name in [
            "<end_of_turn>",
            "<turn|>",
            "<|eot_id|>",
            "<|eom_id|>",
            "<|end_of_text|>",
            "<|endoftext|>",
            "<|im_end|>",
            "<eos>",
            "</s>",
        ] {
            if let Some(&id) = v.token_to_id.get(name)
                && (v.types.is_empty() || v.types[id as usize] == TOKEN_CONTROL)
            {
                v.add_eog(id);
            }
        }
        // FunctionGemma ends its turn early after a function call, where the
        // caller is to supply the function's response.
        if v.developer_turn {
            v.add_eog(v.find_token(&["<start_function_response>"]));
        }
        Ok(v)
    }

    pub fn n_vocab(&self) -> usize {
        self.tokens.len()
    }

    pub fn valid(&self, id: i32) -> bool {
        id >= 0 && (id as usize) < self.n_vocab()
    }

    /// Encode UTF-8 text. Control-token spellings in `text` stay plain text;
    /// user-defined tokens are matched whole, as llama.cpp and Hugging Face do.
    pub fn tokenize(&self, text: &str, add_bos: bool) -> Vec<i32> {
        let mut out = Vec::new();
        if add_bos && self.bos_id >= 0 {
            out.push(self.bos_id);
        }

        // Split the text at user-defined tokens, leftmost and then longest
        // first; the pieces between them are encoded on their own (each with
        // the leading space, if the vocabulary adds one).
        let bytes = text.as_bytes();
        let mut start = 0;
        let mut i = 0;
        while i < bytes.len() {
            let b = bytes[i] as usize;
            let candidates = &self.user_defined[self.user_defined_at[b] as usize..self.user_defined_at[b + 1] as usize];
            let Some((s, id)) = candidates.iter().find(|(s, _)| bytes[i..].starts_with(s.as_bytes())) else {
                i += 1;
                continue;
            };
            if i > start {
                self.encode_fragment(&text[start..i], self.add_space_prefix, &mut out);
            }
            out.push(*id);
            i += s.len();
            start = i;
        }
        if start < text.len() {
            self.encode_fragment(&text[start..], self.add_space_prefix, &mut out);
        }
        out
    }

    /// How a byte-level BPE vocabulary's pre-tokenizer cuts `text` (which
    /// holds no user-defined token): the pieces whose bytes are merged
    /// separately.
    #[cfg(test)]
    pub fn pretokenize(&self, text: &str) -> Vec<String> {
        self.pretokenize_pieces(text).into_iter().map(str::to_owned).collect()
    }

    /// The bytes `token` renders to (none for control tokens and invalid ids).
    pub fn piece(&self, token: i32) -> &[u8] {
        if self.valid(token) {
            &self.pieces[token as usize]
        } else {
            &[]
        }
    }

    pub fn detokenize(&self, tokens: &[i32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &token in tokens {
            out.extend_from_slice(self.piece(token));
        }
        out
    }

    pub fn is_eog(&self, token: i32) -> bool {
        token >= 0 && self.eog_ids.contains(&token)
    }

    /// The id of the first of `names` the vocabulary has, or -1.
    fn find_token(&self, names: &[&str]) -> i32 {
        names
            .iter()
            .find_map(|&name| self.token_to_id.get(name).copied())
            .unwrap_or(-1)
    }

    fn add_eog(&mut self, id: i32) {
        if id >= 0 && !self.eog_ids.contains(&id) {
            self.eog_ids.push(id);
        }
    }

    /// The bytes token `id` renders to. Without token types, the special ids
    /// count as control tokens and `<0xHH>` as byte tokens.
    fn render_piece(&self, id: usize) -> Vec<u8> {
        let token = &self.tokens[id];
        let ty = if !self.types.is_empty() {
            self.types[id]
        } else if [self.bos_id, self.eos_id, self.pad_id, self.unk_id].contains(&(id as i32)) {
            TOKEN_CONTROL
        } else if parse_byte_token(token).is_some() {
            TOKEN_BYTE
        } else {
            TOKEN_NORMAL
        };
        let gpt2 = self.kind == VocabKind::Gpt2;
        match ty {
            TOKEN_CONTROL | TOKEN_UNUSED | TOKEN_UNKNOWN => Vec::new(),
            TOKEN_BYTE => parse_byte_token(token).map_or_else(Vec::new, |b| vec![b]),
            TOKEN_USER_DEFINED if gpt2 => token.as_bytes().to_vec(),
            _ if gpt2 => unicode_to_bytes(token),
            _ => token.replace(SPACE_MARK, " ").into_bytes(),
        }
    }

    fn pretokenize_pieces<'t>(&self, text: &'t str) -> Vec<&'t str> {
        let max_digits = if self.pre == VocabPre::Llama3 { 3 } else { 1 };
        pretokenize_bpe(text, max_digits, self.pre == VocabPre::Qwen35)
    }

    /// Encode text that holds no user-defined token. `space_prefix` is the
    /// SentencePiece leading space, which llama.cpp gives the start of the
    /// text and the text after each user-defined token.
    fn encode_fragment(&self, text: &str, space_prefix: bool, out: &mut Vec<i32>) {
        let mut key = String::new();
        let mut by_rank = |pair: &str, split: usize| {
            key.clear();
            key.push_str(&pair[..split]);
            key.push(' ');
            key.push_str(&pair[split..]);
            self.merge_rank.get(&key).map(|&rank| -f64::from(rank))
        };

        if self.kind == VocabKind::Gpt2 {
            let mut mapped = String::new();
            let mut starts = Vec::new();
            for chunk in self.pretokenize_pieces(text) {
                mapped.clear();
                starts.clear();
                for &b in chunk.as_bytes() {
                    starts.push(mapped.len());
                    mapped.push(BYTE_TO_UNICODE[b as usize]);
                }
                let symbols = merge_symbols(&mapped, &starts, &mut by_rank);
                self.emit_symbols(&mapped, &symbols, out);
            }
            return;
        }

        let prefixed = if space_prefix {
            format!(" {text}")
        } else {
            text.to_owned()
        };
        let marked = prefixed.replace(' ', SPACE_MARK);
        let starts: Vec<usize> = marked.char_indices().map(|(i, _)| i).collect();
        let symbols = if self.kind == VocabKind::Spm {
            // SentencePiece: merge the pair whose result is the highest-scoring
            // vocabulary entry.
            merge_symbols(&marked, &starts, |pair: &str, _| {
                let id = *self.token_to_id.get(pair)?;
                Some(self.scores.get(id as usize).map_or(0.0, |&score| f64::from(score)))
            })
        } else {
            merge_symbols(&marked, &starts, &mut by_rank)
        };
        self.emit_symbols(&marked, &symbols, out);
    }

    /// Map final symbols to ids: whole pieces where the vocabulary has them,
    /// otherwise one `<0xHH>` token per byte, otherwise `<unk>`.
    fn emit_symbols(&self, text: &str, symbols: &[Range<usize>], out: &mut Vec<i32>) {
        for symbol in symbols {
            let s = &text[symbol.clone()];
            if let Some(&id) = self.token_to_id.get(s) {
                out.push(id);
                continue;
            }
            for &b in s.as_bytes() {
                if self.byte_ids[b as usize] >= 0 {
                    out.push(self.byte_ids[b as usize]);
                } else if self.unk_id >= 0 {
                    out.push(self.unk_id);
                }
            }
        }
    }
}

/// ▁ (U+2581), SentencePiece's space.
const SPACE_MARK: &str = "\u{2581}";

/// GPT-2's byte-to-unicode mapping, which Llama 3's and Qwen's byte-level
/// vocabularies share: each byte is spelled as a printable codepoint. The
/// printable bytes ('!'..'~', '¡'..'¬', '®'..'ÿ') spell themselves and the
/// others, in order, U+0100 onwards.
const BYTE_TO_UNICODE: [char; 256] = {
    let mut table = ['\0'; 256];
    let mut next = 0x100;
    let mut b: u32 = 0;
    while b < 256 {
        let cp = if matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF) {
            b
        } else {
            next += 1;
            next - 1
        };
        table[b as usize] = match char::from_u32(cp) {
            Some(c) => c,
            None => panic!("not a codepoint"),
        };
        b += 1;
    }
    table
};

/// The byte each codepoint of `BYTE_TO_UNICODE` spells.
const UNICODE_TO_BYTE: [Option<u8>; 0x144] = {
    let mut table = [None; 0x144];
    let mut b = 0;
    while b < 256 {
        table[BYTE_TO_UNICODE[b] as usize] = Some(b as u8);
        b += 1;
    }
    table
};

/// The inverse of `BYTE_TO_UNICODE` over a whole token: each mapped
/// codepoint becomes its byte, anything else passes through.
fn unicode_to_bytes(token: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(token.len());
    for c in token.chars() {
        match UNICODE_TO_BYTE.get(c as usize) {
            Some(&Some(b)) => out.push(b),
            _ => out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()),
        }
    }
    out
}

/// "<0xHH>" -> HH.
fn parse_byte_token(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?.as_bytes();
    if hex.len() != 2 {
        return None;
    }
    let digit = |c: u8| char::from(c).to_digit(16);
    Some((digit(hex[0])? * 16 + digit(hex[1])?) as u8)
}

/// A pair of neighbouring symbols that may merge, as it was when queued.
struct Candidate {
    priority: f64,
    left: usize,
    rev_left: i32,
    rev_right: i32,
}

impl Ord for Candidate {
    /// The higher priority first, and between equals the pair further left.
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .partial_cmp(&other.priority)
            .unwrap_or(Ordering::Equal)
            .then(other.left.cmp(&self.left))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

/// No symbol: the ends of the list.
const NONE: usize = usize::MAX;

/// Repeatedly merge the highest-priority adjacent pair of symbols until no
/// pair qualifies; ties go to the leftmost pair. The symbols are the pieces of
/// `text` that begin at `starts`. `priority(pair, split)` is given the text of
/// two neighbours, the left one ending at `split`, and returns `None` when
/// they may not merge. A priority queue with stale-entry checks keeps this
/// O(n log n) rather than rescanning per merge. Returns the merged symbols.
fn merge_symbols(
    text: &str,
    starts: &[usize],
    mut priority: impl FnMut(&str, usize) -> Option<f64>,
) -> Vec<Range<usize>> {
    let n = starts.len();
    if n == 0 {
        return Vec::new();
    }
    let mut prev: Vec<usize> = (0..n).map(|i| if i == 0 { NONE } else { i - 1 }).collect();
    let mut next: Vec<usize> = (1..=n).map(|i| if i == n { NONE } else { i }).collect();
    // Bumped when a symbol takes in its right neighbour; -1 once taken in.
    let mut rev = vec![0i32; n];
    // A symbol runs to the start of the next.
    let end = |next: &[usize], i: usize| if next[i] == NONE { text.len() } else { starts[next[i]] };

    let mut heap = BinaryHeap::with_capacity(n);
    let mut push = |heap: &mut BinaryHeap<Candidate>, next: &[usize], rev: &[i32], left: usize| {
        if left == NONE || next[left] == NONE {
            return;
        }
        let right = next[left];
        if let Some(priority) = priority(&text[starts[left]..end(next, right)], starts[right] - starts[left]) {
            heap.push(Candidate {
                priority,
                left,
                rev_left: rev[left],
                rev_right: rev[right],
            });
        }
    };
    for i in 0..n - 1 {
        push(&mut heap, &next, &rev, i);
    }

    while let Some(c) = heap.pop() {
        let right = next[c.left];
        if rev[c.left] != c.rev_left || right == NONE || rev[right] != c.rev_right {
            continue; // stale
        }
        rev[c.left] += 1;
        rev[right] = -1;
        let after = next[right];
        next[c.left] = after;
        if after != NONE {
            prev[after] = c.left;
        }
        push(&mut heap, &next, &rev, prev[c.left]);
        push(&mut heap, &next, &rev, c.left);
    }

    // Symbol 0 is never merged away (nothing lies to its left).
    let mut out = Vec::new();
    let mut i = 0;
    while i != NONE {
        out.push(starts[i]..end(&next, i));
        i = next[i];
    }
    out
}

// ---- llama-bpe pre-tokenizer ------------------------------------------
//
// Byte-level BPE vocabularies are trained on text pre-split by a regex, and
// their merges never cross those boundaries. This reproduces the llama-bpe
// split without a regex engine:
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,3}
//   | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//
// tried in that order at each position. Qwen2's regex has \p{N} in place of
// \p{N}{1,3}, so every digit is a piece of its own; Qwen3.5's also counts
// combining marks (\p{M}) as letters, where the others count them as
// punctuation. The codepoint classes are in unicode.rs.

struct Codepoint {
    /// Byte offset in the text.
    offset: usize,
    ch: char,
    cls: CpClass,
}

fn pretokenize_bpe(text: &str, max_digits: usize, marks_are_letters: bool) -> Vec<&str> {
    let cps: Vec<Codepoint> = text
        .char_indices()
        .map(|(offset, ch)| {
            let cls = match classify(ch as u32) {
                CpClass::Mark if marks_are_letters => CpClass::Letter,
                CpClass::Mark => CpClass::Other,
                cls => cls,
            };
            Codepoint { offset, ch, cls }
        })
        .collect();
    let offset = |k: usize| cps.get(k).map_or(text.len(), |c| c.offset);
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < cps.len() {
        let end = piece_end(&cps, i, max_digits);
        chunks.push(&text[offset(i)..offset(end)]);
        i = end;
    }
    chunks
}

/// Where the pre-tokenizer's piece that starts at codepoint `i` ends.
fn piece_end(cps: &[Codepoint], i: usize, max_digits: usize) -> usize {
    use CpClass::{Digit, Letter, Newline, Other, Space};
    let n = cps.len();
    let cls = |k: usize| cps[k].cls;
    let lower = |k: usize| cps[k].ch.to_ascii_lowercase();

    if cps[i].ch == '\'' && i + 1 < n {
        let c1 = lower(i + 1);
        let c2 = if i + 2 < n { lower(i + 2) } else { '\0' };
        if matches!((c1, c2), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
            return i + 3;
        }
        if matches!(c1, 's' | 't' | 'm' | 'd') {
            return i + 2;
        }
    }

    let mut j = i;
    if !matches!(cls(j), Letter | Digit | Newline) && j + 1 < n && cls(j + 1) == Letter {
        j += 1;
    }
    if cls(j) == Letter {
        let mut k = j;
        while k < n && cls(k) == Letter {
            k += 1;
        }
        return k;
    }

    if cls(i) == Digit {
        let mut k = i;
        while k < n && k - i < max_digits && cls(k) == Digit {
            k += 1;
        }
        return k;
    }

    let mut j = i;
    if cps[j].ch == ' ' && j + 1 < n && cls(j + 1) == Other {
        j += 1;
    }
    if cls(j) == Other {
        let mut k = j;
        while k < n && cls(k) == Other {
            k += 1;
        }
        while k < n && cls(k) == Newline {
            k += 1;
        }
        return k;
    }

    let mut k = i;
    let mut last_newline = None;
    while k < n && matches!(cls(k), Space | Newline) {
        if cls(k) == Newline {
            last_newline = Some(k);
        }
        k += 1;
    }
    if let Some(nl) = last_newline {
        nl + 1 // \s*[\r\n]+
    } else if k > i && k < n && k - i >= 2 {
        k - 1 // \s+(?!\S): leave one space for the word
    } else if k > i {
        k // \s+
    } else {
        i + 1
    }
}

/// An integer of any width and signedness, cut to `i32`.
fn kv_int(f: &GgufFile, key: &str) -> Option<i32> {
    Some(match *f.get(key)? {
        GgufValue::U8(v) => v as i32,
        GgufValue::U16(v) => v as i32,
        GgufValue::U32(v) => v as i32,
        GgufValue::U64(v) => v as i32,
        GgufValue::I8(v) => v as i32,
        GgufValue::I16(v) => v as i32,
        GgufValue::I32(v) => v,
        GgufValue::I64(v) => v as i32,
        _ => return None,
    })
}

fn kv_bool(f: &GgufFile, key: &str) -> Option<bool> {
    f.get(key).and_then(GgufValue::as_bool)
}

/// A token type: a signed integer, cut to `i32`; anything else reads as 0.
fn signed_or_zero(v: &GgufValue) -> i32 {
    match *v {
        GgufValue::I8(v) => v as i32,
        GgufValue::I16(v) => v as i32,
        GgufValue::I32(v) => v,
        GgufValue::I64(v) => v as i32,
        _ => 0,
    }
}

/// Where `needle` first occurs in `hay` at or after byte `from`.
fn find_at(hay: &str, needle: &str, from: usize) -> Option<usize> {
    Some(hay.get(from..)?.find(needle)? + from)
}

/// TranslateGemma's chat template opens with a Jinja dictionary of the
/// languages it knows: `{%- set languages = {"en": "English", ...} -%}`.
fn parse_languages(tpl: &str, out: &mut HashMap<String, String>) {
    let Some(at) = tpl.find("set languages") else { return };
    let Some(open) = find_at(tpl, "{", at) else { return };
    let Some(close) = find_at(tpl, "}", open) else { return };
    let mut key = None;
    let mut i = open + 1;
    while i < close {
        if tpl.as_bytes()[i] != b'"' {
            i += 1;
            continue;
        }
        let Some(end) = find_at(tpl, "\"", i + 1).filter(|&end| end <= close) else {
            break;
        };
        let s = &tpl[i + 1..end];
        match key.take() {
            Some(key) => {
                out.entry(String::from(key)).or_insert_with(|| s.to_owned());
            }
            None => key = Some(s),
        }
        i = end + 1;
    }
}

/// A Jinja string literal's text: "\n" (two characters in the template)
/// becomes a newline.
fn jinja_unescape(s: &str) -> String {
    s.replace("\\n", "\n")
}

/// The instruction a template with a reasoning_effort setting (Qwen3.8) puts
/// at the top of the system turn when thinking is on, for its default effort:
///
/// ```text
/// reasoning_effort|default('xhigh') ...
/// {%- if resolved_reasoning_effort == 'xhigh' %}
///     {%- set reasoning_instructions = '...' %}
/// ```
fn chatml_think_instructions(tpl: &str) -> Option<String> {
    const DEFAULT: &str = "reasoning_effort|default('";
    const SET: &str = "set reasoning_instructions = '";
    let name = tpl.find(DEFAULT)? + DEFAULT.len();
    let name_end = find_at(tpl, "'", name)?;
    let test = format!("resolved_reasoning_effort == '{}'", &tpl[name..name_end]);
    let branch = find_at(tpl, &test, name_end)?;
    let at = find_at(tpl, SET, branch)?;
    // The literal must belong to this branch, not to a later effort's.
    if find_at(tpl, "resolved_reasoning_effort ==", branch + 1).is_some_and(|other| other < at) {
        return None;
    }
    let text = at + SET.len();
    let end = find_at(tpl, "'", text)?;
    Some(jinja_unescape(&tpl[text..end]))
}

/// What a ChatML template writes on its own: the system prompt it falls back
/// to - Qwen2.5's `'<|im_start|>system\nYou are Qwen, ...<|im_end|>\n'`, a
/// literal with nothing spliced in - and how its generation prompt opens the
/// reply. Templates with an `enable_thinking` switch write an empty `<think>`
/// block to turn thinking off, which a session does unless asked to let the
/// model think, as it does Gemma 4's empty thought channel; with thinking on,
/// Qwen3.5 and 3.8 write `"<think>\n"` and Qwen3 nothing. A model that always
/// thinks has its reply opened with `"<think>\n"`.
fn parse_chatml_template(tpl: &str, v: &mut Vocab) {
    const OPEN: &str = "'<|im_start|>system\\n";
    const CLOSE: &str = "<|im_end|>\\n'";
    let mut at = tpl.find(OPEN);
    while let Some(a) = at {
        let text = a + OPEN.len();
        let Some(end) = find_at(tpl, CLOSE, text) else { break };
        let body = &tpl[text..end];
        if !body.is_empty() && !body.contains(['\'', '"', '{', '}', '+']) {
            v.default_system = jinja_unescape(body);
            break;
        }
        at = find_at(tpl, OPEN, a + 1);
    }
    let Some(generation) = tpl.rfind("add_generation_prompt") else {
        return;
    };
    let tail = &tpl[generation..];
    let markers = v.tok_thought_open >= 0 && v.tok_thought_close >= 0;
    if tail.contains("<think>\\n\\n</think>\\n\\n") {
        v.reply_open = "<think>\n\n</think>\n\n".to_owned();
        v.can_think = markers;
        if tail.contains("'<think>\\n'") {
            v.reply_open_think = "<think>\n".to_owned();
        }
        v.think_instructions = chatml_think_instructions(tpl).unwrap_or_default();
    } else if tail.contains("<|im_start|>assistant\\n<think>\\n") {
        v.reply_open = "<think>\n".to_owned();
        v.always_thinks = markers;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::error::ErrorKind;
    use crate::gguf::tests::GgufWriter;

    /// A GGUF holding just a tokenizer: `tokens` (text, GGUF token type),
    /// with descending scores, string KVs and, for byte-level BPE
    /// vocabularies, `merges`.
    pub(crate) fn vocab_gguf(tokens: &[(&str, i32)], strings: &[(&str, &str)], merges: &[&str]) -> GgufWriter {
        let mut w = GgufWriter::default();
        let texts: Vec<&str> = tokens.iter().map(|t| t.0).collect();
        let types: Vec<i32> = tokens.iter().map(|t| t.1).collect();
        let scores: Vec<f32> = (0..tokens.len()).map(|i| -(i as f32)).collect();
        w.strs("tokenizer.ggml.tokens", &texts)
            .i32s("tokenizer.ggml.token_type", &types)
            .f32s("tokenizer.ggml.scores", &scores);
        for (key, value) in strings {
            w.str(key, value);
        }
        if !merges.is_empty() {
            w.strs("tokenizer.ggml.merges", merges);
        }
        w
    }

    pub(crate) fn load_gguf(w: &GgufWriter) -> Result<Vocab> {
        Vocab::load(&GgufFile::parse_bytes(&w.bytes(), "vocab.gguf")?)
    }

    /// The vocabulary `vocab_gguf` writes.
    pub(crate) fn load_vocab(tokens: &[(&str, i32)], strings: &[(&str, &str)], merges: &[&str]) -> Vocab {
        load_gguf(&vocab_gguf(tokens, strings, merges)).unwrap()
    }

    /// A vocabulary in which every printable ASCII character is a token, so
    /// that any prompt text encodes, and renders back, exactly; `extra` are
    /// the model's special tokens. Byte-level (gpt2) for ChatML,
    /// SentencePiece for Gemma.
    pub(crate) fn load_ascii_vocab(extra: &[(&str, i32)], byte_level: bool, chat_template: &str) -> Vocab {
        let ascii: Vec<String> = (0x21u8..0x7F).map(|c| char::from(c).to_string()).collect();
        let mut tokens = extra.to_vec();
        tokens.extend(ascii.iter().map(|c| (c.as_str(), TOKEN_NORMAL)));
        let mut strings = vec![("tokenizer.chat_template", chat_template)];
        let mut merges = Vec::new();
        if byte_level {
            tokens.push(("\u{120}", TOKEN_NORMAL)); // a space, byte-level
            tokens.push(("\u{10a}", TOKEN_NORMAL)); // a newline
            tokens.push(("ab", TOKEN_NORMAL));
            strings.push(("tokenizer.ggml.model", "gpt2"));
            merges.push("a b");
        } else {
            tokens.push((SPACE_MARK, TOKEN_NORMAL)); // a space, SentencePiece
        }
        load_vocab(&tokens, &strings, &merges)
    }

    pub(crate) const CHATML_SPECIALS: &[(&str, i32)] = &[
        ("<|endoftext|>", 3),
        ("<|im_start|>", 3),
        ("<|im_end|>", 3),
        ("<think>", 4),
        ("</think>", 4),
    ];
    pub(crate) const GEMMA4_SPECIALS: &[(&str, i32)] = &[
        ("<pad>", 3),
        ("<eos>", 3),
        ("<bos>", 3),
        ("<|turn>", 3),
        ("<turn|>", 3),
        ("\n", 4),
        ("<|think|>", 3),
        ("<|channel>", 4),
        ("<channel|>", 4),
    ];

    /// Qwen3.5's generation prompt (thinking off unless asked for), cut down
    /// to the parts the loader reads.
    pub(crate) const QWEN35_TEMPLATE: &str = concat!(
        r"{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- if enable_thinking is defined and ",
        r"enable_thinking is true %}{{- '<think>\n' }}{%- else %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}",
        r"{%- endif %}",
    );
    /// Qwen3.8's (on unless switched off, with an instruction for its
    /// reasoning effort).
    pub(crate) const QWEN38_TEMPLATE: &str = concat!(
        r"{%- if enable_thinking is undefined or enable_thinking is true %}",
        r"{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}",
        r"{%- if resolved_reasoning_effort == 'xhigh' %}",
        r"{%- set reasoning_instructions = 'Think hard.\nThen answer.' %}",
        r"{%- elif resolved_reasoning_effort == 'low' %}{%- set reasoning_instructions = 'Be brief.' %}{%- endif %}",
        r"{%- endif %}",
        r"{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- if enable_thinking is defined and ",
        r"enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- else %}{{- '<think>\n' }}{%- endif %}",
        r"{%- endif %}",
    );

    /// User-defined tokens (FunctionGemma's tags, Gemma's newline and space
    /// runs) are one token wherever they appear, longest first; control
    /// tokens are not. A vocabulary with function tags is FunctionGemma's:
    /// its system prompt gets a developer turn, and a function response ends
    /// generation.
    #[test]
    fn vocab_user_defined_tokens() {
        let tokens = [
            ("<pad>", 3),
            ("<eos>", 3),
            ("<bos>", 3),
            ("<unk>", 3),
            ("a", 1),
            ("b", 1),
            (SPACE_MARK, 1),
            ("<escape>", 4),
            ("\n", 4),
            ("\n\n", 4),
            ("  ", 4),
            ("<start_of_turn>", 3),
            ("<end_of_turn>", 3),
            ("<start_function_declaration>", 4),
            ("<start_function_response>", 4),
        ];
        let v = load_vocab(&tokens, &[("tokenizer.ggml.model", "llama")], &[]);
        assert_eq!(v.tokenize("a<escape>b\n\n\na  b", false), [4, 7, 5, 9, 8, 4, 10, 5]);
        assert_eq!(v.chat, ChatFormat::Gemma);
        assert!(v.developer_turn);
        assert!(v.is_eog(14) && v.is_eog(12) && !v.is_eog(13));
    }

    /// TranslateGemma's chat template carries its language list.
    #[test]
    fn vocab_translation_languages() {
        let tokens = [
            ("<pad>", 3),
            ("<eos>", 3),
            ("<bos>", 3),
            ("<start_of_turn>", 3),
            ("<end_of_turn>", 3),
            ("\n", 4),
        ];
        let tpl = "{%- set languages = {\n    \"de\": \"German\",\n    \"pt-BR\": \"Portuguese\",\n}\n\
                   -%}{{ bos_token }}{{ content[\"source_lang_code\"] }}";
        let v = load_vocab(&tokens, &[("tokenizer.chat_template", tpl)], &[]);
        assert_eq!(v.chat, ChatFormat::Gemma);
        assert!(!v.developer_turn);
        assert_eq!(v.languages.len(), 2);
        assert_eq!(v.languages["de"], "German");
        assert_eq!(v.languages["pt-BR"], "Portuguese");
    }

    /// Qwen's pre-tokenizers take digits one at a time, llama-bpe's in runs of
    /// up to three. All keep full-width and CJK punctuation and symbols such
    /// as arrows out of the words beside them, and only Qwen3.5's keeps
    /// combining marks (Devanagari's vowel signs) inside them.
    #[test]
    fn vocab_pretokenize_qwen() {
        let tokens = [("a", 1), ("b", 1), ("ab", 1)];
        let v = ["llama-bpe", "qwen2", "qwen35"].map(|pre| {
            load_vocab(
                &tokens,
                &[("tokenizer.ggml.model", "gpt2"), ("tokenizer.ggml.pre", pre)],
                &["a b"],
            )
        });
        assert_eq!(
            [v[0].pre, v[1].pre, v[2].pre],
            [VocabPre::Llama3, VocabPre::Qwen2, VocabPre::Qwen35]
        );

        let text = "ab 12345 你好？→x";
        let want_qwen = ["ab", " ", "1", "2", "3", "4", "5", " 你好", "？→", "x"];
        let want_llama = ["ab", " ", "123", "45", " 你好", "？→", "x"];
        // नमस्ते: न म स ् त े, where ् and े are combining marks.
        let hindi = "\u{928}\u{92e}\u{938}\u{94d}\u{924}\u{947}";
        let want_split = ["\u{928}\u{92e}\u{938}", "\u{94d}\u{924}", "\u{947}"];
        assert_eq!(v[0].pretokenize(text), want_llama);
        assert_eq!(v[1].pretokenize(text), want_qwen);
        assert_eq!(v[2].pretokenize(text), want_qwen);
        assert_eq!(v[0].pretokenize(hindi), want_split);
        assert_eq!(v[1].pretokenize(hindi), want_split);
        assert_eq!(v[2].pretokenize(hindi), [hindi]);
        assert_eq!(v[1].tokenize("ab", false), [2]);
    }

    /// A vocabulary with <|im_start|> and <|im_end|> takes ChatML turns, and
    /// the template says what a session writes on its own: Qwen2.5's default
    /// system prompt, the empty <think> block that turns thinking off (Qwen3,
    /// 3.5, 3.8), or the open one of a model that always thinks - and, with
    /// thinking on, how the reply opens (Qwen3.5 and 3.8 write "<think>\n",
    /// Qwen3 nothing) and the instruction for Qwen3.8's default reasoning
    /// effort.
    #[test]
    fn vocab_chatml() {
        let tokens = [
            ("<|endoftext|>", 3),
            ("<|im_start|>", 3),
            ("<|im_end|>", 3),
            ("a", 1),
            ("b", 1),
            ("ab", 1),
            ("<think>", 4),
            ("</think>", 4),
        ];
        let templates = [
            concat!(
                r"{{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}{%- else %}",
                r"{{- '<|im_start|>system\nYou are Qwen, a helpful assistant.<|im_end|>\n' }}{%- endif %}",
                r"{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}",
            ),
            concat!(
                r"{{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}",
                r"{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}",
                r"{%- if enable_thinking is defined and ",
                r"enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}{%- endif %}",
            ),
            r"{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n<think>\n' }}{%- endif %}",
            QWEN35_TEMPLATE,
            QWEN38_TEMPLATE,
        ];
        let want_system = ["You are Qwen, a helpful assistant.", "", "", "", ""];
        let want_open = [
            "",
            "<think>\n\n</think>\n\n",
            "<think>\n",
            "<think>\n\n</think>\n\n",
            "<think>\n\n</think>\n\n",
        ];
        let want_open_think = ["", "", "", "<think>\n", "<think>\n"];
        let want_can_think = [false, true, false, true, true];
        let want_instructions = ["", "", "", "", "Think hard.\nThen answer."];
        for (i, tpl) in templates.into_iter().enumerate() {
            let v = load_vocab(
                &tokens,
                &[("tokenizer.ggml.model", "gpt2"), ("tokenizer.chat_template", tpl)],
                &["a b"],
            );
            assert_eq!(v.chat, ChatFormat::ChatMl, "template {i}");
            assert_eq!((v.tok_turn_start, v.tok_turn_end), (1, 2), "template {i}");
            assert_eq!(v.default_system, want_system[i], "template {i}");
            assert_eq!(v.reply_open, want_open[i], "template {i}");
            assert!(v.is_eog(2) && v.is_eog(0), "template {i}");
            assert_eq!((v.tok_thought_open, v.tok_thought_close), (6, 7), "template {i}");
            assert_eq!(v.reply_open_think, want_open_think[i], "template {i}");
            assert_eq!(v.can_think, want_can_think[i], "template {i}");
            assert_eq!(v.always_thinks, i == 2, "template {i}");
            assert_eq!(v.think_instructions, want_instructions[i], "template {i}");
        }
    }

    /// SentencePiece merges by score; text the vocabulary has no piece for
    /// falls back to <0xHH> tokens, and bytes without one to <unk>, a token
    /// per byte. Llama-architecture files get the leading space.
    #[test]
    fn spm_byte_fallback() {
        let tokens = [
            ("<unk>", TOKEN_UNKNOWN),
            ("<s>", TOKEN_CONTROL),
            ("</s>", TOKEN_CONTROL),
            ("<0xE4>", TOKEN_BYTE),
            ("<0xBD>", TOKEN_BYTE),
            ("<0xA0>", TOKEN_BYTE),
            (SPACE_MARK, 1),
            ("h", 1),
            ("i", 1),
            ("hi", 1),
            ("\u{2581}hi", 1),
        ];
        let mut w = vocab_gguf(
            &tokens,
            &[("tokenizer.ggml.model", "llama"), ("general.architecture", "llama")],
            &[],
        );
        w.u32("tokenizer.ggml.bos_token_id", 1)
            .u32("tokenizer.ggml.eos_token_id", 2);
        w.u32("tokenizer.ggml.unknown_token_id", 0);
        let v = load_gguf(&w).unwrap();
        assert_eq!(v.kind, VocabKind::Spm);
        assert!(v.add_space_prefix && v.add_bos && !v.add_eos);
        assert_eq!((v.bos_id, v.eos_id, v.unk_id, v.pad_id), (1, 2, 0, -1));
        assert_eq!(v.byte_ids[0xE4], 3);
        assert_eq!(v.byte_ids[0x41], -1);

        let ids = v.tokenize("hi你", true);
        assert_eq!(ids, [1, 10, 3, 4, 5]);
        assert_eq!(v.detokenize(&ids), " hi你".as_bytes());
        assert_eq!(v.tokenize("é", false), [6, 0, 0]);
        assert_eq!(v.tokenize("", true), [1]);
        assert!(v.is_eog(2) && !v.is_eog(1) && !v.is_eog(-1));

        assert_eq!(v.piece(0), b"");
        assert_eq!(v.piece(1), b"");
        assert_eq!(v.piece(3), [0xE4]);
        assert_eq!(v.piece(10), b" hi");
        assert_eq!(v.piece(-1), b"");
        assert_eq!(v.piece(11), b"");
        assert!(v.valid(10) && !v.valid(11) && !v.valid(-1));
        assert_eq!(v.n_vocab(), 11);
    }

    /// Without token types, the special ids render as nothing and any
    /// <0xHH> token as its byte.
    #[test]
    fn pieces_without_types() {
        let mut w = GgufWriter::default();
        w.strs(
            "tokenizer.ggml.tokens",
            &["<unk>", "<s>", "</s>", "<0x0A>", "\u{2581}a", "<0xZZ>", "ab"],
        )
        .str("tokenizer.ggml.model", "llama")
        .u32("tokenizer.ggml.bos_token_id", 1)
        .u32("tokenizer.ggml.eos_token_id", 2)
        .u32("tokenizer.ggml.unknown_token_id", 0)
        .bool("tokenizer.ggml.add_bos_token", false);
        let v = load_gguf(&w).unwrap();
        assert!(v.types.is_empty() && v.scores.is_empty() && !v.add_bos && v.user_defined.is_empty());
        assert_eq!(v.byte_ids[0x0A], 3);
        assert_eq!(v.detokenize(&[1, 4, 3, 5, 2, 0]), b" a\n<0xZZ>");
        assert_eq!(v.tokenize("a\n", false), [0, 3]);
        assert_eq!(v.tokenize(" a\n", false), [4, 3]);
        // Without scores every merge ties, so the leftmost pair goes first:
        // "▁a" and then "b", which only <unk> covers.
        assert_eq!(v.tokenize(" ab", false), [4, 0]);
        assert_eq!(v.eog_ids, [2]);
    }

    /// ▁-for-space BPE (Gemma 4) merges by rank, not by score.
    #[test]
    fn bpe_merges_by_rank() {
        let tokens = [
            ("a", 1),
            ("b", 1),
            ("c", 1),
            ("ab", 1),
            ("bc", 1),
            ("abc", 1),
            (SPACE_MARK, 1),
            ("\u{2581}a", 1),
        ];
        let v = load_vocab(
            &tokens,
            &[("tokenizer.ggml.model", "gemma4")],
            &["b c", "a b", "a bc", "\u{2581} a"],
        );
        assert_eq!(v.kind, VocabKind::Bpe);
        assert!(!v.add_space_prefix);
        // "b c" outranks "a b"; merging "a b" first would leave "ab" "c".
        assert_eq!(v.tokenize("abc", false), [5]);
        assert_eq!(v.tokenize("a abc", false), [0, 6, 5]);
        assert_eq!(v.tokenize(" ab", false), [6, 3]);
        assert_eq!(v.tokenize(" a", false), [7]);
        assert_eq!(v.detokenize(&[6, 3, 7]), b" ab a");
    }

    /// Equal scores merge the leftmost pair first.
    #[test]
    fn merge_ties_go_leftmost() {
        let v = load_vocab(&[("a", 1), ("aa", 1)], &[], &[]);
        assert_eq!(v.tokenize("aaa", false), [1, 0]);
        assert_eq!(v.tokenize("aaaa", false), [1, 1]);
        assert_eq!(v.tokenize("aaaaa", false), [1, 1, 0]);
    }

    /// Byte-level BPE spells bytes as printable codepoints and renders them
    /// back; user-defined tokens render as written, control tokens as nothing,
    /// and control-token spellings in the text stay text.
    #[test]
    fn gpt2_bytes_and_pieces() {
        let v = load_ascii_vocab(CHATML_SPECIALS, true, "");
        assert_eq!(v.kind, VocabKind::Gpt2);
        let x = v.token_to_id["x"];
        assert_eq!(v.tokenize(" ab\n", false), [99, 101, 100]);
        assert_eq!(v.tokenize("<think>x</think>", false), [3, x, 4]);
        assert_eq!(v.piece(99), b" ");
        assert_eq!(v.piece(100), b"\n");
        assert_eq!(v.piece(3), b"<think>");
        assert_eq!(v.piece(1), b"");

        let ids = v.tokenize("<|im_start|>", false);
        assert_eq!(ids.len(), 12);
        assert!(!ids.contains(&1));
        assert_eq!(v.detokenize(&ids), b"<|im_start|>");
        assert_eq!(v.pretokenize("<|im_start|>"), ["<|", "im", "_start", "|>"]);
        assert_eq!(
            unicode_to_bytes("\u{120}x\u{10a}\u{100}\u{143}é€"),
            b" x\n\x00\xAD\xE9\xE2\x82\xAC"
        );
    }

    #[test]
    fn byte_to_unicode_table() {
        assert_eq!(BYTE_TO_UNICODE[b'!' as usize], '!');
        assert_eq!(BYTE_TO_UNICODE[0], '\u{100}');
        assert_eq!(BYTE_TO_UNICODE[b' ' as usize], '\u{120}');
        assert_eq!(BYTE_TO_UNICODE[b'\n' as usize], '\u{10a}');
        assert_eq!(BYTE_TO_UNICODE[0xAD], '\u{143}');
        assert_eq!(BYTE_TO_UNICODE[0xFF], 'ÿ');
        for b in 0..=255u8 {
            assert_eq!(UNICODE_TO_BYTE[BYTE_TO_UNICODE[b as usize] as usize], Some(b));
        }
    }

    #[test]
    fn pretokenize_llama3_regex() {
        let v = load_vocab(&[("a", 1)], &[("tokenizer.ggml.model", "gpt2")], &["a a"]);
        assert_eq!(
            v.pretokenize("I'M  ok\n\n  x!?\n"),
            ["I", "'M", " ", " ok", "\n\n", " ", " x", "!?\n"]
        );
        assert_eq!(v.pretokenize("we're 'll'VE't"), ["we", "'re", " '", "ll", "'VE", "'t"]);
        assert_eq!(v.pretokenize("a  "), ["a", "  "]);
        assert_eq!(v.pretokenize("x \t\r\n\ty"), ["x", " \t\r\n", "\ty"]);
        assert_eq!(v.pretokenize("1234567"), ["123", "456", "7"]);
        assert_eq!(v.pretokenize("\u{a0}word"), ["\u{a0}word"]);
        assert!(v.pretokenize("").is_empty());
    }

    /// The SentencePiece leading space goes before the text and after each
    /// user-defined token; the file can turn it off.
    #[test]
    fn space_prefix() {
        let tokens = [(SPACE_MARK, 1), ("a", 1), ("\u{2581}a", 1), ("<x>", TOKEN_USER_DEFINED)];
        let llama = [("tokenizer.ggml.model", "llama"), ("general.architecture", "llama")];
        let v = load_vocab(&tokens, &llama, &[]);
        assert!(v.add_space_prefix);
        assert_eq!(v.tokenize("a<x>a", false), [2, 3, 2]);
        assert_eq!(v.tokenize("<x><x>", false), [3, 3]);

        let mut w = vocab_gguf(&tokens, &llama, &[]);
        w.bool("tokenizer.ggml.add_space_prefix", false);
        let v = load_gguf(&w).unwrap();
        assert!(!v.add_space_prefix);
        assert_eq!(v.tokenize("a<x>a", false), [1, 3, 1]);

        let v = load_vocab(
            &tokens,
            &[("tokenizer.ggml.model", "llama"), ("general.architecture", "gemma3")],
            &[],
        );
        assert!(!v.add_space_prefix);
    }

    #[test]
    fn load_errors() {
        let err = load_gguf(&GgufWriter::default()).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnsupportedModel);
        assert_eq!(err.message(), "the file has no tokenizer vocabulary");

        let mut w = GgufWriter::default();
        w.strs("tokenizer.ggml.tokens", &[]);
        assert_eq!(load_gguf(&w).unwrap_err().kind(), ErrorKind::UnsupportedModel);

        let err = load_gguf(&vocab_gguf(&[("a", 1)], &[("tokenizer.ggml.model", "gpt2")], &[])).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnsupportedModel);
        assert_eq!(err.message(), "the byte-level BPE tokenizer has no merges");
    }

    #[test]
    fn llama3_chat_format() {
        let tokens = [
            ("<|begin_of_text|>", 3),
            ("<|end_of_text|>", 3),
            ("<|start_header_id|>", 3),
            ("<|end_header_id|>", 3),
            ("<|eot_id|>", 3),
            ("<|eom_id|>", 3),
            ("a", 1),
            ("b", 1),
            ("ab", 1),
        ];
        let mut w = vocab_gguf(
            &tokens,
            &[("tokenizer.ggml.model", "gpt2"), ("tokenizer.ggml.pre", "llama-bpe")],
            &["a b"],
        );
        w.u32("tokenizer.ggml.bos_token_id", 0)
            .u32("tokenizer.ggml.eos_token_id", 1);
        let v = load_gguf(&w).unwrap();
        assert_eq!(
            (v.kind, v.pre, v.chat),
            (VocabKind::Gpt2, VocabPre::Llama3, ChatFormat::Llama3)
        );
        assert_eq!(v.model, "gpt2");
        assert_eq!((v.tok_turn_start, v.tok_header_end, v.tok_turn_end), (2, 3, 4));
        assert_eq!(v.eog_ids, [1, 4, 5]);
        assert_eq!(v.tok_newline, -1);
        assert_eq!(v.tokenize("ab", true), [0, 8]);
    }

    /// Gemma 4: <|turn> turns, the thought channel as thought markers, and
    /// thinking when the template has <|think|>; an empty thought channel
    /// opens replies when the template writes one.
    #[test]
    fn gemma4_thinking() {
        let tpl = r"{%- if enable_thinking -%}{{- '<|think|>\n' -}}{%- endif -%}";
        let e2b = load_ascii_vocab(GEMMA4_SPECIALS, false, tpl);
        let big = load_ascii_vocab(
            GEMMA4_SPECIALS,
            false,
            &format!(r"{tpl}{{{{- '<|channel>thought\n<channel|>' -}}}}"),
        );
        for v in [&e2b, &big] {
            assert_eq!(v.chat, ChatFormat::Gemma4);
            assert_eq!((v.tok_turn_start, v.tok_turn_end, v.tok_newline), (3, 4, 5));
            assert_eq!((v.tok_channel_start, v.tok_channel_end), (7, 8));
            assert_eq!((v.tok_thought_open, v.tok_thought_close, v.tok_think), (7, 8, 6));
            assert!(v.thought_label && v.can_think && !v.always_thinks && !v.developer_turn);
            assert_eq!(v.eog_ids, [4, 1]);
            assert_eq!(v.user_defined.iter().map(|u| u.1).collect::<Vec<_>>(), [5, 7, 8]);
        }
        assert!(!e2b.empty_thought && big.empty_thought);

        let none = load_ascii_vocab(GEMMA4_SPECIALS, false, "");
        assert!(!none.can_think && !none.empty_thought && none.thought_label);
    }

    /// The ASCII vocabularies encode and render text back exactly.
    #[test]
    fn ascii_round_trip() {
        let text = "Hi there!\nHow are you? <|think|> <|im_start|>{\"a\": [1, 2]}\n\n";
        for v in [
            load_ascii_vocab(CHATML_SPECIALS, true, ""),
            load_ascii_vocab(GEMMA4_SPECIALS, false, ""),
        ] {
            let ids = v.tokenize(text, true);
            assert_eq!(v.detokenize(&ids), text.as_bytes(), "{:?}", v.kind);
        }
    }

    /// Texts the comparison with the C++ reference tokenizes: punctuation and
    /// numbers, code and runs of whitespace, many scripts with their marks,
    /// emoji sequences, and the spellings of special tokens.
    const REFERENCE_TEXTS: &[&str] = &[
        "Hello, world! It's 2026: 12345 apples.",
        "I'M HERE, you're there; they'll go. We'd've DON'T stop 'til it's done' O'Neil's",
        concat!(
            "fn main() {\n    let x = 42;\t// the answer\n\n\n",
            "        if x > 1 {\r\n\t\tprintln!(\"{x}\");\n    }\n}\n   \n\t \n",
        ),
        "def f(a, b):\n\treturn a  +  b    # spaces\n\n\n\n    pass\n\t\t\t\tx = [1,2,3]  \n",
        "   leading and trailing   ",
        " ",
        "\n",
        "\n\n\n",
        "  \t  \n  \n\n x",
        "你好，世界！我们今天学习中文。こんにちは、世界。カタカナとひらがな、漢字。안녕하세요, 세계! 한국어 텍스트입니다.",
        "नमस्ते दुनिया, यह हिन्दी है। कृपया ध्यान दें। আমি বাংলায় গান গাই। வணக்கம் உலகம்",
        "สวัสดีชาวโลก ภาษาไทยไม่มีช่องว่าง ๑๒๓",
        "مَرْحَبًا بِالْعَالَمِ، كَيْفَ حَالُكَ؟ ١٢٣٤٥ שָׁלוֹם עוֹלָם, מַה שְּׁלוֹמְךָ?",
        "👋 👨‍👩‍👧‍👦 🏳️‍🌈 👍🏽 ❤️ 🇯🇵 🏴󠁧󠁢󠁥󠁮󠁧󠁿 😀😃😄",
        "3.14159 1,000,000 0x1F 12345678901234567890 ١٢٣ ๑๒๓ ２０２６ x²³ ½¼¾ €£¥₹",
        "Wait... what?! (yes) [no] {maybe} — “quotes” ‘single’ «guillemets» → ← ≠ ≤ ∞ ©®™ °C ┌─┐ 「」『』【】",
        "café naïve e\u{301} n\u{303} Ångström ΑΒΓ αβγ Привет мир 𝐀𝐁𝐂 𝑥² + 𝑦² = 𝑧²",
        "a\u{a0}b\u{2003}c\u{3000}d e\u{200b}f\u{feff}g\u{85}h",
        "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n<bos><eos><pad><unk><mask>",
        "<|im_start|>system\nYou are<|im_end|>\n<|im_start|>assistant\n<think>\nhm\n</think>\n\n\
         <tool_call>{\"a\": 1}</tool_call><|endoftext|>",
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|eom_id|><|python_tag|>",
        "<|turn>user\nx<turn|><|channel>thought\n<channel|><|think|><|tool_call>call:f{}<tool_call|><|\"|>",
        "<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>\
         <start_function_response>",
        "<0x41><0x0A> <s></s> <unused0> [INST] <<SYS>> <image_soft_token>",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa!!!!!!!!!!!!!!!!!!!!!!!!!.................          end",
        "",
        "abc123def4567ghi 你好123世界 x1y22z333 1a 22b 333c 4444d ٣a 𝟏𝟐",
        "𠀀𠀁𪜀 \u{e000}\u{f8ff} \u{1100}\u{1161}\u{11a8} ﬁﬂ ﷺ ﻼ \u{1}\u{7}\u{7f}\u{b}\u{c}\r x\r\ry",
        "'s'S's ' '' ''' 'twas' rock'n'roll 'LL 'Ve 'RE 'D 'M 'T x'",
        concat!(
            "https://example.com/path/to/file.html?q=1&r=two#frag user@example.org ",
            "C:\\Users\\me\\file.txt ~/.bashrc $HOME %PATH%",
        ),
        "Supercalifragilisticexpialidocious pneumonoultramicroscopicsilicovolcanoconiosis ΚΑΛΗΜΈΡΑ İstanbul ǅ ß ẞ",
        concat!(
            "The quick brown fox jumps over the lazy dog. Le cœur a ses raisons que la raison ne connaît point. ",
            "Der Bär läuft über die Straße. ¿Dónde está la biblioteca? Сегодня хорошая погода. ",
            "東京は日本の首都です。北京是中国的首都。서울은 한국의 수도입니다. मुंबई भारत का एक शहर है। ",
            "กรุงเทพมหานครเป็นเมืองหลวง القاهرة هي عاصمة مصر. ירושלים היא עיר עתיקה. ",
            "Numbers: 1, 22, 333, 4444, 55555, 3.14, -2.5e10, 1/2, 100%, $9.99, €1.234,56. ",
            "Code: for (int i = 0; i < n; ++i) { sum += a[i] * b[i]; } // done\n",
            "    if (x != y && y >= z || !flag) return -1;\n\tprintf(\"%d\\n\", x);\n",
            "Emoji: 🙂🙃 👩🏽‍💻 🧑‍🤝‍🧑 ✌️ 🫶🏻 #️⃣ 1️⃣ 🔟. Done!",
        ),
    ];

    /// The input echo, token ids and detokenized text in the output of
    /// `llmcpu --gemma-tokenize`.
    struct Reference {
        input: Vec<u8>,
        ids: Vec<i32>,
        detok: Vec<u8>,
    }

    fn find_bytes(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
        hay.get(from..)?
            .windows(needle.len())
            .position(|w| w == needle)
            .map(|i| i + from)
    }

    fn parse_reference(stdout: &[u8]) -> Option<Reference> {
        // stdout is in text mode, which writes each "\n" as "\r\n".
        let mut out = Vec::with_capacity(stdout.len());
        for (i, &b) in stdout.iter().enumerate() {
            if !(b == b'\r' && stdout.get(i + 1) == Some(&b'\n')) {
                out.push(b);
            }
        }
        let input = find_bytes(&out, b"input  : ", 0)? + 9;
        let count_at = find_bytes(&out, b"\ntokens : ", input)?;
        let count_end = find_bytes(&out, b"\n", count_at + 10)?;
        let n: usize = std::str::from_utf8(&out[count_at + 10..count_end])
            .ok()?
            .trim()
            .parse()
            .ok()?;
        let mut ids = Vec::with_capacity(n);
        let mut at = count_end;
        for k in 0..n {
            let line = format!("\n  [{k:3}] id=");
            at = find_bytes(&out, line.as_bytes(), at)? + line.len();
            let digits = out[at..].iter().take_while(|b| b.is_ascii_digit()).count();
            ids.push(std::str::from_utf8(&out[at..at + digits]).ok()?.parse().ok()?);
        }
        let detok = out.windows(10).rposition(|w| w == b"\ndetok  : ")? + 10;
        let detok_end = if out.ends_with(b"\n") { out.len() - 1 } else { out.len() };
        Some(Reference {
            input: out[input..count_at].to_vec(),
            ids,
            detok: out[detok..detok_end].to_vec(),
        })
    }

    /// The vocabulary part of `llmcpu --gemma-info`: the special ids, the chat
    /// format, the end-of-generation tokens and how the model thinks.
    fn info_summary(v: &Vocab) -> String {
        let chat = match v.chat {
            ChatFormat::None => "none",
            ChatFormat::Gemma => "gemma",
            ChatFormat::Gemma4 => "gemma4",
            ChatFormat::Llama3 => "llama3",
            ChatFormat::ChatMl => "chatml",
        };
        let mut s = format!(
            "tokens, {} (bos={} eos={} unk={} pad={}), chat={chat}, end-of-generation:",
            v.model, v.bos_id, v.eos_id, v.unk_id, v.pad_id
        );
        for &id in &v.eog_ids {
            s += &format!(" {id}'{}'", v.tokens[id as usize]);
        }
        if v.can_think || v.always_thinks {
            let open = if v.always_thinks {
                &v.reply_open
            } else {
                &v.reply_open_think
            };
            let open = open.replace('\n', "\\n");
            let system = match (v.tok_think >= 0, v.think_instructions.is_empty()) {
                (true, _) => "<|think|>".to_owned(),
                (false, true) => "-".to_owned(),
                (false, false) => format!("\"{}\"", v.think_instructions),
            };
            let when = if v.always_thinks { "always" } else { "with --think" };
            s += &format!("\n  thinking: {when}; the reply opens \"{open}\"; the system turn starts {system}");
        }
        s + "\n  "
    }

    /// Compares tokenization, token for token, with the C++ reference,
    /// `llmcpu --gemma-tokenize`, on real models, and what `--gemma-info`
    /// says of the vocabulary. NNCPU_REFERENCE_EXE names the llmcpu binary and
    /// NNCPU_TOKENIZER_MODELS lists GGUF files, separated by ';';
    /// NNCPU_REFERENCE_JOBS caps the processes run at once:
    ///
    /// `cargo test -p nncpu --lib tokenizer::tests::matches_reference -- --ignored --nocapture`
    #[test]
    #[ignore = "needs the C++ reference binary and model files"]
    fn matches_reference() {
        let (Some(exe), Ok(models)) = (
            std::env::var_os("NNCPU_REFERENCE_EXE"),
            std::env::var("NNCPU_TOKENIZER_MODELS"),
        ) else {
            println!("set NNCPU_REFERENCE_EXE and NNCPU_TOKENIZER_MODELS to compare with the reference");
            return;
        };
        let max_jobs: Option<usize> = std::env::var("NNCPU_REFERENCE_JOBS").ok().and_then(|j| j.parse().ok());
        let mut texts: Vec<String> = REFERENCE_TEXTS.iter().map(|t| t.to_string()).collect();
        texts.push(format!(
            "a{}b{}c{}d{}e",
            " ".repeat(100),
            "\n".repeat(40),
            "\t".repeat(37),
            " \n".repeat(20)
        ));
        texts.push(REFERENCE_TEXTS[REFERENCE_TEXTS.len() - 1].repeat(4));
        let texts = &texts;
        let mut failures = Vec::new();
        for model in models.split(';').map(str::trim).filter(|m| !m.is_empty()) {
            let f = match GgufFile::load_header(model) {
                Ok(f) => f,
                Err(e) => {
                    println!("{model}: skipped ({e})");
                    continue;
                }
            };
            let v = match Vocab::load(&f) {
                Ok(v) => v,
                Err(e) => {
                    println!("{model}: no vocabulary ({e})");
                    continue;
                }
            };
            // BF16 files load fastest mapped; others with 4-bit weights, as
            // many at once as memory allows.
            let bf16 = f.tensors.iter().any(|t| t.ggml_type == 30)
                && !f.tensors.iter().any(|t| (2..=23).contains(&t.ggml_type));
            let jobs = if bf16 {
                8
            } else {
                ((40u64 << 30) / f.file_size.max(1)).clamp(1, 8) as usize
            };
            let jobs = max_jobs.map_or(jobs, |m| jobs.min(m.max(1)));
            let weights = if bf16 { "--bf16" } else { "--q4" };
            // One run per text, then --gemma-info.
            let runs: Vec<Vec<&str>> = texts
                .iter()
                .map(|t| vec!["--gemma-tokenize", model, t.as_str()])
                .chain([vec!["--gemma-info", model]])
                .collect();
            let runs = &runs;
            let exe = &exe;
            let mut outputs: Vec<(usize, std::process::Output)> = std::thread::scope(|s| {
                let workers: Vec<_> = (0..jobs)
                    .map(|j| {
                        s.spawn(move || {
                            (j..runs.len())
                                .step_by(jobs)
                                .map(|i| {
                                    let out = std::process::Command::new(exe)
                                        .arg(weights)
                                        .args(&runs[i])
                                        .output()
                                        .expect("cannot run the reference binary");
                                    (i, out)
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                workers.into_iter().flat_map(|w| w.join().unwrap()).collect()
            });
            outputs.sort_by_key(|o| o.0);
            let info = outputs.pop().unwrap().1;
            let summary = info_summary(&v);
            if !String::from_utf8_lossy(&info.stdout)
                .replace("\r\n", "\n")
                .contains(&summary)
            {
                failures.push(format!(
                    "{model}: --gemma-info differs; expected {summary:?} in\n{}",
                    String::from_utf8_lossy(&info.stdout)
                ));
            }

            let (mut matched, mut round_trips, mut compared) = (0, 0, 0);
            for (i, out) in outputs {
                let text = &texts[i];
                let got = v.tokenize(text, v.bos_id >= 0);
                let detok = v.detokenize(&got);
                let Some(want) = parse_reference(&out.stdout) else {
                    failures.push(format!(
                        "{model}: text {i}: unreadable reference output: {}{}",
                        String::from_utf8_lossy(&out.stdout),
                        String::from_utf8_lossy(&out.stderr)
                    ));
                    continue;
                };
                if want.input != text.as_bytes() {
                    failures.push(format!(
                        "{model}: text {i}: the reference saw {:?}",
                        String::from_utf8_lossy(&want.input)
                    ));
                } else if want.ids != got {
                    let at = want.ids.iter().zip(&got).take_while(|(a, b)| a == b).count();
                    let window = |ids: &[i32]| ids[at.saturating_sub(3)..(at + 4).min(ids.len())].to_vec();
                    let short: String = text.chars().take(60).collect();
                    failures.push(format!(
                        "{model}: text {i} {short:?}: {} reference tokens, {} here; from token {at}: {:?} vs {:?}",
                        want.ids.len(),
                        got.len(),
                        window(&want.ids),
                        window(&got),
                    ));
                } else if want.detok != detok {
                    failures.push(format!(
                        "{model}: text {i}: detokenized {:?}, reference {:?}",
                        String::from_utf8_lossy(&detok),
                        String::from_utf8_lossy(&want.detok)
                    ));
                } else {
                    matched += 1;
                    compared += got.len();
                }
                let spaced = format!(" {text}");
                if detok == text.as_bytes() || (v.add_space_prefix && !text.is_empty() && detok == spaced.as_bytes()) {
                    round_trips += 1;
                }
            }
            println!(
                "{model}: {:?}/{:?}/{:?}, {} tokens: {matched}/{} texts ({compared} tokens) match the reference, \
                 {round_trips} render back exactly\n    {}",
                v.kind,
                v.pre,
                v.chat,
                v.n_vocab(),
                texts.len(),
                summary.trim_end(),
            );
        }
        for failure in &failures {
            println!("MISMATCH {failure}");
        }
        assert!(failures.is_empty(), "{} mismatches", failures.len());
    }
}
