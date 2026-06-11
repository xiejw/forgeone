//! A tiny, std-only JSON reader/writer specialized to a list of [`Record`]s.
//!
//! The crate carries no third-party crates, so JSON is hand-rolled here just
//! like the PRNG in [`crate::rng`]. The format is deliberately narrow: the file
//! is a JSON array whose elements are objects of the shape
//!
//! ```json
//! [
//!   { "rollout": "…", "reward": 1.5 },
//!   { "rollout": "…", "reward": -0.25 }
//! ]
//! ```
//!
//! [`parse`] turns such text into `Vec<Record>` and [`write`] turns a
//! `&[Record]` back into text that [`parse`] round-trips. Anything outside this
//! shape is a parse error rather than a panic, since the input is external.

use std::fmt;

// === --- Record ------------------------------------------------------- ===

/// One rollout and the reward it earned. This is the only object shape the
/// parser/writer understand.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    /// The rollout payload, stored verbatim (JSON string escapes decoded).
    pub rollout: String,
    /// The scalar reward for this rollout.
    pub reward: f32,
    /// The training generation this rollout belongs to.
    pub generation: i64,
}

/// The three object keys, named once so the reader and writer agree.
const KEY_ROLLOUT: &str = "rollout";
const KEY_REWARD: &str = "reward";
const KEY_GENERATION: &str = "generation";

// === --- Errors ------------------------------------------------------- ===

/// A parse failure, carrying a human-readable reason and the byte offset into
/// the input where the parser gave up.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonError {
    /// What went wrong.
    pub message: String,
    /// Byte offset into the input at the point of failure.
    pub pos: usize,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "json parse error at byte {}: {}", self.pos, self.message)
    }
}

impl std::error::Error for JsonError {}

/// Result alias for the parser.
pub type Result<T> = std::result::Result<T, JsonError>;

// === --- Public API --------------------------------------------------- ===

/// Parses `input` as a JSON array of [`Record`] objects.
///
/// Each element must be an object with exactly the keys `"rollout"` (a string),
/// `"reward"` (a number), and `"generation"` (an integer). Keys may appear in
/// any order. Returns a [`JsonError`] pointing at the first malformed byte.
pub fn parse(input: &str) -> Result<Vec<Record>> {
    let mut p = Parser::new(input);
    let records = p.parse_array()?;
    p.skip_ws();
    if !p.at_end() {
        return Err(p.error("trailing data after top-level array"));
    }
    Ok(records)
}

/// Serializes `records` as a pretty-printed JSON array that [`parse`] accepts.
///
/// The output is one object per line, two-space indented, with a trailing
/// newline — easy to diff and to eyeball in a file.
pub fn write(records: &[Record]) -> String {
    if records.is_empty() {
        return "[]\n".to_string();
    }
    let mut out = String::from("[\n");
    for (i, rec) in records.iter().enumerate() {
        out.push_str("  {\"");
        out.push_str(KEY_ROLLOUT);
        out.push_str("\": ");
        write_string(&rec.rollout, &mut out);
        out.push_str(", \"");
        out.push_str(KEY_REWARD);
        out.push_str("\": ");
        write_reward(rec.reward, &mut out);
        out.push_str(", \"");
        out.push_str(KEY_GENERATION);
        out.push_str("\": ");
        out.push_str(&rec.generation.to_string());
        out.push('}');
        // Commas between elements, newline after each.
        if i + 1 < records.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("]\n");
    out
}

// === --- Writer helpers ----------------------------------------------- ===

/// Appends `s` to `out` as a quoted JSON string, escaping per the spec.
fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            // Other control chars must be \u-escaped; everything else is literal.
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Appends `reward` to `out` as a JSON number. Rust's default float formatting
/// yields the shortest decimal that round-trips back to the same `f32`. JSON has
/// no NaN/Infinity, so non-finite values are written as `0` (they cannot occur
/// for a real reward and would otherwise produce invalid JSON).
fn write_reward(reward: f32, out: &mut String) {
    if reward.is_finite() {
        out.push_str(&format!("{reward}"));
    } else {
        out.push('0');
    }
}

// === --- Parser ------------------------------------------------------- ===

/// A cursor over the input bytes. JSON is ASCII-structured, so we scan bytes and
/// only assemble UTF-8 when building string contents.
struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Parser {
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    /// True once the cursor has consumed every byte.
    fn at_end(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    /// The byte under the cursor, or `None` at end of input.
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    /// Consumes and returns the byte under the cursor, or errors at end.
    fn next_byte(&mut self) -> Result<u8> {
        match self.peek() {
            Some(b) => {
                self.pos += 1;
                Ok(b)
            }
            None => Err(self.error("unexpected end of input")),
        }
    }

    /// Builds a [`JsonError`] at the current position.
    fn error(&self, message: &str) -> JsonError {
        JsonError {
            message: message.to_string(),
            pos: self.pos,
        }
    }

    /// Advances past JSON whitespace (space, tab, newline, carriage return).
    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    /// Consumes the expected byte after whitespace, or errors.
    fn expect(&mut self, want: u8) -> Result<()> {
        self.skip_ws();
        match self.peek() {
            Some(b) if b == want => {
                self.pos += 1;
                Ok(())
            }
            _ => Err(self.error(&format!("expected '{}'", want as char))),
        }
    }

    /// Parses the top-level `[ obj, obj, … ]`.
    fn parse_array(&mut self) -> Result<Vec<Record>> {
        self.expect(b'[')?;
        let mut records = Vec::new();
        self.skip_ws();
        // Empty array: `[]`.
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(records);
        }
        loop {
            records.push(self.parse_record()?);
            self.skip_ws();
            match self.next_byte()? {
                b',' => self.skip_ws(),
                b']' => break,
                _ => {
                    self.pos -= 1;
                    return Err(self.error("expected ',' or ']' after array element"));
                }
            }
        }
        Ok(records)
    }

    /// Parses one `{ "rollout": "…", "reward": n, "generation": k }` object, in
    /// any key order.
    fn parse_record(&mut self) -> Result<Record> {
        self.expect(b'{')?;
        let mut rollout: Option<String> = None;
        let mut reward: Option<f32> = None;
        let mut generation: Option<i64> = None;
        self.skip_ws();
        // Reject an empty object: all keys are required.
        if self.peek() == Some(b'}') {
            return Err(self.error("object is missing its required keys"));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.expect(b':')?;
            self.skip_ws();
            match key.as_str() {
                KEY_ROLLOUT => {
                    if rollout.is_some() {
                        return Err(self.error("duplicate 'rollout' key"));
                    }
                    rollout = Some(self.parse_string()?);
                }
                KEY_REWARD => {
                    if reward.is_some() {
                        return Err(self.error("duplicate 'reward' key"));
                    }
                    reward = Some(self.parse_number()?);
                }
                KEY_GENERATION => {
                    if generation.is_some() {
                        return Err(self.error("duplicate 'generation' key"));
                    }
                    generation = Some(self.parse_integer()?);
                }
                other => {
                    return Err(self.error(&format!("unexpected key '{other}'")));
                }
            }
            self.skip_ws();
            match self.next_byte()? {
                b',' => continue,
                b'}' => break,
                _ => {
                    self.pos -= 1;
                    return Err(self.error("expected ',' or '}' after object member"));
                }
            }
        }
        match (rollout, reward, generation) {
            (Some(rollout), Some(reward), Some(generation)) => Ok(Record {
                rollout,
                reward,
                generation,
            }),
            (None, _, _) => Err(self.error("object is missing 'rollout'")),
            (_, None, _) => Err(self.error("object is missing 'reward'")),
            (_, _, None) => Err(self.error("object is missing 'generation'")),
        }
    }

    /// Parses a JSON string literal, decoding escape sequences into UTF-8.
    fn parse_string(&mut self) -> Result<String> {
        self.skip_ws();
        if self.next_byte()? != b'"' {
            self.pos -= 1;
            return Err(self.error("expected string"));
        }
        let mut out = String::new();
        loop {
            let b = self.next_byte()?;
            match b {
                b'"' => return Ok(out),
                b'\\' => self.parse_escape(&mut out)?,
                // Raw control characters are not allowed inside JSON strings.
                0x00..=0x1f => {
                    self.pos -= 1;
                    return Err(self.error("control character in string"));
                }
                // ASCII byte: push directly.
                0x20..=0x7f => out.push(b as char),
                // Start of a multi-byte UTF-8 sequence: copy it through verbatim.
                _ => self.push_utf8_continuation(b, &mut out)?,
            }
        }
    }

    /// Decodes one escape sequence (the `\` is already consumed) into `out`.
    fn parse_escape(&mut self, out: &mut String) -> Result<()> {
        let esc = self.next_byte()?;
        match esc {
            b'"' => out.push('"'),
            b'\\' => out.push('\\'),
            b'/' => out.push('/'),
            b'b' => out.push('\u{08}'),
            b'f' => out.push('\u{0c}'),
            b'n' => out.push('\n'),
            b'r' => out.push('\r'),
            b't' => out.push('\t'),
            b'u' => self.parse_unicode_escape(out)?,
            _ => {
                self.pos -= 1;
                return Err(self.error("invalid escape sequence"));
            }
        }
        Ok(())
    }

    /// Decodes a `\uXXXX` escape (the `\u` is already consumed), joining a UTF-16
    /// surrogate pair into a single code point when present.
    fn parse_unicode_escape(&mut self, out: &mut String) -> Result<()> {
        let hi = self.parse_hex4()?;
        let code = if (0xd800..=0xdbff).contains(&hi) {
            // High surrogate: a low surrogate `\uXXXX` must immediately follow.
            if self.next_byte()? != b'\\' || self.next_byte()? != b'u' {
                return Err(self.error("expected low surrogate after high surrogate"));
            }
            let lo = self.parse_hex4()?;
            if !(0xdc00..=0xdfff).contains(&lo) {
                return Err(self.error("invalid low surrogate"));
            }
            0x10000 + (((hi - 0xd800) as u32) << 10) + (lo - 0xdc00) as u32
        } else if (0xdc00..=0xdfff).contains(&hi) {
            return Err(self.error("unexpected low surrogate"));
        } else {
            hi as u32
        };
        match char::from_u32(code) {
            Some(c) => {
                out.push(c);
                Ok(())
            }
            None => Err(self.error("invalid unicode code point")),
        }
    }

    /// Reads exactly four hex digits as a `u16`.
    fn parse_hex4(&mut self) -> Result<u16> {
        let mut value: u16 = 0;
        for _ in 0..4 {
            let b = self.next_byte()?;
            let digit = match b {
                b'0'..=b'9' => (b - b'0') as u16,
                b'a'..=b'f' => (b - b'a' + 10) as u16,
                b'A'..=b'F' => (b - b'A' + 10) as u16,
                _ => {
                    self.pos -= 1;
                    return Err(self.error("expected hex digit"));
                }
            };
            value = value * 16 + digit;
        }
        Ok(value)
    }

    /// Copies a multi-byte UTF-8 sequence starting at lead byte `lead` into
    /// `out`. The input is valid UTF-8 (it came from a `&str`), so the
    /// continuation bytes are guaranteed present and well-formed.
    fn push_utf8_continuation(&mut self, lead: u8, out: &mut String) -> Result<()> {
        let extra = match lead {
            0xc0..=0xdf => 1,
            0xe0..=0xef => 2,
            0xf0..=0xf7 => 3,
            _ => return Err(self.error("invalid UTF-8 lead byte")),
        };
        let start = self.pos - 1;
        self.pos += extra;
        let slice = &self.bytes[start..self.pos];
        let s = std::str::from_utf8(slice).expect("input was valid UTF-8 by construction");
        out.push_str(s);
        Ok(())
    }

    /// Parses a JSON number into an `f32`. Accepts an optional sign, integer and
    /// fraction digits, and an exponent — the standard JSON number grammar.
    fn parse_number(&mut self) -> Result<f32> {
        self.skip_ws();
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        self.consume_digits();
        if self.peek() == Some(b'.') {
            self.pos += 1;
            self.consume_digits();
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            self.consume_digits();
        }
        if self.pos == start {
            return Err(self.error("expected number"));
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .expect("number is ASCII by construction");
        text.parse::<f32>().map_err(|_| JsonError {
            message: format!("invalid number '{text}'"),
            pos: start,
        })
    }

    /// Parses a JSON integer into an `i64`. Accepts an optional sign followed by
    /// digits; a fraction or exponent is rejected, since `generation` is whole.
    fn parse_integer(&mut self) -> Result<i64> {
        self.skip_ws();
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        self.consume_digits();
        if self.pos == start || (self.pos == start + 1 && self.bytes[start] == b'-') {
            return Err(self.error("expected integer"));
        }
        // A fraction or exponent means this is not an integer.
        if matches!(self.peek(), Some(b'.' | b'e' | b'E')) {
            return Err(self.error("expected integer, found a fractional number"));
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .expect("integer is ASCII by construction");
        text.parse::<i64>().map_err(|_| JsonError {
            message: format!("invalid integer '{text}'"),
            pos: start,
        })
    }

    /// Advances the cursor over a run of ASCII digits.
    fn consume_digits(&mut self) {
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
    }
}

// === --- Tests -------------------------------------------------------- ===

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_list() {
        let text = r#"[
            {"rollout": "left right", "reward": 1.5, "generation": 0},
            {"rollout": "up", "reward": -2, "generation": 17}
        ]"#;
        let got = parse(text).expect("valid input");
        assert_eq!(
            got,
            vec![
                Record {
                    rollout: "left right".into(),
                    reward: 1.5,
                    generation: 0
                },
                Record {
                    rollout: "up".into(),
                    reward: -2.0,
                    generation: 17
                },
            ]
        );
    }

    #[test]
    fn parses_empty_array() {
        assert_eq!(parse("[]").expect("valid"), vec![]);
        assert_eq!(parse("  [ ]  ").expect("valid"), vec![]);
    }

    #[test]
    fn key_order_is_flexible() {
        let text = r#"[{"reward": 0.25, "generation": 3, "rollout": "x"}]"#;
        let got = parse(text).expect("valid");
        assert_eq!(
            got,
            vec![Record {
                rollout: "x".into(),
                reward: 0.25,
                generation: 3
            }]
        );
    }

    #[test]
    fn decodes_string_escapes() {
        let text = r#"[{"rollout": "a\t\"b\"\nAé", "reward": 0, "generation": 0}]"#;
        let got = parse(text).expect("valid");
        assert_eq!(got[0].rollout, "a\t\"b\"\nA\u{e9}");
    }

    #[test]
    fn decodes_surrogate_pair() {
        // U+1F600 GRINNING FACE encoded as a UTF-16 surrogate pair.
        let text = r#"[{"rollout": "😀", "reward": 0, "generation": 0}]"#;
        let got = parse(text).expect("valid");
        assert_eq!(got[0].rollout, "\u{1f600}");
    }

    #[test]
    fn round_trips_through_writer() {
        let records = vec![
            Record {
                rollout: "hello \"world\"\n\t".into(),
                reward: 1.25,
                generation: 1,
            },
            Record {
                rollout: "café 😀".into(),
                reward: -3.5,
                generation: -42,
            },
            Record {
                rollout: String::new(),
                reward: 0.0,
                generation: 0,
            },
        ];
        let text = write(&records);
        let back = parse(&text).expect("writer output parses");
        assert_eq!(records, back);
    }

    #[test]
    fn writes_empty_list() {
        assert_eq!(write(&[]), "[]\n");
    }

    #[test]
    fn rejects_missing_key() {
        assert!(parse(r#"[{"rollout": "x", "reward": 1}]"#).is_err());
        assert!(parse(r#"[{"reward": 1, "generation": 0}]"#).is_err());
        assert!(parse(r#"[{"rollout": "x", "generation": 0}]"#).is_err());
    }

    #[test]
    fn rejects_unknown_key() {
        assert!(parse(r#"[{"rollout": "x", "reward": 1, "generation": 0, "extra": 2}]"#).is_err());
    }

    #[test]
    fn rejects_trailing_data() {
        assert!(parse(r#"[] junk"#).is_err());
    }

    #[test]
    fn rejects_wrong_value_types() {
        // reward must be a number, rollout must be a string.
        assert!(parse(r#"[{"rollout": 5, "reward": 1, "generation": 0}]"#).is_err());
        assert!(parse(r#"[{"rollout": "x", "reward": "high", "generation": 0}]"#).is_err());
    }

    #[test]
    fn rejects_non_integer_generation() {
        // generation must be a whole number: no fraction or exponent allowed.
        assert!(parse(r#"[{"rollout": "x", "reward": 1, "generation": 1.5}]"#).is_err());
        assert!(parse(r#"[{"rollout": "x", "reward": 1, "generation": 1e3}]"#).is_err());
        // A plain negative integer is fine, though.
        let got = parse(r#"[{"rollout": "x", "reward": 1, "generation": -7}]"#).expect("valid");
        assert_eq!(got[0].generation, -7);
    }

    #[test]
    fn error_reports_position() {
        let err = parse("[{}]").expect_err("empty object is invalid");
        assert!(err.pos > 0);
        assert!(!err.message.is_empty());
    }
}
