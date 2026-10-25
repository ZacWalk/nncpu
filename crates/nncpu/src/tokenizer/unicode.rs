//! Codepoint classes for the byte-level BPE pre-tokenizer.
//!
//! Letters and digits are exact for ASCII. Beyond it, codepoints are letters
//! unless they are whitespace, or digits, marks, punctuation or symbols in the
//! blocks ordinary text draws on (`CP_RANGES` and the Indic layout in
//! [`classify`]).

/// What the pre-tokenizer's regex makes of a codepoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CpClass {
    Letter,
    Digit,
    Space,
    Newline,
    Other,
    /// A combining mark: becomes `Letter` or `Other`, per pre-tokenizer.
    Mark,
}

use CpClass::{Digit, Letter, Mark, Newline, Other, Space};

/// Codepoints past ASCII whose class is not the default, letter: inclusive
/// ranges, sorted and disjoint, for a binary search.
#[rustfmt::skip]
const CP_RANGES: &[(u32, u32, CpClass)] = &[
    (0x00A1, 0x00A9, Other), (0x00AB, 0x00B1, Other), (0x00B2, 0x00B3, Digit),
    (0x00B4, 0x00B4, Other), (0x00B6, 0x00B8, Other), (0x00B9, 0x00B9, Digit),
    (0x00BB, 0x00BB, Other), (0x00BC, 0x00BE, Digit), (0x00BF, 0x00BF, Other),
    (0x00D7, 0x00D7, Other), (0x00F7, 0x00F7, Other), (0x0300, 0x036F, Mark),
    (0x0483, 0x0489, Mark), (0x055A, 0x055F, Other), (0x0589, 0x058A, Other),
    (0x0591, 0x05BD, Mark), (0x05BE, 0x05BE, Other), (0x05BF, 0x05BF, Mark),
    (0x05C0, 0x05C0, Other), (0x05C1, 0x05C2, Mark), (0x05C3, 0x05C3, Other),
    (0x05C4, 0x05C5, Mark), (0x05C6, 0x05C6, Other), (0x05C7, 0x05C7, Mark),
    (0x05F3, 0x05F4, Other), (0x0600, 0x060F, Other), (0x0610, 0x061A, Mark),
    (0x061B, 0x061F, Other), (0x064B, 0x065F, Mark), (0x0660, 0x0669, Digit),
    (0x066A, 0x066D, Other), (0x0670, 0x0670, Mark), (0x06D4, 0x06D4, Other),
    (0x06D6, 0x06DC, Mark), (0x06DD, 0x06DE, Other), (0x06DF, 0x06E4, Mark),
    (0x06E7, 0x06E8, Mark), (0x06E9, 0x06E9, Other), (0x06EA, 0x06ED, Mark),
    (0x06F0, 0x06F9, Digit), (0x06FD, 0x06FE, Other), (0x0E31, 0x0E31, Mark),
    (0x0E34, 0x0E3A, Mark), (0x0E3F, 0x0E3F, Other), (0x0E47, 0x0E4E, Mark),
    (0x0E4F, 0x0E4F, Other), (0x0E50, 0x0E59, Digit), (0x0E5A, 0x0E5B, Other),
    (0x0EB1, 0x0EB1, Mark), (0x0EB4, 0x0EBC, Mark), (0x0EC8, 0x0ECE, Mark),
    (0x0ED0, 0x0ED9, Digit), (0x0F20, 0x0F33, Digit), (0x1040, 0x1049, Digit),
    (0x17E0, 0x17E9, Digit), (0x1810, 0x1819, Digit), (0x1AB0, 0x1AFF, Mark),
    (0x1DC0, 0x1DFF, Mark), (0x200B, 0x205E, Other), (0x2060, 0x206F, Other),
    (0x2070, 0x2070, Digit), (0x2074, 0x2079, Digit), (0x207A, 0x207E, Other),
    (0x2080, 0x2089, Digit), (0x208A, 0x208E, Other), (0x20A0, 0x20C0, Other),
    (0x20D0, 0x20F0, Mark), (0x2150, 0x2189, Digit), (0x218A, 0x218B, Other),
    (0x2190, 0x245F, Other), (0x2460, 0x249B, Digit), (0x249C, 0x24E9, Other),
    (0x24EA, 0x24FF, Digit), (0x2500, 0x2775, Other), (0x2776, 0x2793, Digit),
    (0x2794, 0x2BFF, Other), (0x2E00, 0x2E7F, Other), (0x3001, 0x3004, Other),
    (0x3007, 0x3007, Digit), (0x3008, 0x3020, Other), (0x3021, 0x3029, Digit),
    (0x302A, 0x302F, Mark), (0x3030, 0x3030, Other), (0x3036, 0x3037, Other),
    (0x3038, 0x303A, Digit), (0x303D, 0x303F, Other), (0x3099, 0x309A, Mark),
    (0x309B, 0x309C, Other), (0x30A0, 0x30A0, Other), (0x30FB, 0x30FB, Other),
    (0x3190, 0x3191, Other), (0x3192, 0x3195, Digit), (0x3196, 0x319F, Other),
    (0x3200, 0x321E, Other), (0x3220, 0x3229, Digit), (0x322A, 0x3247, Other),
    (0x3248, 0x324F, Digit), (0x3250, 0x3250, Other), (0x3251, 0x325F, Digit),
    (0x3260, 0x327F, Other), (0x3280, 0x3289, Digit), (0x328A, 0x32B0, Other),
    (0x32B1, 0x32BF, Digit), (0x32C0, 0x33FF, Other), (0xFE00, 0xFE0F, Mark),
    (0xFE10, 0xFE19, Other), (0xFE20, 0xFE2F, Mark), (0xFE30, 0xFE6B, Other),
    (0xFEFF, 0xFEFF, Other), (0xFF01, 0xFF0F, Other), (0xFF10, 0xFF19, Digit),
    (0xFF1A, 0xFF20, Other), (0xFF3B, 0xFF40, Other), (0xFF5B, 0xFF65, Other),
    (0xFFE0, 0xFFEE, Other), (0xFFF9, 0xFFFD, Other), (0x1D7CE, 0x1D7FF, Digit),
    (0x1F000, 0x1F0FF, Other), (0x1F100, 0x1F10C, Digit), (0x1F10D, 0x1FBEF, Other),
    (0x1FBF0, 0x1FBF9, Digit), (0xE0000, 0xE007F, Other), (0xE0100, 0xE01EF, Mark),
];

const _: () = {
    let mut i = 0;
    while i < CP_RANGES.len() {
        let (lo, hi, _) = CP_RANGES[i];
        assert!(
            lo <= hi && (i == 0 || lo > CP_RANGES[i - 1].1),
            "CP_RANGES must be sorted and disjoint"
        );
        i += 1;
    }
};

pub(super) fn classify(cp: u32) -> CpClass {
    match cp {
        0x0A | 0x0D => return Newline,
        0x20 | 0x09 | 0x0B | 0x0C | 0x85 | 0xA0 | 0x2000..=0x200A | 0x2028 | 0x2029 | 0x202F | 0x205F | 0x3000 => {
            return Space;
        }
        _ => {}
    }
    if cp < 0x80 {
        return match cp {
            0x61..=0x7A | 0x41..=0x5A => Letter,
            0x30..=0x39 => Digit,
            _ => Other,
        };
    }
    // Devanagari to Malayalam share one layout per 128-codepoint block:
    // vowel signs, virama and other marks, then the digits.
    if (0x0900..0x0D80).contains(&cp) {
        return match cp & 0x7F {
            0x66..=0x6F => Digit,
            0x00..=0x03 | 0x3A..=0x3C | 0x3E..=0x4F | 0x51..=0x57 | 0x62 | 0x63 => Mark,
            _ if matches!(cp, 0x0964 | 0x0965 | 0x0970) => Other,
            _ => Letter,
        };
    }
    let i = CP_RANGES.partition_point(|&(lo, _, _)| lo <= cp);
    match i.checked_sub(1).map(|i| CP_RANGES[i]) {
        Some((_, hi, cls)) if cp <= hi => cls,
        _ => Letter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes() {
        let cases = [
            ('a', Letter),
            ('Z', Letter),
            ('7', Digit),
            ('_', Other),
            ('\t', Space),
            ('\r', Newline),
            ('\u{a0}', Space),
            ('\u{3000}', Space),
            ('\u{200b}', Other),
            ('é', Letter),
            ('²', Digit),
            ('\u{301}', Mark),
            ('\u{5b4}', Mark),
            ('\u{5be}', Other),
            ('٣', Digit),
            ('\u{64e}', Mark),
            ('क', Letter),
            ('\u{93f}', Mark),
            ('\u{94d}', Mark),
            ('।', Other),
            ('०', Digit),
            ('\u{bcd}', Mark),
            ('\u{e31}', Mark),
            ('๓', Digit),
            ('你', Letter),
            ('，', Other),
            ('？', Other),
            ('２', Digit),
            ('→', Other),
            ('𝟏', Digit),
            ('😀', Other),
            ('\u{fe0f}', Mark),
            ('\u{e0067}', Other),
        ];
        for (c, want) in cases {
            assert_eq!(classify(c as u32), want, "{c:?}");
        }
    }
}
