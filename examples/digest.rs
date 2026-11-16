//! Prints a digest of the output for every chart × preset.
//!
//! Output is a pure function of the input and the configuration: the digests must be identical
//! across runs, with or without the `parallel` feature, and between native and WebAssembly
//! builds (the JS wrapper's `digest` check prints the same table).
//!
//! ```text
//! cargo run --release --example digest
//! cargo run --release --example digest --no-default-features
//! ```

use smart_downscaler::prelude::*;
use smart_downscaler::synth;

fn digest(r: &DownscaleResult) -> u64 {
    r.digest()
}

fn main() {
    let presets: [(&str, DownscaleConfig); 6] = [
        ("fast", DownscaleConfig::fast()),
        ("balanced", DownscaleConfig::balanced()),
        ("high", DownscaleConfig::high()),
        ("maximum", DownscaleConfig::maximum()),
        ("exact", DownscaleConfig::exact_colors()),
        ("legacy", DownscaleConfig::legacy()),
    ];
    let rgb: [(&str, Vec<Rgb>, usize, usize, u32, u32); 5] = [
        ("portrait", synth::portrait(256, 256), 256, 256, 64, 64),
        ("illustration", synth::illustration(320, 180), 320, 180, 96, 54),
        ("gradient", synth::gradient(200, 120), 200, 120, 37, 23),
        ("texture", synth::texture(192, 192), 192, 192, 48, 48),
        ("pixel_art", synth::pixel_art(32, 32, 6).0, 192, 192, 40, 40),
    ];
    let mut all = 0xcbf2_9ce4_8422_2325u64;
    for (pname, cfg) in &presets {
        for (cname, px, w, h, tw, th) in &rgb {
            let d = digest(&smart_downscale(px, *w, *h, *tw, *th, cfg));
            println!("{pname:9} {cname:13} {d:016x}");
            all = (all ^ d).wrapping_mul(0x0000_0100_0000_01b3);
        }
        for soft in [false, true] {
            let rgba = synth::sprite_rgba(160, 160, soft);
            let d = digest(&smart_downscale_rgba(&rgba, 160, 160, 40, 40, cfg));
            println!("{pname:9} sprite(soft={soft:5}) {d:016x}");
            all = (all ^ d).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    println!("combined {all:016x}");
}
