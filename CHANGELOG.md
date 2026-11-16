# Changelog

## 1.0.1

### Performance

- The palette stages scale much better with output size and palette size. Nearest-entry
  searches go through a lightness-sorted index with an exact lower bound (`D ≥ w_L·ΔL²`), the
  metric's per-colour terms are computed once per colour instead of once per comparison, the
  reservation step skips colours a candidate cannot improve, and the coherence pass revisits
  only the neighbourhoods of pixels that changed. Each of these returns exactly what the
  exhaustive version returned: results are bit-identical to 1.0.0 in `balanced`, `high`,
  `maximum`, exact-colour mode and the legacy engine.
  Measured from a prepared 1 MP image (native, 2 cores), sum of five outputs of 160–320 px with
  48–64 colours: `balanced` 1 107–1 341 ms → 668–810 ms, `high` 1 705–2 162 ms → 1 064–1 121 ms.

### Changed

- `fast` now runs contour bridging like the other modes (it works on the output grid and costs
  next to nothing). Thin lines that 1.0.0 could leave open at T-junctions in `fast` are closed.
  **`fast` results differ from 1.0.0; all other modes are unchanged.**

### Fixed

- The thin-feature release gate compared against the legacy *dominant* tile mode
  (`k_centroid = 2`) while calling it "salient". It now compares against both `2` and the real
  salient mode (`k_centroid = 4`), and a new gate tests saturated thin lines (red on skin, blue
  on grey; 1–3 px wide) against the salient mode: gap-free, one pixel wide, no orientation
  lost, nothing invented.
- Exact-colour mode (`palette_strategy = Medoid`) could output colours that are not in the
  source when the source exceeded the resolution cap (area averaging) or the colour-preprocess
  threshold (colour reduction) and the mode was selected without the `exact_colors()` preset.
  Preparing with the medoid strategy now samples instead of averaging and skips the colour
  reduction.
- Documentation listed three legacy tile modes; there are four (4 = salient).

### Added

- `synth::thin_lines_with` (thin-line chart with chosen colours).

### Verification

169 unit tests and 26 release gates pass; native and WebAssembly builds are bit-identical
(combined digest `6d9db53676008e43`; the digest of 1.0.0 was `9e38d13c08509990`, the difference being the
`fast` preset on one chart).

## 1.0.0

The first stable release. A new default pipeline — **PAX, Perceptual Adaptive eXtraction** —
replaces averaging-based tile sampling; the 0.x pipeline stays available as `Engine::Legacy`.
All 0.x Rust functions, configuration fields, WebAssembly exports and CLI flags are kept.
(The design document this release implements is titled "Smart Downscaler 2.0"; the crate
version is 1.0.0.)

### Added

**Pipeline**
- `Engine::Pax` (default): source analysis → structural analysis → adaptive K-Centroid
  extraction → importance-based reconstruction → global palette → automatic compensation →
  artifact validation. See `docs/ARCHITECTURE.md`.
- Adaptive K-Centroid: exact fractional footprints for any size and aspect ratio (including
  non-integer and anisotropic reductions); 1–6 candidates per footprint chosen from content;
  deterministic density-based seeding; composite perceptual distance
  (`color::PerceptualMetric`) with chroma-dependent tolerances; transition (anti-aliasing)
  pixels discounted so candidates are pure colours; cluster validation by separation index.
- Importance-based selection `I = Aᵅ · Sᵝ · Cᵞ · Eᵟ` (area, saliency, chroma, feature evidence)
  instead of "largest cluster wins"; orientation-aware thin-feature handling (one pixel wide,
  no doubling) and contour gap bridging.
- Global palette built from the selected colours: perceptual importance weights, colour
  families (a tint is not represented by a neutral of another family when avoidable),
  gain-driven reservation of under-served colours, joint assign/update optimisation,
  optional split–merge.
- APC — Adaptive Palette Compensation: automatic chroma, contrast and lightness-distribution
  recovery on palette entries with hue preservation, perceptual gamut mapping,
  highlight/shadow protection, per-entry drift limits, halo back-off and a joint objective.
  `CompensationMode::{Off, Automatic, Manual(gain)}`.
- AAS — Artifact Avoidance System: edge-aware assignment coherence with a colour-bleeding
  guard, isolated-pixel validation against source evidence, structure and colour measurements.
- Alpha-aware processing: `smart_downscale_rgba`, `prepare_image_rgba`,
  `DownscaleResult::alpha`, `AlphaMode::{Auto, Binary, Preserve}`; transparent pixels never
  contribute colour.
- A perceptual working space (`color::Oklr`: Oklab with Ottosson's lightness toe) and
  deterministic math (`math` module) so that results are bit-identical across platforms.

**API**
- `DownscaleConfig`: `engine`, `quality_mode`, `centroid`, `compensation`, `structure`,
  `artifacts`; presets `fast()`, `balanced()`, `high()`, `maximum()`, `quality()`,
  `vibrant()`, `exact_colors()`, `legacy()`; `with_quality(mode)`.
- `QualityMode::{Fast, Balanced, High, Maximum}`.
- `DownscaleDiagnostics` and `ArtifactReport` (`DownscaleResult::diagnostics`).
- `DownscaleResult::digest()`, `colors_used()`, `to_rgba_image()`.
- `try_smart_downscale`, `try_smart_downscale_rgba`, `DownscaleError`.
- `smart_downscale_rgba_with_palette`, `downscale_rgba(&RgbaImage, …)`.
- `synth`: procedural test charts.
- Name parsing for the bindings and the CLI: `Engine::parse`, `QualityMode::parse`,
  `PaletteStrategy::parse`, `CompensationMode::parse`, `AlphaMode::parse`, …

**WebAssembly / JavaScript**
- `WasmDownscaleConfig` mirrors the whole configuration; presets `balanced()`, `high()`,
  `maximum()`, `legacy()`; `clone()`.
- `WasmDownscaleResult`: `alpha`, `has_alpha`, `colors_used`, `diagnostics`, `digest`.
- `get_quality_modes()`, `get_engines()`, `test_chart()`.
- Wrapper: `prepare()` (reusable prepared images), `createConfig()`, `testChart()`,
  `getQualityModes()`, `getEngines()`, `getPresets()`, `getPresetPaletteNames()`,
  C64 and Apple II palettes, support for `OffscreenCanvas`, `ImageBitmap`, video elements and
  plain `{data, width, height}` objects (works in Node and workers).
- `build-wasm.sh` builds with `cargo` + `wasm-bindgen` (web, bundler, Node), writes a correct
  `package.json`; `scripts/check-wasm.mjs` verifies native/WebAssembly bit-identity.

**CLI**
- `-c/--colors`, `-q/--quality`, `--preset`, `--engine`, `--use-palette`, `--save-palette`,
  `--chroma/--contrast/--luminosity`, `--strength`, `--color-preservation`,
  `--no-compensation`, structure options, `--alpha`, `--alpha-threshold`, `--preview`,
  `--diagnostics`, `--quiet`, `--version`. RGBA input and output.

**Tooling**
- `tests/release_gates.rs` (24 acceptance tests), `examples/regression.rs` (contact sheets and
  measurements), `examples/bench_compare.rs`, `examples/digest.rs`, rewritten criterion bench.

### Changed

- **Default pipeline is PAX.** Select the legacy engine for 0.x output.
- **Default segmentation is `None`** (was `Hierarchy` in Rust, `hierarchy_fast` in the
  bindings). PAX does not need a segmentation; one is still honoured if requested.
- **RGBA input is honoured** by the bindings and the CLI (it was ignored).
  `artifacts.preserve_alpha = false` restores the old behaviour.
- `k_centroid`, `k_centroid_iterations`, `neighbor_weight`, `region_weight`,
  `two_pass_refinement`, `refinement_iterations` and `chroma_recovery` only affect the legacy
  engine.
- Bindings throw on unknown strategy / segmentation / mode names and on size mismatches
  instead of silently substituting a default or trapping.
- CLI: one of `-w` / `-h` alone keeps the aspect ratio.
- Float → 8-bit conversion rounds (it truncated).
- `DownscaleResult` gained the fields `alpha` and `diagnostics`.
- The `wasm` feature no longer depends on `serde` / `serde-wasm-bindgen`; `rand`, `getrandom`
  and `thiserror` are no longer dependencies.

### Fixed

- Legacy palette extraction was not deterministic on native targets (hash-map iteration
  order); it is now, on every platform.
- `benches/downscale_bench.rs` did not compile.
- The JavaScript wrapper imported its bindings from a path that only worked inside the
  repository, and its typings declared functions the module did not export.
- `analyze_colors` ordering of equally frequent colours was not stable.
- `quantize_to_palette` dropped the alpha channel; `extract_palette_from_image` counted fully
  transparent pixels.
- `package.json` generated by the build script carried version 0.1.0 and a wrong repository URL.
- README and CLI disagreed on flag names (`-c`, `-p`, `-s`).
- An orphaned integration test under `src/` and a duplicated `.d.ts` under `src/bin/` were
  removed.

### Performance

Measured on a 2-core x86-64 VM, 512 × 512 → 64 × 64, 16 colours (see the README for more):

| Configuration | Native | WebAssembly |
|---|---:|---:|
| 0.x Rust default (`hierarchy`) | 2 336 ms | — |
| 0.x bindings default (`hierarchy_fast`) | 57.8 ms | 55.4 ms |
| 0.x without segmentation | 8.2 ms | 12.3 ms |
| 1.0 `balanced` (default) | 21.7 ms | 38.6 ms |

The 1.0 default is faster than both 0.x defaults and slower than the cheapest 0.x
configuration, which does no structural work.

### Verification

- 168 unit tests, 24 release-gate tests and the doc tests pass.
- Digests of all presets on all built-in charts are identical across repeated runs, with the
  `parallel` feature on and off, and between the native build and the WebAssembly build
  (combined digest `9e38d13c08509990`).
