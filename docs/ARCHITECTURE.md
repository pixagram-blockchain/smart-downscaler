# Architecture of smart-downscaler 1.0

This document describes the PAX pipeline as implemented, the reasoning behind its main design
decisions, how the implementation maps onto the design specification ("Smart Downscaler 2.0 —
Perceptual, Artifact-Free Pixel Reduction", released as crate version 1.0.0), where it deviates
from that specification, and how the acceptance criteria were verified.

Constants quoted here are the values in the source; each lives next to the code that uses it
and is covered by unit tests. Distances and thresholds are in working-space units, where 0.02
is roughly one just-noticeable difference (JND).

## 1. Overview

```text
RGBA / RGB source
  │ 01 source analysis             pipeline/prepare.rs, preprocess.rs, sampling/alpha.rs, analysis/*
  │ 02 structural analysis         analysis/structural.rs          ─┐ cached in PreparedImage,
  ▼                                                                 ┘ reused for every target size
WorkImage (8 bytes per pixel: L, a, b, transition, edge) + SourceStats
  │ 03 adaptive K-Centroid         sampling/footprint.rs, centroid/{initialize,assignment,clustering}.rs
  │    extraction and selection    centroid/{importance,reconstruction}.rs, artifacts/edge_consistency.rs
  ▼
Reconstruction (one target colour per output pixel + flags), from 1–6 candidates per footprint
  │ 04 global palette construction palette/{candidates,families,merging,allocation,optimization}.rs
  │ 05 colour compensation (APC)   palette/compensation.rs, color/{gamut,tone_mapping}.rs
  │ 06 pixel reconstruction        pipeline/refine.rs
  │ 07 artifact validation (AAS)   artifacts/{noise,color_drift,validation}.rs
  ▼
DownscaleResult (pixels, palette, indices, alpha, diagnostics)
```

Stage numbers follow the specification. Stages 04–06 are computed jointly (the palette is
optimised against the assignment it produces), so the code interleaves them.

Three decisions shape everything else:

1. **K-Centroid is the sampling mechanism, not a palette strategy.** The palette is built from
   the colours the footprints selected, so it cannot undo their decisions.
2. **Colour is corrected by moving palette entries against measured losses**, never by a
   saturation or contrast filter applied to the result.
3. **Artifact prevention acts only on evidence.** A pixel is changed when the source shows it
   is wrong, not because it is isolated.

## 2. Working space and determinism

### 2.1 Oklr

All decisions are made in `Oklr`: Oklab with Ottosson's lightness estimate
`Lr = ½(k₃L − k₁ + √((k₃L − k₁)² + 4k₂k₃L))`, `k₁ = 0.206`, `k₂ = 0.03`, `k₃ = (1 + k₁)/(1 + k₂)`,
and with `a`, `b` scaled by `Lr / L` so chroma shrinks with lightness the same way.

*Why not plain Oklab?* Its cube-root lightness overstates differences near black about six
times: sRGB (8, 0, 0) is ΔE ≈ 0.09 from black. Sensor noise and compression artifacts in dark
areas then look like structure — clusters split, "features" are detected in shadows, palette
entries are spent on indistinguishable blacks. With the toe, equal steps are approximately
equally visible over the whole range, and one set of thresholds works everywhere.

### 2.2 Deterministic arithmetic

Cube roots, the sRGB transfer functions, `ln`, `exp`, `pow` and `atan2` are implemented in
`math.rs` with fixed sequences of IEEE operations (`+ − × ÷ √` are correctly rounded on every
platform; libm's transcendental functions are not). Decision paths contain no random numbers
and no iteration over hash maps; parallel work is reduced in a fixed order. Consequences:

- native and WebAssembly builds produce bit-identical output (verified, §9);
- multi-threaded and single-threaded builds produce bit-identical output;
- a result can be identified by `DownscaleResult::digest()`.

Float → 8-bit conversion rounds to nearest.

### 2.3 Working pixels

`WorkPixel` is 8 bytes (`repr(C)`): `l, a, b: i16`, `mix: u8`, `edge: u8`. A megapixel is 8 MB,
one cache line holds eight pixels, and the structural terms travel with the colour. Conversion
is memoised by exact colour (65 536-slot table), because images have far fewer colours than
pixels.

## 3. Stages 01–02: analysis

**Resolution cap.** Above `max_resolution_mp` the source is reduced by exact area averaging in
linear light with premultiplied alpha (nearest-neighbour in `fast`).

**Colour preprocessing.** Sources with more than `max_color_preprocess` distinct colours are
reduced to the centroids of their RGB555 cells (≤ 32 768 colours, error below one JND).

**Alpha classification** (`sampling/alpha.rs`): `Opaque`, `Binary` (only 0/255, or partial
alpha confined to a thin fringe: ≥ 90 % of partial pixels within 2 px of both a transparent and
an opaque pixel) or `Soft`. Colour under fully transparent pixels is replaced by padding from
visible neighbours before analysis, so garbage RGB can never be read.

**Structural analysis** (`analysis/structural.rs`). For each pixel `p` and each of the four
opposite neighbour pairs `(n₁, n₂)` (horizontal, vertical, two diagonals), with a weighted norm
(lightness × `2(1 − e)`, colour × `8e`, `e = edge_weight`):

```text
g = ‖n₁ − n₂‖            first difference   (an edge passes through p in this direction)
l = ‖n₁ + n₂ − 2p‖       second difference  (p is not the mean of its neighbours)
edge(p) = max over pairs of max(g, l/2)
mix(p)  = (1 − l/g) · smoothstep(0.04, 0.12, g)      along the pair with the largest g
```

`mix` is the **transition score**. An anti-aliasing or blur pixel lies between its neighbours
(`l ≈ 0`, `g` large → `mix ≈ 1`). A pure pixel next to a hard edge, or on a one-pixel line
(`l` large), scores 0. The **noise floor** `n` is the 25th percentile of the edge histogram
(capped at 0.08); every later threshold is relaxed by it, so noisy photographs and clean
pixel art need no manual tuning.

## 4. Stage 03: adaptive K-Centroid extraction

### 4.1 Footprints

Output pixel `(i, j)` owns the source rectangle `[i·W/w, (i+1)·W/w) × [j·H/h, (j+1)·H/h)`,
computed with integer boundary arithmetic (`sampling/footprint.rs`). Partially covered first
and last rows / columns carry fractional weights. Nothing assumes integer or equal ratios:
16:9, 9:16, 127×93 → 31×17 and target ≥ source are all the same code path. Footprints larger
than the quality mode's sample budget are decimated on a lattice that leaves no row, column or
diagonal unsampled (`sampling/adaptive.rs`).

### 4.2 Sample weights

`weight = coverage · alpha · max(1 − edge_weight · mix, 0.04)`. Blend pixels still count a
little (so a footprint that is all gradient has samples) but cannot found a cluster or drag a
centroid toward a colour that exists nowhere in the artwork.

### 4.3 Clustering

Shortcuts first: a footprint whose colour extent is below `τ_flat = 0.030 + n` is one
candidate; below `τ_smooth = 0.035 + 1.5n` it is a smooth gradient and also one candidate.

Otherwise samples are binned on a small 3-D grid over the footprint's bounding box (≤ 8³
cells of 0.035) and seeded **deterministically**: the heaviest cell first, then repeatedly the
cell with the largest `weight × distance²` among cells at least `τ_distinct = max(0.05, 3n)`
from every seed. Cells that are mostly transition pixels never seed. This is a weighted
farthest-point rule: no duplicates, no empty clusters, no dependence on luck, and the dominant
colour is always first. (`deterministic: false` selects reference-style pseudo-random seeding
with a fixed seed.) Lloyd iterations run on the cells, then one or two passes on the samples.

The **distance** (`color/distance.rs`):

```text
D = w_L·ΔL² + w_C·ΔC²/(S_C(p)·S_C(q)) + w_H·ΔH²/(S_H(p)·S_H(q)) + w_E·Δmix²
S_C(c) = 1 + k_C·C(c)     S_H(c) = 1 + k_H·C(c)      ΔH² = Δa² + Δb² − ΔC²
w_L = 1, w_C = 6, w_H = 6, k_C = 12, k_H = 5, w_E = 0.008
```

- `ΔH` is the metric hue difference `2√(C₁C₂)·sin(Δh/2)`: circular by construction and
  vanishing for neutrals, where a hue angle is meaningless.
- The tolerances `S_C`, `S_H` are the device CIE94 / CIEDE2000 use: a chroma step of 0.035
  turns skin into grey but is an unremarkable change between two saturated oranges. Near the
  neutral axis the opponent axes weigh about `√6 ≈ 2.4` times lightness, in line with the
  observation behind ΔE_OK2 that Oklab's `a`, `b` are under-scaled relative to `L`.
- `Δmix` keeps transition samples from being absorbed into a flat-colour cluster.

**Validation** decides the count. Two clusters at distance `d` with spreads `σ₁`, `σ₂` are
merged when `d < τ_merge = max(0.03, 2n)`, or their separation index `SI = d/(σ₁ + σ₂)` is
below 1.6 (overlapping populations: noise or texture, not two colours), or `SI < 3.6` without
edge evidence between them (`edge_max ≥ 0.4·d`), or one is tiny (< 0.4 % of the footprint) and
close. What remains is the adaptive `K`: 1 for uniform areas, 2 on a boundary, 3 where regions
meet, up to the mode's limit (2 / 3 / 4 / 6) in textures.

Each surviving candidate records its centroid, area, spread, chroma, spatial centroid,
principal axis, elongation, thickness, **coherence** (1 for a solid shape, → 0 for scattered
samples), transition and edge statistics and its medoid (for exact-colour mode).

### 4.4 Memory

Each worker thread owns one `Scratch` with structure-of-arrays sample buffers sized for the
largest footprint; candidates are fixed-size arrays (`MAX_K = 6`); footprint spans are
precomputed. The extraction loop performs no heap allocation and uses no hash maps. Rows are
processed in parallel (`parallel` feature) and concatenated in row order.

## 5. Stage 03, continued: selection

### 5.1 Importance

"Largest cluster wins" deletes outlines, eyes and highlights, which are small by nature. Each
candidate gets

```text
I = Aᵅ · Sᵝ · Cᵞ · Eᵟ              α = 1, β = saliency_weight, γ = chroma_weight, δ = 1
A  area share of the footprint
S  = 1 + min(d_ctx, s_cap)/0.10                 distance from the 3×3 neighbourhood mean
C  = 1 + max(0, chroma − chroma_ctx)/0.06       more colourful than its surroundings
E  = 1 + ε · smoothstep(0.25, 0.60, coherence) · (1 − linstep(0.10, 0.45, share_ctx)) · linstep(0.02, 0.10, d_ctx)
     with t = thin_feature_preservation, s_cap = 0.10 + 0.32t, ε = 2.6t
```

`E` is the feature evidence: a *coherent* shape (a line or dot, not scattered noise) whose
colour is *rare in the neighbourhood* and *visibly different* from it. Noise fails the
coherence test, large regions fail the rarity test, so neither is promoted.

### 5.2 Thin features across footprints

A line narrower than an output pixel is seen by several adjacent footprints. Promoting it in
all of them doubles it; promoting it in none deletes it. `artifacts/edge_consistency.rs`
performs a greedy, orientation-aware non-maximum suppression: features are visited in a fixed
order (area, then position along the feature's normal), lines compare with their neighbours
across the line, blobs with all eight neighbours, and a feature is suppressed when a neighbour
already shows it. The result is one-pixel-wide lines. A second pass closes one-pixel gaps in
contours when a footprint has a plausible candidate (`contour_preservation`).

### 5.3 Representative colour

The winner's centroid is used as is for solid shapes. For textured footprints (low coherence)
and for `edge_preservation < 1` it is blended toward the footprint's mixture **in linear
light** (`φ = texture + (1 − texture)(1 − edge_preservation)²`,
`texture = 1 − smoothstep(0.15, 0.50, coherence)`), which is what the eye sees from a distance.
There is no sharpening step anywhere: edges are crisp because footprints take one side's colour.

Transparency takes part as a pseudo-candidate whose area is biased by `alpha_threshold`, so
sprite silhouettes follow the same rules as colour regions.

## 6. Stage 04: global palette construction

### 6.1 Importance of an output pixel

`ω = alpha · (1 + structure) · (1 + feature) · (1 + chroma) · (1 + tail) · (1 + skin)` where
`structure` combines the pixel's own edge contrast (gain 1, plus `detail_boost`) with the edge
density of its 5×5 neighbourhood (gain 2: a cheek inside a face matters more than the middle
of a wall), `feature` marks thin features (0.5), `chroma` saturated colours (0.5), `tail` the
extremes of the lightness distribution (0.6: highlights and shadows), and `skin` is
`1.5 · skin_protection`. Colours are aggregated on a fine lattice and their weights damped
(`w^0.75`; `color_rarity` lowers the exponent) so that large flat areas do not own the palette.

### 6.2 Colour families

A palette that minimises squared error spends its entries where the pixels are. A face in
front of a warm-grey wall is 2 % of the picture and two JNDs from the wall's colour, so the
error-optimal palette paints the lit side of the face grey and keeps a fourth grey for the
wall. The numbers are right and the picture is wrong: replacing a colour by another shade of
the *same material* is a matter of degree; replacing it by a colour of another *family* is a
category error.

`palette/families.rs` projects all colours onto the chromaticity plane with chroma compressed
as the metric tolerates it (`c′ = ln(1 + k_C·C)/k_C`; lightness is ignored — a family is a
ramp), finds populated regions deterministically (heaviest neighbourhood first, then largest
`weight × distance²` at least 0.020 away, at least 1 % of the weight nearby, at most 12) and
settles them with three Lloyd iterations. Wherever a colour is matched against a palette entry
— refinement, reservation, final assignment — the squared distance is multiplied by `Γ = 4`
when the two belong to different families. Images with a single family are unaffected.

### 6.3 Construction

1. **Initial cut** with the chosen `PaletteStrategy` (median cut in the working space by
   default).
2. **Lloyd refinement** (`kmeans_iterations`) with the family-aware distance.
3. **Reservation.** Candidates for a new entry are the worst-served colours of distinct
   regions and, per family, the centre of what that family is missing. For each, the *actual
   gain* (the error it removes from every colour it would serve better) is compared with the
   cost of the cheapest merge of two existing entries (Ward's criterion). The swap is made only
   when it pays. A lone outlier gains only its own error; an iris, a logo or a face gains the
   region's. `reserve_colors` forces a minimum number of swaps.
4. **Joint optimisation** against the reconstruction (1 / 2 / 4 / 8 iterations): assign →
   statistics → move entries to their weighted means → reseed entries that attract nothing →
   (`high`, `maximum`) split the entry that swallows the most edge energy and merge the
   cheapest pair.

The result is the **faithful palette**. Compensation produces the **displayed palette** from
it; assignment always uses the faithful one, so compensation cannot reshuffle pixels.

## 7. Stage 05: automatic colour compensation (APC)

Downscaling loses saturation (mixtures are greyer than their parts), contrast (extremes are
averaged) and the tails of the lightness distribution. APC measures each loss on the palette
and restores it there.

**Chroma.** For entry `i` with chroma `C` and members whose source chroma averages `C_src`:
`C′ = C + λ_C · conf · (C_src − C)` along the entry's own hue, with
`conf = smoothstep(0.015, 0.05, C_src) · smoothstep(0.35, 0.75, C/C_src)` — greys stay grey,
and entries whose members disagree on hue (a real mixture) are left alone. A source gate
halves the gain when the output is already more colourful than the source as a whole.

**Lightness distribution.** One global monotone curve (piecewise cubic, Fritsch–Carlson
tangents) through the 3 / 10 / 25 / 50 / 75 / 90 / 97 % percentiles of the palette's and the
target image's lightness distributions. It is one-sided about the median (tails move outwards,
never inwards) and its slope is limited to [0.85, 1.6]. Equal values are merged before
quantiles are taken, so a lossless result (pixel art reduced to its logical size) yields the
identity curve.

**Contrast.** A gain about the mean, `g = √(g_σ · g_edge)` with local adaptation (`g_σ` from the
standard deviations, `g_edge` from the contrast across palette-entry borders versus the source),
capped at 1.35 and reduced when the output already reaches the source's contrast.

**Constraints.** Soft limits at the source's own shadow floor and highlight ceiling
(0.2 / 99.8 %); mean lightness preserved; contrast gain backed off if any border between
entries would exceed the source's contrast (halo guard); per-entry drift limited to
0.05 / 0.08 / 0.12 (`color_preservation`); perceptual gamut mapping (hue and lightness kept,
chroma reduced to the sRGB boundary).

**Joint objective.** The strengths of the chroma and tone corrections are chosen on a 5 × 5
grid by minimising

```text
E(P) = 1.0·E_color + 2.0·E_chroma + 2.0·E_contrast + 1.0·E_luminance + 2.0·E_structure + 1.0·E_gamut + 0.5·E_palette
```

so a correction is applied only as far as it improves the whole. `Manual(gain)` bypasses the
source gates for that channel but not the constraints. Resolution awareness scales automatic
corrections by `0.85 + 0.30·linstep(2, 16, reduction)`: stronger reductions lose more, with
more certainty.

## 8. Stages 06–07: pixel reconstruction and artifact validation (AAS)

**Pixel reconstruction** (`pipeline/refine.rs`). Exhaustive nearest entry, then an edge-aware coherence
pass (iterated conditional modes on a Potts energy):
`E_j(i) = D(T_j, P_i) − β · Σ_n a_jn · [index_n = i]`, `β = 0.5 · spatial_coherence · ē` (`ē` =
mean quantisation error, so clean images are not smoothed at all), affinity
`a_jn = 1 − linstep(τ, 2.5τ, ‖T_j − T_n‖)` (zero across edges; diagonals × 0.7;
`τ = 0.04 + min(n, 0.04)`). The **bleeding guard** only lets a pixel move to an entry within
0.03 of its best match.

**Isolated pixels** (`artifacts/noise.rs`). Connected components of one pixel (two in `high`
and `maximum`) are examined. A component is kept when the reconstruction shows evidence for it
(it was a selected feature, or its target colour differs from its surroundings by ≥ 0.05);
otherwise it is given its neighbours' entry if that entry is nearly as good. Majority voting
is never used.

| Artifact | Prevention |
|---|---|
| Muddy colours | Transition pixels excluded from centroids; chroma recovery on the palette; joint objective |
| Blurred contours | A footprint on a boundary takes one side's colour, never the average |
| Missing thin features | Importance with feature evidence; palette weight for features |
| Jagged / doubled / broken lines | Orientation-aware suppression across footprints; contour bridging |
| Colour bleeding | Affinity zero across edges; bleeding guard; family penalty |
| Isolated noise pixels | Component validation against source evidence |
| Halos | No sharpening; one monotone tone curve; halo guard on contrast gain |
| Banding | Rarity-damped allocation gives gradients entries; coherence keeps band contours clean |
| Hue shifts | Hue-preserving chroma scaling and gamut mapping; hue shift reported |
| Excessive contrast | Slope limits, gain cap, source gates |
| Lost highlights / crushed shadows | Tail weighting in the palette; range limits in the tone curve |
| Transparent-colour contamination | Premultiplied weights; padding under alpha 0; alpha classification |

## 9. Verification

### 9.1 Acceptance criteria

| Gate (specification §11) | Result | Evidence |
|---|---|---|
| Determinism | **Met.** Identical across runs, thread counts and native / WebAssembly | `identical_inputs_give_identical_outputs`, `examples/digest.rs`, `scripts/check-wasm.mjs`: 42 results, combined digest `6d9db53676008e43` (1.0.1) |
| Colour compensation reduces chroma, contrast and lightness drift | **Met** on the charts | `compensation_never_increases_drift_and_reports_it`, `muted_and_neutral_images_are_not_saturated` |
| Edge preservation better than the current release | **Met** on the structural set | `edges_are_crisper_and_structure_more_accurate_than_the_legacy_engine`; see the README image |
| Thin-feature retention: no regression against the salient mode | **Met** | `thin_features_are_kept_without_inventing_any` (against `k_centroid = 2` and the salient mode `k_centroid = 4`; with its defaults the legacy engine loses 2-px lines at 4× entirely) and `colourful_thin_features_hold_up_against_the_legacy_salient_mode` (saturated lines 1–3 px wide: gap-free, one pixel wide, no orientation lost) |
| Alpha correctness | **Met** | `transparent_pixels_never_contaminate_colour`, `alpha_modes_and_opt_out` |
| Performance: ≤ 20 % overhead in Balanced | **Met against the 0.x defaults, not against the cheapest 0.x configuration** | Balanced is 2.7–4.4× faster than `hierarchy_fast` and ≈ 100× faster than `hierarchy`, but 1.8–2.7× slower than 0.x without segmentation (README, Performance) |
| Memory: reusable buffers, no per-tile heap allocations | **Met** for extraction (by construction, §4.4); the palette stages allocate per call, not per pixel | code |
| No unexplained degradation on reference images | Checked visually on five photographs and twelve charts during development; **the specification's reference images were not available** | `examples/regression.rs` |

### 9.2 Test suite

169 unit tests (next to the code they test), 26 release-gate tests on procedural charts
(`tests/release_gates.rs`), doc tests. The charts cover the specification's list: pixel art
(contours, isolated pixels, dither), an illustration with outlines, a portrait (skin, lips,
hair), gradients, high-frequency texture, thin lines in four orientations, checkerboards,
transparent sprites, one-pixel details, extreme ratios (down to 1 × 1) and palette budgets of
1, 2, 4 and 8. Compensation (`palette/compensation.rs`) is covered end to end by the release
gates (drift never increases, identity on lossless input, muted images stay muted) rather than
by unit tests of its own.

## 10. Specification compliance

| Specification | Status | Notes |
|---|---|---|
| PAX pipeline, stages 01–07 | Implemented | §3–§8 |
| Exact footprints, any ratio and aspect | Implemented | §4.1 |
| Adaptive cluster count 1 / 2 / 3 / 4–6 | Implemented | By validation rather than by a classifier: the count is what survives merging |
| Deterministic initialisation | Implemented | Density-weighted farthest-point on a bin grid; equivalent in intent to weighted k-means++ without randomness |
| Composite distance `w_L ΔL² + w_C ΔC² + w_H ΔH² + w_E ΔE_edge²` | Implemented, extended | Chroma-dependent tolerances added; `ΔE_edge` realised as the transition-score difference |
| Importance `Aᵅ Sᵝ Cᵞ Eᵟ` | Implemented | §5.1 |
| Edge-aware reconstruction, no unconditional sharpening | Implemented | §5.3 |
| Automatic chroma / contrast / luminosity compensation | Implemented | §7. The linear `σ_s/σ_r` contrast model is used only as one factor of the constrained gain, as the specification recommends for production |
| Joint objective `E(P)` with seven terms | Implemented | Grid search over two strengths rather than gradient descent: bounded, deterministic |
| Resolution-aware compensation "only with evidence" | Implemented | Source gates, confidence terms, identity on lossless input |
| Artifact table (12 classes) | Implemented | §8 |
| Spatial coherence without erosion | Implemented | §8 |
| Alpha-aware processing | Implemented | §3, §5.3 |
| Three-level palette (local candidates → global → optimisation) | Implemented | §4–§6 |
| Palette relationships (skin/hair, outline/fill, warm/cool …) | Implemented as colour families | §6.2: a generic mechanism rather than named relationships |
| Module layout | Followed | `analysis/`, `color/`, `sampling/`, `centroid/`, `palette/`, `artifacts/`, `pipeline/`, `wasm.rs`; plus `palette/families.rs`, `math.rs`, `synth.rs`, `diagnostics.rs`, `error.rs`, `config.rs` |
| Configuration and diagnostics API | Implemented | Additional fields: `engine`, `centroid.chroma_weight`, `centroid.metric`, `compensation.gamut_mapping`, `compensation.color_preservation`, `artifacts.alpha_mode`, `artifacts.alpha_threshold`; additional diagnostics |
| Quality presets table | Implemented | README, Quality modes |
| Contiguous memory, reusable buffers, fixed arrays, no hash maps in clustering | Implemented | §4.4 |
| SIMD where measured | **Not done** | Loops are written to auto-vectorise; no explicit SIMD |
| Parallel tile extraction with deterministic reduction | Implemented | `parallel` feature; `prepare` itself is sequential |
| Cached working-space representations | Implemented | `PreparedImage` |
| Metrics: SSIM / MS-SSIM / LPIPS | **Not implemented** | The suite measures colour error, edge retention, structural error, chroma / contrast / lightness retention and feature recall, and writes contact sheets for human comparison |
| Reference datasets | **Partly** | Procedural charts with known content; `examples/regression.rs` accepts any images |

### Deviations worth knowing

- **Working space.** Oklr instead of plain Oklab / OKLCH (§2.1). Public Oklab utilities are
  unchanged.
- **Colour families** are an addition: the specification asks for palette relationships to be
  preserved and for allocation by perceptual importance; families are how that is achieved.
- **Exact-colour mode** (`PaletteStrategy::Medoid`) disables compensation, since its promise is
  that every output colour exists in the source.
- **Default segmentation is none.** PAX's structure handling does not need the 0.x
  segmentations; they remain available.
- **"Artifact-free"** is treated as the objective that the validation stage works toward and
  reports on, not as a property that can be guaranteed for arbitrary input and palette size.

## 11. Limitations and next steps

- `prepare` (55–75 % of a call) is single-threaded and scalar. Row-parallel conversion with
  deterministic merging of the statistics and explicit SIMD in the structure kernel are the
  obvious next optimisations.
- The family separation (0.020) and penalty (Γ = 4) are fixed. They were tuned on portraits,
  animals, objects and scenes; an adaptive separation derived from the image's chromaticity
  density would be more principled.
- Regular patterns close to the sampling limit at non-integer ratios produce a regular but
  biased pattern.
- Perceptual quality was judged on charts and a handful of photographs. A larger reference set
  with human comparison, as the specification asks, is still to be assembled; the tooling for
  it is `examples/regression.rs`.
- No dithering and no temporal coherence (animations are processed frame by frame; use a fixed
  palette to keep frames consistent).
