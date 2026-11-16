//! Bounded sampling of very large footprints.
//!
//! At extreme reduction ratios one footprint can cover tens of thousands of source samples.
//! Clustering all of them buys nothing — a structure thinner than a few percent of an output
//! pixel cannot be represented anyway — so above a per-quality budget the footprint is
//! decimated: in row `r` only the columns `≡ offset(r) (mod n)` are used.
//!
//! The offsets are chosen so that the pattern has **no blind lines**: every row, every column
//! and every 45° diagonal of the footprint is sampled at the same rate `1/n`. (A plain
//! "every n-th row and column" grid misses `1 − 1/n` of all one-pixel lines outright, and a
//! checkerboard misses half of all diagonals.) For prime `n ≥ 5` this is a rank-1 lattice
//! `offset(r) = −m·r mod n` with `m ∉ {0, ±1}`; for `n = 2` and `n = 3`, where no such lattice
//! exists, short hand-built offset sequences with the same property are used.
//!
//! The pattern is a pure function of the position inside the footprint: fully deterministic.

/// Decimation pattern: in footprint row `r`, columns `first_col(r), first_col(r) + n, …`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleLattice {
    /// Decimation factor (1 = keep every sample).
    pub n: u32,
    /// Lattice multiplier for prime `n ≥ 5` (unused otherwise).
    pub m: u32,
}

/// Offsets for `n = 2` (period 4): rows, columns and both diagonals each see both residues.
const PAT2: [u8; 4] = [0, 1, 1, 0];
/// Offsets for `n = 3` (period 9): `o(r)`, `o(r) − r` and `o(r) + r` each cover all residues.
const PAT3: [u8; 9] = [0, 1, 2, 1, 2, 0, 2, 0, 1];

/// Prime decimations with multipliers near `n/φ²` (well-spread rank-1 lattices).
const PRIMES: [(u32, u32); 31] = [
    (5, 2), (7, 3), (11, 4), (13, 5), (17, 7), (19, 7), (23, 9), (29, 11), (31, 12), (37, 14),
    (41, 16), (47, 18), (53, 20), (61, 23), (71, 27), (83, 32), (97, 37), (113, 43), (131, 50),
    (151, 58), (179, 68), (211, 81), (251, 96), (293, 112), (347, 133), (409, 156), (479, 183),
    (563, 215), (661, 252), (773, 295), (907, 346),
];

impl SampleLattice {
    /// Keep every sample.
    pub const FULL: SampleLattice = SampleLattice { n: 1, m: 0 };

    /// First used column in footprint row `row` (then every `n`-th).
    #[inline(always)]
    pub fn first_col(&self, row: usize) -> usize {
        match self.n {
            0 | 1 => 0,
            2 => PAT2[row & 3] as usize,
            3 => PAT3[row % 9] as usize,
            n => {
                let n = n as usize;
                (n - (self.m as usize * row) % n) % n
            }
        }
    }

    #[inline(always)]
    pub fn step(&self) -> usize {
        self.n.max(1) as usize
    }
}

/// Lightest decimation that keeps a `samples`-sample footprint within `max_samples`.
pub fn lattice_for(samples: usize, max_samples: usize) -> SampleLattice {
    if max_samples == 0 || samples <= max_samples {
        return SampleLattice::FULL;
    }
    let want = (samples + max_samples - 1) / max_samples; // ceil
    match want {
        0 | 1 => SampleLattice::FULL,
        2 => SampleLattice { n: 2, m: 0 },
        3 => SampleLattice { n: 3, m: 0 },
        _ => {
            for &(n, m) in PRIMES.iter() {
                if n as usize >= want {
                    return SampleLattice { n, m };
                }
            }
            let (n, m) = PRIMES[PRIMES.len() - 1];
            SampleLattice { n, m }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hits(l: SampleLattice, w: usize, h: usize) -> Vec<Vec<bool>> {
        let mut g = vec![vec![false; w]; h];
        for (row, r) in g.iter_mut().enumerate() {
            let mut c = l.first_col(row);
            while c < w {
                r[c] = true;
                c += l.step();
            }
        }
        g
    }

    #[test]
    fn full_lattice_keeps_everything() {
        assert_eq!(lattice_for(100, 256), SampleLattice::FULL);
        assert_eq!(lattice_for(256, 256), SampleLattice::FULL);
        let g = hits(SampleLattice::FULL, 5, 4);
        assert!(g.iter().flatten().all(|&b| b));
    }

    #[test]
    fn budget_is_respected_and_density_is_uniform() {
        for &(w, h, budget) in &[(40usize, 40usize, 256usize), (100, 62, 1024), (300, 300, 4096), (17, 17, 256), (23, 23, 256)] {
            let l = lattice_for(w * h, budget);
            assert!(l.n > 1);
            let g = hits(l, w, h);
            let count = g.iter().flatten().filter(|&&b| b).count();
            assert!(count <= budget + w.max(h), "{w}x{h}: {count} > {budget}");
            assert!(count * 3 >= budget, "{w}x{h}: far sparser than needed: {count} of {budget}");
        }
    }

    #[test]
    fn no_blind_rows_columns_or_diagonals() {
        // Every decimation that can be chosen, on a footprint a few periods wide.
        let mut all: Vec<SampleLattice> = vec![SampleLattice { n: 2, m: 0 }, SampleLattice { n: 3, m: 0 }];
        all.extend(PRIMES.iter().take(12).map(|&(n, m)| SampleLattice { n, m }));
        for l in all {
            let size = (l.n as usize * 9).max(36);
            let g = hits(l, size, size);
            let span = size - l.n as usize * 4; // lines at least 4 periods long
            for i in 0..size {
                assert!(g[i].iter().any(|&b| b), "n={} row {i}", l.n);
                assert!((0..size).any(|r| g[r][i]), "n={} column {i}", l.n);
            }
            for d in 0..(size - span) {
                assert!((0..size - d).any(|i| g[i][i + d]), "n={} diagonal +{d}", l.n);
                assert!((0..size - d).any(|i| g[i + d][i]), "n={} diagonal -{d}", l.n);
                assert!((0..size - d).any(|i| g[i][size - 1 - d - i]), "n={} anti-diagonal +{d}", l.n);
                assert!((0..size - d).any(|i| g[i + d][size - 1 - i]), "n={} anti-diagonal -{d}", l.n);
            }
        }
    }

    #[test]
    fn multipliers_avoid_degenerate_lattices() {
        for &(n, m) in PRIMES.iter() {
            assert!(m >= 2 && m <= n - 2, "({n}, {m})");
            assert!((2..n).all(|d| n % d != 0 || d * d > n) || is_prime(n), "{n} must be prime");
        }
        fn is_prime(n: u32) -> bool {
            n >= 2 && (2..).take_while(|d| d * d <= n).all(|d| n % d != 0)
        }
        assert!(PRIMES.iter().all(|&(n, _)| is_prime(n)));
    }
}
