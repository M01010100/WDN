import os
import time
import math
import numpy as np
from itertools import combinations
from collections import defaultdict

import torch
import torch.nn as nn
from scipy.linalg import hadamard
from scipy.stats import spearmanr


def walsh_hadamard_transform(f_values):
    n_points = len(f_values)
    n_bits   = int(np.log2(n_points))
    assert 2**n_bits == n_points, "Input length must be a power of 2"

    H    = f_values.copy().astype(np.float64)
    step = 1
    while step < n_points:
        for i in range(0, n_points, step * 2):
            for j in range(i, i + step):
                u, v = H[j], H[j + step]
                H[j] = u + v
                H[j + step] = u - v
        step *= 2

    return H


def index_to_subset(idx, n_bits):
    return frozenset(i for i in range(n_bits) if (idx >> i) & 1)


def subset_to_index(S, n_bits):
    # frozenset S → integer index
    return sum(1 << i for i in S)


def compute_algebraic_degree(walsh_coeffs, threshold=1e-3):
    # Effective algebraic degree from Walsh spectrum.
    n_bits = int(np.log2(len(walsh_coeffs)))
    max_degree = 0
    sig_terms = {}

    for idx in range(len(walsh_coeffs)):
        if abs(walsh_coeffs[idx]) > threshold * len(walsh_coeffs):
            S = index_to_subset(idx, n_bits)
            max_degree = max(max_degree, len(S))
            sig_terms[S] = walsh_coeffs[idx]

    return max_degree, sig_terms


class WalshBasisFunction(nn.Module):
    # Fixed (non-trainable) Walsh character χ_S(x) = (-1)^{popcount(S & x)}.
    def __init__(self, S_bits, n_bits):
        super().__init__()
        self.n_bits = n_bits
        mask = torch.zeros(n_bits, dtype=torch.float32)
        for i in S_bits:
            mask[i] = 1.0
        self.register_buffer('mask', mask)

    def forward(self, x):
        # x : (batch, n_bits) binary tensor in {0, 1}
        # Returns ±1 per sample.
        inner = (x * self.mask).sum(dim=-1)
        parity = (inner % 2).long()
        return (1 - 2 * parity).float()


class WalshDecompositionNetwork(nn.Module):
    #Explicitly learns the Walsh spectrum of the distinguishing function.
    def __init__(self, n_bits, d_max=4, init_sparse=True):
        super().__init__()
        self.n_bits = n_bits
        self.d_max = d_max

        # Build all subsets |S| <= d_max
        self.subsets = []
        for degree in range(d_max + 1):
            for S in combinations(range(n_bits), degree):
                self.subsets.append(list(S))

        n_terms = len(self.subsets)
        print(f"WDN: {n_bits} bits, max degree {d_max}, "
              f"{n_terms} Walsh basis functions")

        self.basis_functions = nn.ModuleList([
            WalshBasisFunction(S, n_bits) for S in self.subsets
        ])

        # Canonical trainable Walsh coefficients
        self.coefficients = nn.Parameter(
            torch.zeros(n_terms) if init_sparse
            else torch.randn(n_terms) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(1))

    # forward / inference
    def forward(self, x):
        basis_vals = torch.stack(
            [bf(x) for bf in self.basis_functions], dim=1)  # (batch, n_terms)
        return (basis_vals * self.coefficients).sum(dim=1) + self.bias

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        with torch.no_grad():
            return self.forward(x).numpy()

    def predict_binary(self, x, threshold=0.0):
        return (self.predict(x) > threshold).astype(int)

    def accuracy(self, x, y, threshold=0.0):
        y_pred = self.predict_binary(x, threshold)
        y_true = np.asarray(y).astype(int)
        return np.mean(y_pred == y_true)

    # Spectrum extraction / analysis

    def get_spectrum(self):
        spectrum = {}
        coeffs = self.coefficients.detach().numpy()
        for i, S in enumerate(self.subsets):
            if abs(coeffs[i]) > 1e-4:
                spectrum[frozenset(S)] = float(coeffs[i])
        return spectrum

    def get_degree_profile(self):
        coeffs = self.coefficients.detach().numpy()
        profile = {}
        for i, S in enumerate(self.subsets):
            d = len(S)
            profile[d] = profile.get(d, 0.0) + coeffs[i] ** 2
        total = sum(profile.values()) + 1e-10
        return {d: v / total for d, v in profile.items()}

    def get_significant_bits(self, top_k=10):
        coeffs = self.coefficients.detach().numpy()
        bit_importance = np.zeros(self.n_bits)
        for i, S in enumerate(self.subsets):
            for bit in S:
                bit_importance[bit] += coeffs[i] ** 2
        top_bits = np.argsort(bit_importance)[::-1][:top_k]
        return top_bits, bit_importance


# Model Persistence 

def save_wdn(model, path, metadata=None):
    checkpoint = {
        'n_bits'    : model.n_bits,
        'd_max'     : model.d_max,
        'state_dict': model.state_dict(),
        'metadata'  : metadata or {},
    }
    torch.save(checkpoint, path)
    size_kb = os.path.getsize(path) / 1024
    print(f"Saved WDN → {path}  ({size_kb:.1f} KB)")


def load_wdn(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model      = WalshDecompositionNetwork(
        n_bits     = checkpoint['n_bits'],
        d_max      = checkpoint['d_max'],
        init_sparse= True,
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    metadata = checkpoint.get('metadata', {})
    print(f"Loaded WDN ← {path}  "
          f"(n_bits={model.n_bits}, d_max={model.d_max}, meta={metadata})")
    return model, metadata


# Classical WHT Distinguisher

class ClassicalWHTDistinguisher:
    # Empirical Walsh coefficient estimator.

    def __init__(self, n_bits, d_max=3):
        self.n_bits     = n_bits
        self.d_max      = d_max
        self.spectrum_  = {}
        self.variances_ = {}
        self.n_samples_ = 0


    def fit(self, X, y):
        #Estimate Walsh coefficients from labelled data.
        # X : (N, n_bits) int8 array in {0,1}
        # y : (N,) labels in {+1,−1} or {0,1}

        X = np.asarray(X, dtype=np.int8)
        y = np.asarray(y, dtype=np.float32)
        if set(np.unique(y)) <= {0, 1}:
            y = 2 * y - 1.0

        self.n_samples_ = len(X)

        for degree in range(self.d_max + 1):
            for S in combinations(range(self.n_bits), degree):
                chi = self._walsh_char(X, S)
                coeff = np.mean(y * chi)
                variance = np.var(y * chi) / len(y)
                self.spectrum_[frozenset(S)] = float(coeff)
                self.variances_[frozenset(S)] = float(variance)

        return self

    def _walsh_char(self, X, S):
        if len(S) == 0:
            return np.ones(len(X), dtype=np.float32)
        parity = X[:, list(S)].sum(axis=1) % 2
        return (1 - 2 * parity).astype(np.float32)

    def get_spectrum(self, threshold=1e-3):
        return {S: c for S, c in self.spectrum_.items() if abs(c) > threshold}

    def get_degree_profile(self):
        profile = defaultdict(float)
        for S, c in self.spectrum_.items():
            profile[len(S)] += c ** 2
        total = sum(profile.values()) + 1e-10
        return {d: v / total for d, v in sorted(profile.items())}

    def get_significant_bits(self, top_k=10):
        bit_importance = np.zeros(self.n_bits)
        for S, c in self.spectrum_.items():
            for bit in S:
                bit_importance[bit] += c ** 2
        top_bits = np.argsort(bit_importance)[::-1][:top_k]
        return top_bits, bit_importance

    def predict(self, X):
        X      = np.asarray(X, dtype=np.int8)
        scores = np.zeros(len(X))
        for S, c in self.spectrum_.items():
            if abs(c) < 1e-6:
                continue
            scores += c * self._walsh_char(X, list(S))
        return scores

    def predict_binary(self, X, threshold=0.0):
        return (self.predict(X) > threshold).astype(int)

    def accuracy(self, X, y):
        y = np.asarray(y)
        y_binary = (y > 0).astype(int) if set(np.unique(y)) <= {-1, 1} else y
        return np.mean(self.predict_binary(X) == y_binary)

# Spectrum Comparison & Convergence Utilities

def compare_spectra(spectrum_classical, spectrum_wdn, label=""):
    # Quantitative comparison of a classical WHT spectrum vs a WDN spectrum.

    all_terms = set(spectrum_classical) | set(spectrum_wdn)
    c_vals = np.array([spectrum_classical.get(t, 0.0) for t in all_terms])
    w_vals = np.array([spectrum_wdn.get(t, 0.0)       for t in all_terms])

    denom = (np.linalg.norm(c_vals) * np.linalg.norm(w_vals)) + 1e-10
    cosine_sim = float(np.dot(c_vals, w_vals) / denom)
    l1_error = float(np.mean(np.abs(c_vals - w_vals)))

    sig_c = set(t for t, c in spectrum_classical.items() if abs(c) > 1e-3)
    sig_w = set(t for t, c in spectrum_wdn.items()       if abs(c) > 1e-3)
    overlap = (len(sig_c & sig_w) / len(sig_c | sig_w)) if (sig_c | sig_w) else 0.0

    # Guard against ConstantInputWarning: spearmanr is undefined when either
    # array has zero variance (e.g. one spectrum is empty → all-zeros array).
    _a, _b = np.abs(c_vals), np.abs(w_vals)
    if len(_a) > 2 and np.std(_a) > 0 and np.std(_b) > 0:
        rho = float(spearmanr(_a, _b)[0])
    else:
        rho = float('nan')

    results = {
        'cosine_similarity': cosine_sim, 'l1_error': l1_error,
        'term_overlap': overlap, 'rank_correlation': rho,
        'n_terms_classical': len(sig_c), 'n_terms_wdn': len(sig_w),
        'n_terms_shared': len(sig_c & sig_w),
    }

    if label:
        print(f"\n=== Spectrum Comparison: {label} ===")
        for k, v in results.items():
            print(f"  {k:<30}: {v}")

    return results


def spectrum_convergence_test(X, y, n_bits, d_max=3, sample_sizes=None):
    if sample_sizes is None:
        N = len(X)
        sample_sizes = [N // 16, N // 8, N // 4, N // 2, N]

    print("=== Classical WHT Convergence Test ===")
    spectra = {}

    for N in sample_sizes:
        idx = np.random.choice(len(X), size=min(N, len(X)), replace=False)
        clf = ClassicalWHTDistinguisher(n_bits, d_max)
        clf.fit(X[idx], y[idx])
        spectra[N] = clf.get_spectrum(threshold=1e-3)

        sig = clf.get_spectrum(threshold=1e-3)
        profile = clf.get_degree_profile()
        print(f"  N={N:>8}: {len(sig)} significant terms, "
              f"degree profile: { {d: f'{v:.3f}' for d, v in profile.items()} }")

    print("\nConvergence (L1 distance between successive sizes):")
    sizes = sorted(spectra.keys())
    for i in range(1, len(sizes)):
        all_t = set(spectra[sizes[i-1]]) | set(spectra[sizes[i]])
        l1 = np.mean([abs(spectra[sizes[i]].get(t, 0) -
                             spectra[sizes[i-1]].get(t, 0)) for t in all_t])
        print(f"  N={sizes[i-1]} → N={sizes[i]}: L1={l1:.6f}")

    return spectra


#  Canonicity Stability Tests (empirical)

def bootstrap_coefficient_ci(X, y, target_subset, n_bootstrap=200, confidence=0.95):
    #Bootstrap CI for a single Walsh coefficient f̂(S).
    # Returns (estimate, lower_ci, upper_ci, cv).
    X = np.asarray(X, dtype=np.int8)
    y = np.asarray(y, dtype=np.float32)
    if set(np.unique(y)) <= {0, 1}:
        y = 2 * y - 1.0

    S = list(target_subset)
    estimates = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        Xb, yb = X[idx], y[idx]
        if len(S) == 0:
            chi = np.ones(len(Xb))
        else:
            chi = (1 - 2 * (Xb[:, S].sum(axis=1) % 2)).astype(np.float32)
        estimates.append(float(np.mean(yb * chi)))

    estimates = np.array(estimates)
    alpha = 1 - confidence
    lower = np.percentile(estimates, 100 * alpha / 2)
    upper = np.percentile(estimates, 100 * (1 - alpha / 2))
    cv = np.std(estimates) / (abs(np.mean(estimates)) + 1e-10)

    return float(np.mean(estimates)), lower, upper, cv


def empirical_canonicity_test(X_train, y_train, X_test, y_test,
                               n_bits, d_max=3, n_bootstrap=100):
    # Three-condition empirical canonicity test.

    print("\nEMPIRICAL CANONICITY TEST\n")

    print("\n--- Condition 1: Spectrum Convergence ---")
    spectrum_convergence_test(X_train, y_train, n_bits, d_max)

    print("\n--- Condition 2: Full Spectrum Estimation ---")
    clf = ClassicalWHTDistinguisher(n_bits, d_max)
    clf.fit(X_train, y_train)
    classical_spectrum = clf.get_spectrum(threshold=1e-3)
    print(f"Classical WHT: {len(classical_spectrum)} significant terms")
    print(f"Degree profile: {clf.get_degree_profile()}")
    print(f"Training accuracy: {clf.accuracy(X_train, y_train):.4f}")
    print(f"Test accuracy:     {clf.accuracy(X_test,  y_test):.4f}")

    print(f"\n--- Condition 3: Bootstrap CIs (n={n_bootstrap}) ---")
    sorted_terms = sorted(classical_spectrum.items(),
                           key=lambda kv: abs(kv[1]), reverse=True)[:10]
    stable_terms = []

    for S, coeff in sorted_terms:
        est, lo, hi, cv = bootstrap_coefficient_ci(
            X_train, y_train, S, n_bootstrap=n_bootstrap
        )
        is_stable = cv < 0.2 and (abs(est) > 3 * np.sqrt(clf.variances_.get(S, 1e-6)))
        if is_stable:
            stable_terms.append(S)
        bits = sorted(S)
        print(f"  χ_{{{','.join(map(str, bits))}}}: "
              f"est={est:.4f}, 95%CI=[{lo:.4f}, {hi:.4f}], "
              f"CV={cv:.3f}, stable={'YES' if is_stable else 'NO'}")

    print(f"\nConclusion: {len(stable_terms)}/{len(sorted_terms)} top terms are stable.")
    return clf, classical_spectrum, stable_terms


def classical_vs_wdn_demo(X, y, wdn_model, n_bits, d_max=3):
    print("\nCLASSICAL WHT vs WDN COMPARISON\n")

    clf = ClassicalWHTDistinguisher(n_bits, d_max)
    clf.fit(X, y)
    classical_spectrum = clf.get_spectrum(threshold=1e-3)
    wdn_spectrum = wdn_model.get_spectrum()

    print(f"\nClassical WHT: {len(classical_spectrum)} terms, "
          f"profile={clf.get_degree_profile()}")
    print(f"WDN:{len(wdn_spectrum)} terms, "
          f"profile={wdn_model.get_degree_profile()}")

    metrics = compare_spectra(classical_spectrum, wdn_spectrum,
                               label="Classical WHT vs WDN")

    top_bits_c, _    = clf.get_significant_bits(top_k=8)
    # When the WDN was trained on more bits than the classical fit, restrict
    # the importance comparison to the same bit range (0…n_bits-1) so the
    _, wdn_importance = wdn_model.get_significant_bits(top_k=wdn_model.n_bits)
    top_bits_w = np.argsort(wdn_importance[:n_bits])[::-1][:8]
    overlap_bits = len(set(top_bits_c.tolist()) & set(top_bits_w.tolist()))
    print("\nTop 8 bits by importance (restricted to bits 0–{n_bits-1}):")
    print(f"  Classical: {list(top_bits_c)}")
    print(f"  WDN:       {list(top_bits_w)}")
    print(f"  Shared:    {overlap_bits}/8")

    return metrics, clf


# SIMON vs SPECK Validation Experiment

def compare_simon_vs_speck(n_bits=64, d_max=3, n_samples=200000, use_cache=True,
                            model_save_dir='.'):
    """
    SIMON (linear over GF(2)) → expect degree-1 spectrum only.
    SPECK (modular addition)   → expect degree-2 and degree-3 terms.
    """
    from cipher_generation import train_wdn

    results = {}

    print("=== SIMON: Expect degree-1 spectrum (linear cipher) ===")
    if use_cache:
        try:
            from cache_datasets import load_dataset
            X_simon, y_simon = load_dataset(f'simon_8rounds_{n_samples//1000}k')
            print("Loaded cached SIMON dataset")
        except (ImportError, FileNotFoundError):
            print("Cache unavailable — generating on-the-fly")
            from cipher_generation import make_dataset
            X_simon, y_simon = make_dataset(n_samples, cipher='simon', rounds=8,
                                             input_diff=0x00400000)
    else:
        from cipher_generation import make_dataset
        X_simon, y_simon = make_dataset(n_samples, cipher='simon', rounds=8,
                                         input_diff=0x00400000)

    model_simon = WalshDecompositionNetwork(n_bits, d_max)
    model_simon, hist_simon = train_wdn(model_simon, X_simon, y_simon,
                                        n_epochs=55, verbose=True)
    save_wdn(model_simon,
             os.path.join(model_save_dir, 'wdn_simon.pt'),
             {'cipher': 'simon', 'rounds': 8, 'n_samples': n_samples,
              'n_bits': n_bits, 'd_max': d_max})

    spectrum_simon = model_simon.get_spectrum()
    profile_simon  = model_simon.get_degree_profile()
    print(f"\nSIMON Degree Profile : {profile_simon}")
    print(f"SIMON Spectrum       : {len(spectrum_simon)} significant terms")
    results['simon'] = {'spectrum': spectrum_simon, 'profile': profile_simon,
                        'model': model_simon}

    print("\n=== SPECK: Expect higher-degree spectrum (nonlinear) ===")
    if use_cache:
        try:
            from cache_datasets import load_dataset
            X_speck, y_speck = load_dataset(f'speck_7rounds_{n_samples//1000}k')
            print("Loaded cached SPECK dataset")
        except (ImportError, FileNotFoundError):
            print("Cache unavailable — generating on-the-fly")
            from cipher_generation import make_dataset
            X_speck, y_speck = make_dataset(n_samples, cipher='speck', rounds=7,
                                             input_diff=0x00400000)
    else:
        from cipher_generation import make_dataset
        X_speck, y_speck = make_dataset(n_samples, cipher='speck', rounds=7,
                                         input_diff=0x00400000)

    model_speck = WalshDecompositionNetwork(n_bits, d_max)
    model_speck, hist_speck = train_wdn(model_speck, X_speck, y_speck,
                                        n_epochs=55, verbose=True)
    save_wdn(model_speck,
             os.path.join(model_save_dir, 'wdn_speck.pt'),
             {'cipher': 'speck', 'rounds': 7, 'n_samples': n_samples,
              'n_bits': n_bits, 'd_max': d_max})

    spectrum_speck = model_speck.get_spectrum()
    profile_speck = model_speck.get_degree_profile()
    print(f"\nSPECK Degree Profile : {profile_speck}")
    print(f"SPECK Spectrum       : {len(spectrum_speck)} significant terms")
    results['speck'] = {'spectrum': spectrum_speck, 'profile': profile_speck,
                        'model': model_speck}

    print("\n=== RESIDUAL: degree-3 terms unique to SPECK ===")
    speck_terms = set(spectrum_speck)
    simon_terms = set(spectrum_simon)
    residual_terms = speck_terms - simon_terms
    residual_high = {t: spectrum_speck[t] for t in residual_terms if len(t) >= 3}

    print(f"Terms unique to SPECK          : {len(residual_terms)}")
    print(f"High-order terms (degree >= 3) : {len(residual_high)}")

    if residual_high:
        print("\nTop 5 high-order residual terms:")
        for term, coeff in sorted(residual_high.items(),
                                   key=lambda x: abs(x[1]), reverse=True)[:5]:
            print(f"  χ_{{{','.join(map(str, sorted(term)))}}}: "
                  f"coeff={coeff:.4f}, degree={len(term)}")

    simon_unique = simon_terms - speck_terms
    simon_unique_high = {t: spectrum_simon[t] for t in simon_unique if len(t) >= 3}
    print(f"\nTerms unique to SIMON          : {len(simon_unique)}")
    print(f"High-order terms unique to SIMON: {len(simon_unique_high)}")

    results['residual'] = {'all_terms': residual_terms, 'high_order': residual_high}
    return results


# Cipher Symmetry Orbit Decomposition

def decompose_by_cipher_symmetry(spectrum, cipher_symmetry_group):
    #Decompose Walsh spectrum into orbits under the cipher's symmetry group.
    orbit_spectrum = {}
    assigned = set()

    for term, coeff in spectrum.items():
        if term in assigned:
            continue
        orbit = set()
        for g in cipher_symmetry_group:
            rotated = frozenset(g(bit) for bit in term)
            orbit.add(rotated)
            assigned.add(rotated)
        rep = min(orbit, key=lambda s: tuple(sorted(s)))
        total_power = sum(abs(spectrum.get(t, 0)) ** 2 for t in orbit)
        orbit_spectrum[rep] = (total_power ** 0.5, len(orbit))

    return orbit_spectrum


def speck_rotation_group(n_half=16):
    #Rotational symmetry group of SPECK (acts on each 16-bit half independently).
    group = []
    for r in range(n_half):
        def make_rotation(rotation):
            def rotate(bit_idx):
                if bit_idx < n_half:
                    return (bit_idx + rotation) % n_half
                else:
                    return n_half + (bit_idx - n_half + rotation) % n_half
            return rotate
        group.append(make_rotation(r))
    return group


# Spectrum Stability Analyzer (DistinguisherTheory)

class SpectrumStabilityAnalyzer:
    # Train N copies of the WDN with different random seeds;
    # measure agreement between their spectra.

    def __init__(self, n_bits, d_max=3, n_runs=10):
        self.n_bits = n_bits
        self.d_max = d_max
        self.n_runs = n_runs
        self.spectra = []

    def run(self, X_train, y_train, n_epochs=20):
        for seed in range(self.n_runs):
            torch.manual_seed(seed)
            model = WalshDecompositionNetwork(self.n_bits, self.d_max)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            dataset = torch.utils.data.TensorDataset(X_train, y_train)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=4096, shuffle=True)

            for epoch in range(n_epochs):
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    scores = model(X_batch)
                    real_mask = y_batch == 1
                    rand_mask = y_batch == 0
                    loss = (-scores[real_mask].mean() +
                            torch.logsumexp(scores[rand_mask], 0) -
                            np.log(rand_mask.sum().item()))
                    loss.backward()
                    optimizer.step()

            spectrum = model.get_spectrum()
            self.spectra.append(spectrum)
            print(f"Seed {seed}: {len(spectrum)} significant terms, "
                  f"degree profile: {model.get_degree_profile()}")

        return self.analyze_stability()

    def analyze_stability(self):
        all_terms = set()
        for spectrum in self.spectra:
            all_terms.update(spectrum.keys())

        stability = {}
        for term in all_terms:
            vals = [s.get(term, 0.0) for s in self.spectra]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            cv = std_val / (abs(mean_val) + 1e-10)
            stability[term] = {
                'mean': mean_val, 'std': std_val,
                'coeff_variation': cv,
                'appearance_rate': sum(1 for v in vals if abs(v) > 1e-4) / self.n_runs,
            }

        stable_terms = {t: v for t, v in stability.items()
                        if v['appearance_rate'] > 0.8 and v['coeff_variation'] < 0.2}

        print(f"\n=== STABILITY ANALYSIS ===")
        print(f"Total unique terms : {len(all_terms)}")
        print(f"Stable terms       : {len(stable_terms)}")

        for term in sorted(stable_terms, key=lambda t: (len(t), -abs(stable_terms[t]['mean']))):
            s = stable_terms[term]
            bits = sorted(term)
            print(f"  χ_{{{','.join(map(str, bits))}}}: "
                  f"mean={s['mean']:.4f}, std={s['std']:.4f}, "
                  f"degree={len(term)}, CV={s['coeff_variation']:.3f}")

        return stable_terms


# MAIN: 

if __name__ == '__main__':

    # Smoke test (no external dependencies)
    print("\nSMOKE TEST — synthetic XOR distinguisher \n")
    np.random.seed(42)
    N_SMOKE, N_BITS_SMOKE = 50_000, 16

    X_smoke = np.random.randint(0, 2, (N_SMOKE, N_BITS_SMOKE), dtype=np.int8)
    parity = (X_smoke[:, 0] + X_smoke[:, 1]) % 2
    y_smoke = (1 - 2 * parity).astype(np.float32)

    clf_smoke = ClassicalWHTDistinguisher(N_BITS_SMOKE, d_max=3)
    clf_smoke.fit(X_smoke, y_smoke)
    sig_smoke = clf_smoke.get_spectrum(threshold=0.1)
    print(f"Significant terms (threshold 0.1): {len(sig_smoke)}")
    for S, c in sorted(sig_smoke.items(), key=lambda kv: abs(kv[1]), reverse=True):
        print(f"  χ_{{{','.join(map(str, sorted(S)))}}}: coeff={c:.4f}")
    print(f"Degree profile: {clf_smoke.get_degree_profile()}")
    ok = frozenset({0, 1}) in sig_smoke
    print("Smoke test PASSED." if ok else "WARNING: expected term {0,1} not found")


    N_BITS = 64
    D_MAX = 3
    N_SAMPLES = 200_000
    SAVE_DIR = '.'

    print("\nWalsh Decomposition Network: Canonical Form Recovery\n")

    try:
        from cipher_generation import make_dataset, train_wdn
    except ImportError:
        print("\ncipher_generation not found — skipping full pipeline.\n"
              "Provide cipher_generation.py to run Experiments 1–4.")
        raise SystemExit(0)

    print("\nEXPERIMENT 1: SIMON vs SPECK Comparison")
    print("Models will be saved to wdn_simon.pt and wdn_speck.pt")
    print("for reuse in ClassicalWHT experiments.\n")

    results = compare_simon_vs_speck(N_BITS, D_MAX, N_SAMPLES,
                                     model_save_dir=SAVE_DIR)

    print("\nEXPERIMENT 2: Spectrum Stability Across Random Seeds")

    speck_model_path = os.path.join(SAVE_DIR, 'wdn_speck.pt')
    if os.path.exists(speck_model_path):
        print(f"Found saved SPECK model at {speck_model_path}; reusing for stability test.")
        # Dataset still needed for training the stability runs
    try:
        from cache_datasets import load_dataset
        X_stab, y_stab = load_dataset(f'speck_7rounds_{N_SAMPLES//2//1000}k')
        print("✓ Loaded cached dataset for stability test")
    except (ImportError, FileNotFoundError):
        X_stab, y_stab = make_dataset(N_SAMPLES // 2, cipher='speck', rounds=7)

    analyzer = SpectrumStabilityAnalyzer(N_BITS, D_MAX, n_runs=5)
    stable_terms = analyzer.run(X_stab, y_stab, n_epochs=15)
    print(f"\nFound {len(stable_terms)} stable terms across all runs")

    print("\nEXPERIMENT 3: Advantage Bit Recovery")
    model_speck = results['speck']['model']
    top_bits, bit_importance = model_speck.get_significant_bits(top_k=16)

    print("Top 16 significant bit positions:")
    for i, bit in enumerate(top_bits):
        half = "Left" if bit < 32 else "Right"
        pos = bit % 32
        print(f"  {i+1:>2}. Bit {bit:>2} ({half} half, position {pos:>2}): "
              f"importance={bit_importance[bit]:.4f}")

    print("\nEXPERIMENT 4: Cipher Symmetry Orbit Decomposition")
    spectrum_speck = results['speck']['spectrum']
    sym_group = speck_rotation_group(n_half=16)
    orbit_spec = decompose_by_cipher_symmetry(spectrum_speck, sym_group)

    print(f"Original spectrum   : {len(spectrum_speck)} terms")
    print(f"Orbit spectrum      : {len(orbit_spec)} orbit representatives")
    if len(orbit_spec):
        print(f"Compression ratio   : {len(spectrum_speck)/len(orbit_spec):.2f}×\n")
    for rep, (power, size) in sorted(orbit_spec.items(),
                                      key=lambda x: x[1][0], reverse=True)[:5]:
        print(f"  χ_{{{','.join(map(str, sorted(rep)))}}}: "
              f"power={power:.4f}, orbit_size={size}, degree={len(rep)}")


    # Load the saved SPECK WDN (n_bits=64).
    model_speck_saved, meta = load_wdn(speck_model_path)

    X16 = X_stab[:50_000, :16]
    y16 = y_stab[:50_000]
    metrics, clf16 = classical_vs_wdn_demo(
        X16, y16,
        wdn_model=model_speck_saved,   
        n_bits=16, d_max=3
    )

    full_saved_spectrum    = model_speck_saved.get_spectrum()
    projected_wdn_spectrum = {
        S: c for S, c in full_saved_spectrum.items()
        if all(b < 16 for b in S)
    }
    classical16_spec = clf16.get_spectrum(threshold=1e-3)

    # L2-normalise the projected WDN spectrum to match classical scale.
    proj_l2 = np.sqrt(sum(c ** 2 for c in projected_wdn_spectrum.values())) + 1e-10
    class_l2 = np.sqrt(sum(c ** 2 for c in classical16_spec.values())) + 1e-10
    scale = class_l2 / proj_l2
    projected_wdn_normalised = {S: c * scale for S, c in projected_wdn_spectrum.items()}

    print(f"\nProjected WDN spectrum (bits 0–15 only) : {len(projected_wdn_spectrum)} terms"
          f"  (from {len(full_saved_spectrum)} total in saved model)")
    print(f"Normalisation scale applied to WDN      : {scale:.4f}  "
          f"(classical L2={class_l2:.4f}, projected WDN L2={proj_l2:.6f})")

    print("\nProjected + normalised spectrum comparison "
          "(16-bit Classical WHT vs WDN bits 0–15, rescaled):")
    cmp = compare_spectra(classical16_spec, projected_wdn_normalised,
                          label="16-bit Classical WHT vs Projected+Normalised WDN")

    # Interpret the three key metrics for the reader.
    print("\n--- Interpretation ---")
    print(f"  term_overlap     = {cmp['term_overlap']:.3f}  "
          f"({'HIGH' if cmp['term_overlap'] > 0.7 else 'LOW'}) — "
          "fraction of significant terms identified by BOTH estimators")
    print(f"  cosine_similarity= {cmp['cosine_similarity']:.4f}  "
          f"({'HIGH' if abs(cmp['cosine_similarity']) > 0.5 else 'LOW'}) — "
          "coefficient direction/magnitude agreement after normalisation")
    rho = cmp['rank_correlation']
    rho_str = f"{rho:.4f}" if not np.isnan(rho) else "nan"
    print(f"  rank_correlation = {rho_str}  "
          f"({'HIGH' if not np.isnan(rho) and abs(rho) > 0.5 else 'LOW'}) — "
          "Spearman rank agreement on coefficient magnitudes")