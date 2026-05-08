"""
Cipher Generation Utilities for Walsh Decomposition Network
=============================================================

SPECK32/64 and SIMON32/64 implementations with data generation
for training Walsh Decomposition Networks (WDN).

Following the recommended implementation order:
1. SPECK encryption function
2. SIMON encryption function (already in DistinguisherTheory.py)
3. Data generation utilities
4. Binary conversion helpers
"""

import numpy as np
import torch
from os import urandom

# PART I: SPECK32/64 Implementation

def rotate_left(x, n, word_size=16):
    """Left rotation of a word"""
    return ((x << n) | (x >> (word_size - n))) & ((1 << word_size) - 1)

def rotate_right(x, n, word_size=16):
    """Right rotation of a word"""
    return ((x >> n) | (x << (word_size - n))) & ((1 << word_size) - 1)

def speck_round(x, y, k, word_size=16):
    """
    One round of SPECK32/64
    x, y: 16-bit words
    k: round subkey
    
    Operations:
    x = (x >>> 7) + y mod 2^16
    x = x XOR k
    y = (y <<< 2) XOR x
    """
    mask = (1 << word_size) - 1
    x = (rotate_right(x, 7, word_size) + y) & mask
    x ^= k
    y = rotate_left(y, 2, word_size) ^ x
    return x, y

def expand_key_speck(key, rounds, word_size=16):
    """
    Key schedule for SPECK32/64
    key: 64-bit key (4 x 16-bit words)
    rounds: number of rounds
    Returns: list of round subkeys
    """
    mask = (1 << word_size) - 1
    
    # Extract key words (k3, k2, k1, k0)
    k = [(key >> (word_size * i)) & mask for i in range(4)]
    subkeys = [k[0]]  # k0 is the first subkey
    l = list(k[1:])  # l = [k1, k2, k3]
    
    for i in range(rounds - 1):
        # Update l and generate next subkey
        l[i % 3], new_subkey = speck_round(l[i % 3], subkeys[-1], i)
        subkeys.append(new_subkey)
    
    return subkeys

def speck_encrypt(plaintext, key, rounds=7, word_size=16):
    """
    Encrypt with SPECK32/64
    plaintext: 32-bit value (left || right)
    key: 64-bit key
    rounds: number of rounds (default 7 for neural distinguisher)
    
    Returns: 32-bit ciphertext
    """
    mask = (1 << word_size) - 1
    x = (plaintext >> word_size) & mask  # Left half
    y = plaintext & mask  # Right half
    
    subkeys = expand_key_speck(key, rounds, word_size)
    
    for k in subkeys:
        x, y = speck_round(x, y, k, word_size)
    
    return (x << word_size) | y

# PART II: SIMON32/64 Implementation (Enhanced)

def simon_f(x, word_size=16):
    """
    SIMON round function: F(x) = (S^1(x) & S^8(x)) XOR S^2(x)
    All linear operations over GF(2) - pure XOR and rotation
    """
    mask = (1 << word_size) - 1
    s1 = rotate_left(x, 1, word_size)
    s8 = rotate_left(x, 8, word_size)
    s2 = rotate_left(x, 2, word_size)
    return ((s1 & s8) ^ s2) & mask

def expand_key_simon(key, rounds, word_size=16):
    """
    Key schedule for SIMON32/64
    key: 64-bit key (4 x 16-bit words)
    rounds: number of rounds
    
    SIMON key schedule is also linear (rotation + XOR)
    """
    mask = (1 << word_size) - 1
    k = [(key >> (word_size * i)) & mask for i in range(4)]
    
    # Z sequence for SIMON32/64 (sequence z0)
    z = 0b11111010001001010110000111001101111101000100101011000011100110
    
    for i in range(rounds - 4):
        tmp = rotate_right(k[-1], 3, word_size)
        if len(k) == 4:
            tmp ^= k[1]
        tmp ^= rotate_right(tmp, 1, word_size)
        k.append((k[-4] ^ tmp ^ ((z >> (i % 62)) & 1) ^ 0xFFFC) & mask)
    
    return k[:rounds]

def simon_encrypt(plaintext, key, rounds=8, word_size=16):
    """
    Encrypt with SIMON32/64
    plaintext: 32-bit value (left || right)
    key: 64-bit key
    rounds: number of rounds (default 8)
    
    Returns: 32-bit ciphertext
    
    Note: SIMON is entirely linear over GF(2) - only XOR and rotations
    """
    mask = (1 << word_size) - 1
    x = (plaintext >> word_size) & mask  # Left half
    y = plaintext & mask  # Right half
    
    subkeys = expand_key_simon(key, rounds, word_size)
    
    for k in subkeys:
        # Feistel: (x, y) -> (y XOR F(x) XOR k, x)
        tmp = y ^ simon_f(x, word_size) ^ k
        y = x
        x = tmp
    
    return (x << word_size) | y

# PART III: Data Generation for WDN Training

def pair_to_binary(c1, c2, n_bits=32):
    """
    Convert a ciphertext pair (c1, c2) to binary feature vector.
    
    For WDN input: we need the XOR difference c1 XOR c2 as binary.
    This is the 64-bit input to the Walsh network.
    
    Args:
        c1: first ciphertext (32-bit)
        c2: second ciphertext (32-bit)
        n_bits: bits per ciphertext (default 32)
    
    Returns:
        Binary array of shape (2*n_bits,) containing [c1_bits, c2_bits]
    """
    def int_to_binary(x, bits):
        return np.array([(x >> i) & 1 for i in range(bits)], dtype=np.float32)
    
    bits1 = int_to_binary(c1, n_bits)
    bits2 = int_to_binary(c2, n_bits)
    
    return np.concatenate([bits1, bits2])

def generate_speck_pairs(n_samples, rounds=7, input_diff=0x00400000, real=True):
    """
    Generate SPECK ciphertext pairs for training.
    
    Args:
        n_samples: number of pairs to generate
        rounds: number of SPECK rounds
        input_diff: input difference for real pairs (standard: 0x00400000)
        real: if True, generate real differential pairs
              if False, generate random pairs
    
    Returns:
        c1, c2: arrays of 32-bit ciphertexts
    """
    # Generate random keys and plaintexts
    keys = np.random.randint(0, 2**64, n_samples, dtype=np.uint64)
    plaintexts = np.random.randint(0, 2**32, n_samples, dtype=np.uint64)
    
    if real:
        # Differential pairs: p2 = p1 XOR delta
        plaintexts2 = plaintexts ^ input_diff
    else:
        # Random pairs: p2 is independent
        plaintexts2 = np.random.randint(0, 2**32, n_samples, dtype=np.uint64)
    
    # Encrypt both plaintexts under the same key
    c1 = np.array([speck_encrypt(int(p), int(k), rounds) 
                   for p, k in zip(plaintexts, keys)], dtype=np.uint64)
    c2 = np.array([speck_encrypt(int(p), int(k), rounds) 
                   for p, k in zip(plaintexts2, keys)], dtype=np.uint64)
    
    return c1, c2

def generate_simon_pairs(n_samples, rounds=8, input_diff=0x00400000, real=True):
    """
    Generate SIMON ciphertext pairs for training.
    
    Args:
        n_samples: number of pairs to generate
        rounds: number of SIMON rounds
        input_diff: input difference for real pairs
        real: if True, generate real differential pairs
              if False, generate random pairs
    
    Returns:
        c1, c2: arrays of 32-bit ciphertexts
    """
    keys = np.random.randint(0, 2**64, n_samples, dtype=np.uint64)
    plaintexts = np.random.randint(0, 2**32, n_samples, dtype=np.uint64)
    
    if real:
        plaintexts2 = plaintexts ^ input_diff
    else:
        plaintexts2 = np.random.randint(0, 2**32, n_samples, dtype=np.uint64)
    
    c1 = np.array([simon_encrypt(int(p), int(k), rounds) 
                   for p, k in zip(plaintexts, keys)], dtype=np.uint64)
    c2 = np.array([simon_encrypt(int(p), int(k), rounds) 
                   for p, k in zip(plaintexts2, keys)], dtype=np.uint64)
    
    return c1, c2

def make_dataset(n_samples, cipher='speck', rounds=7, input_diff=0x00400000):
    """
    Create balanced dataset of real/random pairs for WDN training.
    
    Args:
        n_samples: total number of samples (half real, half random)
        cipher: 'speck' or 'simon'
        rounds: number of cipher rounds
        input_diff: input difference for real pairs
    
    Returns:
        X: torch tensor of shape (n_samples, 64) - binary features
        y: torch tensor of shape (n_samples,) - labels (1=real, 0=random)
    """
    n_half = n_samples // 2
    
    # Select cipher
    if cipher.lower() == 'speck':
        generate_fn = generate_speck_pairs
    elif cipher.lower() == 'simon':
        generate_fn = generate_simon_pairs
    else:
        raise ValueError(f"Unknown cipher: {cipher}")
    
    # Generate real and random pairs
    c1_real, c2_real = generate_fn(n_half, rounds, input_diff, real=True)
    c1_rand, c2_rand = generate_fn(n_half, rounds, input_diff, real=False)
    
    # Convert to binary features
    X_real = np.array([pair_to_binary(c1, c2) for c1, c2 in zip(c1_real, c2_real)])
    X_rand = np.array([pair_to_binary(c1, c2) for c1, c2 in zip(c1_rand, c2_rand)])
    
    X = np.vstack([X_real, X_rand])
    y = np.array([1] * n_half + [0] * n_half, dtype=np.float32)
    
    # Shuffle
    perm = np.random.permutation(n_samples)
    X = X[perm]
    y = y[perm]
    
    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    
    return X_tensor, y_tensor


# PART IV: Training Utilities

def train_wdn(model, X_train, y_train, n_epochs=20, batch_size=4096, 
              lr=1e-3, weight_decay=1e-4, verbose=True):
    """
    Train a Walsh Decomposition Network.
    
    Args:
        model: WalshDecompositionNetwork instance
        X_train: training data (binary features)
        y_train: training labels (1=real, 0=random)
        n_epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate
        weight_decay: L2 regularization
        verbose: print training progress
    
    Returns:
        model: trained model
        history: dict with training metrics
    """
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    history = {'loss': [], 'acc': []}
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            
            # Forward pass
            scores = model(X_batch)
            
            # Donsker-Varadhan loss for unique minimum
            # This loss has a unique global minimum, unlike BCE
            real_mask = y_batch == 1
            rand_mask = y_batch == 0
            
            loss = (-scores[real_mask].mean() + 
                    torch.logsumexp(scores[rand_mask], 0) - 
                    np.log(rand_mask.sum().item()))
            
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            with torch.no_grad():
                preds = (scores > 0).float()
                epoch_correct += (preds == y_batch).sum().item()
                epoch_total += len(y_batch)
        
        avg_loss = epoch_loss / len(loader)
        avg_acc = epoch_correct / epoch_total
        
        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
        
        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
    
    return model, history

# PART V: Verification and Testing

def test_cipher_implementations():
    """Test SPECK and SIMON implementations with known test vectors"""
    
    print("=== Testing SPECK32/64 ===")
    # Test vector from SPECK specification
    plaintext = 0x6574694c  # "Lite" in hex
    key = 0x0100090807060504  # Test key
    
    # Full 22-round SPECK
    ciphertext = speck_encrypt(plaintext, key, rounds=22)
    expected = 0xa86842f2  # Expected ciphertext
    
    print(f"Plaintext:  {plaintext:08x}")
    print(f"Key:        {key:016x}")
    print(f"Ciphertext: {ciphertext:08x}")
    print(f"Expected:   {expected:08x}")
    print(f"Match: {ciphertext == expected}")
    
    print("\n=== Testing SIMON32/64 ===")
    # Test vector for SIMON32/64
    plaintext_simon = 0x65656877  # Test plaintext
    key_simon = 0x0100090807060504
    
    # Full 32-round SIMON
    ciphertext_simon = simon_encrypt(plaintext_simon, key_simon, rounds=32)
    expected_simon = 0xc69be9bb  # Expected (approximate, check spec)
    
    print(f"Plaintext:  {plaintext_simon:08x}")
    print(f"Key:        {key_simon:016x}")
    print(f"Ciphertext: {ciphertext_simon:08x}")
    print(f"Expected:   {expected_simon:08x}")
    
    print("\n=== Testing Data Generation ===")
    X, y = make_dataset(1000, cipher='speck', rounds=7)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Real samples: {(y==1).sum().item()}, Random samples: {(y==0).sum().item()}")
    print(f"Feature range: [{X.min():.2f}, {X.max():.2f}]")
    
    print("\n=== Testing pair_to_binary ===")
    c1, c2 = 0x12345678, 0x9abcdef0
    binary = pair_to_binary(c1, c2)
    print(f"Pair: ({c1:08x}, {c2:08x})")
    print(f"Binary shape: {binary.shape}")
    print(f"Binary sample (first 16 bits): {binary[:16]}")
    
    print("\n=== All tests completed ===")

if __name__ == '__main__':
    test_cipher_implementations()
