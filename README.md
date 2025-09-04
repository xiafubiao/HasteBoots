# HasteBoots
This is the implementation of HasteBoots: Proving FHE Bootstrapping in Seconds. You can find the paper [here](https://eprint.iacr.org/2025/261.pdf).


Fully Homomorphic Encryption (FHE) enables computations on encrypted data, ensuring privacy for outsourced computation. However, verifying the integrity of FHE computations remains a significant challenge, especially for bootstrapping, the most computationally intensive operation in FHE. Prior approaches, including zkVM-based solutions and general-purpose SNARKs, suffer from inefficiencies, with proof generation times ranging from several hours to days. In this work, we propose HasteBoots, a succinct argument tailored for FHE operations. By designing customized polynomial interactive oracle proofs and optimized polynomial commitment schemes, HasteBoots achieves proof generation in a few seconds for FHE NAND with bootstrapping, significantly outperforming existing methods. Our approach demonstrates the potential for scalable and efficient verifiable FHE, paving the way for practical, privacy-preserving computations. 

Note the library has about 33K lines of Rust code. It includes PIOPs for all atomic and FHE operations, an optimized PCS based on Brakedown, and a framework for generating SNARGs tailored tp FHEW-like schemes.

# Structure

- `algebra`: Provides algebraic functionalities including field operations and extension fields
- `algebra_derive`: Derive macros for algebraic operations
- `helper`: Utility functions including transcript implementation
- `pcs`: Polynomial Commitment Scheme implementations
- `piop`: Polynomial-based Interactive Oracle Proof systems
- `poly`: Polynomial data structures and operations
- `sumcheck`: Implementation of the sum-check protocol

# Setup and Testing

Note the implementation requires rust version >= 1.80.0

To run the code testing, simply follow the steps: 
```
git clone ...
cd HasteBoots
cargo build --release
```
Once the library is compiled successfully, you can run `cargo test` for the functionality testing, and you can run `cargo bench` for the performance testing. Note you can check the ***tables*** in the original paper for the scheme performance details.

# Parameter Configuration
## FHE Parameters

- **𝑛**: 1024
- **𝑞**: 1024  
- **𝑁**: 1024
- **𝑄**: 2<sup>31</sup> − 2<sup>27</sup> + 1

## Extension Field Parameters
- **Extension Field Size**: 𝑄<sup>𝐷</sup>
- **𝐷**: 4 (ensures soundness)
- **Base Field Size**: 𝑄 (underlying FHE instance field)

## Gadget Decomposition Parameters
- **Gadget Size (𝐵)**: 2<sup>7</sup>

## LogUp Lookup Protocol Parameters
- **Batch Block Size**: 3 (minimizes overhead)

## Brakedown PCS Parameters
- **𝛼**: 0.1195 (controls the size of the subcode in each recursive call)
- **𝛽**: 0.0248 (defines the code distance)
- **𝑟**: 1.9 (adjusts the code rate)
- **Recursion Stopping Threshold**: 10 (optimizes performance)

