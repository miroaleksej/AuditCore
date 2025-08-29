# AuditCore: Topological ECDSA Security Auditor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![CI/CD Status](https://github.com/miroaleksej/AuditCore/workflows/ECDSA%20Security%20Check/badge.svg)](https://github.com/miroaleksej/AuditCore/actions)

**AuditCore** is the world's first industrial implementation of topological analysis for ECDSA security auditing. Unlike traditional approaches that require hundreds of real signatures, AuditCore can identify cryptographic vulnerabilities using just a public key and a single real signature from the blockchain.

## Key Features

- **Minimal input requirements**: Only needs a public key and one real signature
- **Automatic signature generation**: Creates valid synthetic signatures for statistically significant analysis
- **Topological vulnerability detection**: Identifies structural weaknesses through persistent homology
- **TVI Score**: Quantitative metric for vulnerability assessment (Torus Vulnerability Index)
- **Pattern recognition**: Detects specific vulnerability types (fixed k, linear dependencies, etc.)
- **Multi-platform**: CPU, GPU, and distributed computing support via DynamicComputeRouter
- **Ready for integration**: CI/CD pipelines, wallet extensions, and security monitoring

## How It Works

AuditCore represents the ECDSA signature space as a topological torus where each signature corresponds to a point with coordinates (u_r, u_z), with:

```
u_r = r · s⁻¹ mod n
u_z = z · s⁻¹ mod n
```

In a secure implementation, these points should form a uniform distribution on the torus with specific topological properties (Betti numbers β₀=1, β₁=2, β₂=1). Deviations from these properties indicate vulnerabilities.

The system's innovation lies in its ability to generate additional valid signatures from just one real signature, enabling thorough topological analysis even for new or low-activity wallets.

## Architecture Overview

AuditCore consists of specialized modules working together:

### Core Analysis Modules

- **TopologicalAnalyzer**: Computes persistent homology and determines topological structure
- **BettiAnalyzer**: Interprets Betti numbers into security metrics
- **TCON (Torus CONformance)**: Measures conformance to expected torus topology
- **GradientAnalysis**: Detects linear dependencies through gradient field analysis
- **CollisionEngine**: Identifies repeated r values and analyzes their structure

### Supporting Modules

- **SignatureGenerator**: Creates valid signatures without private key knowledge
- **HyperCoreTransformer**: Transforms data into proper torus representation
- **DynamicComputeRouter**: Optimizes resource allocation (CPU/GPU/distributed)
- **AIAssistant**: Interprets results and provides actionable recommendations

## Installation

```bash
# Clone the repository
git clone https://github.com/miroaleksej/AuditCore.git
cd AuditCore

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install .
```

## Quick Start

### Basic Usage

```python
from auditcore import AuditCore

# Initialize the auditor
auditor = AuditCore()

# Load public key (x, y coordinates)
public_key = (
    0x8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c,
    0x5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f
)

# Load one real signature (r, s, z)
real_signature = (
    0x57d78d2975a1d1d9e9f5f0b4c8a7b6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9,
    0x68e89f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e,
    0x3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b
)

# Run the audit (automatically generates additional signatures)
result = auditor.audit(public_key, [real_signature])

# Display results
print(f"TVI Score: {result['tvi_score']:.3f}")
print(f"Security Status: {'SECURE' if result['topological_security'] else 'VULNERABLE'}")
print(f"Vulnerabilities: {', '.join(result['critical_vulnerabilities'])}")
```

### Command Line Interface

```bash
python -m auditcore \
  --public-key 8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c,5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f \
  --signature 57d78d2975a1d1d9e9f5f0b4c8a7b6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9,68e89f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e,3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b
```

## Understanding the Results

AuditCore outputs a comprehensive report including:

- **TVI Score** (Torus Vulnerability Index): Quantitative vulnerability metric
  - < 0.2: Secure
  - 0.2-0.5: Potential vulnerability
  - > 0.5: Critical vulnerability

- **Identified vulnerability patterns**:
  - Horizontal lines → Fixed k (like Sony PS3 vulnerability)
  - Diagonal stripes → Linear dependency k = a·t + b
  - Corner clusters → Biased lower bits of k
  - Closed contours → Periodic k generator

- **Actionable recommendations** for each identified vulnerability

## Integration Examples

### CI/CD Pipeline Integration

```yaml
# .github/workflows/ecdsa-check.yml
name: ECDSA Security Check
on: [push]
jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - name: Run ECDSA Audit
        run: |
          python -m auditcore --public-key ${{ secrets.PUBLIC_KEY }} \
                               --signature ${{ secrets.SIGNATURE }}
          if [ $(cat report.json | jq '.tvi_score') > 0.3 ]; then
            echo "Potential vulnerability detected! TVI Score = $(cat report.json | jq '.tvi_score')"
            exit 1
          fi
```

### Wallet Security Extension

AuditCore can be integrated into wallet extensions like MetaMask to:
- Continuously monitor wallet security
- Alert users about potential vulnerabilities
- Recommend key rotation when vulnerabilities are detected

## Technical Details

- **Mathematical Foundation**: Based on representing ECDSA signature space as a torus
- **Topological Analysis**: Uses persistent homology (Vietoris-Rips complex) via giotto-tda
- **Resource Management**: DynamicComputeRouter optimizes between CPU, GPU, and distributed computing
- **Curve Support**: Initially focused on secp256k1 (Bitcoin/Ethereum), extensible to other curves

## Contributing

We welcome contributions! Please read our [CONTRIBUTING.md](https://github.com/miroaleksej/AuditCore/blob/main/Doc/Contributing.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Documentation

Comprehensive documentation is currently in development and will be available in the coming months. For immediate assistance, please contact the development team.

---

*AuditCore represents a paradigm shift in ECDSA security analysis, moving from statistical testing to topological understanding of signature space.*
