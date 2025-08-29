# AuditCore v3.2: Comprehensive User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Installation Guide](#installation-guide)
4. [Configuration](#configuration)
5. [Basic Usage](#basic-usage)
6. [Advanced Analysis Techniques](#advanced-analysis-techniques)
7. [Interpreting Results](#interpreting-results)
8. [Integration with Blockchain Systems](#integration-with-blockchain-systems)
9. [CI/CD Pipeline Integration](#cid-pipeline-integration)
10. [Performance Optimization](#performance-optimization)
11. [Troubleshooting](#troubleshooting)
12. [Security Considerations](#security-considerations)
13. [API Reference](#api-reference)
14. [Case Studies](#case-studies)
15. [Frequently Asked Questions](#frequently-asked-questions)
16. [Support and Community](#support-and-community)

## Introduction

AuditCore v3.2 is the world's first industrial implementation of topological analysis for ECDSA security auditing. Unlike traditional approaches that require hundreds of real signatures, AuditCore can identify cryptographic vulnerabilities using just a public key and a single real signature from the blockchain.

This manual provides comprehensive guidance on installing, configuring, and using AuditCore for ECDSA security analysis. Whether you're a blockchain developer, security researcher, or cryptography enthusiast, this guide will help you leverage topological analysis to detect vulnerabilities that traditional methods miss.

## System Overview

### Core Architecture

AuditCore is a modular system with specialized components working together to provide topological analysis of ECDSA implementations:

```
AuditCore v3.2 Architecture
│
├── TopologicalAnalyzer (Core Analysis)
│   ├── BettiAnalyzer (Topological Invariants)
│   ├── TCON (Torus Conformance)
│   └── GradientAnalysis (Pattern Detection)
│
├── Data Processing
│   ├── SignatureGenerator (Synthetic Data)
│   ├── HyperCoreTransformer (Data Transformation)
│   └── CollisionEngine (Collision Detection)
│
├── Resource Management
│   └── DynamicComputeRouter (CPU/GPU/Distributed)
│
└── Interface Layer
    ├── AIAssistant (Result Interpretation)
    └── AuditCore (Main API)
```

### Mathematical Foundation

AuditCore represents the ECDSA signature space as a topological torus where each signature corresponds to a point with coordinates (u_r, u_z):

```
u_r = r · s⁻¹ mod n
u_z = z · s⁻¹ mod n
```

In a secure implementation, these points should form a uniform distribution on the torus with specific topological properties:
- β₀ = 1 (one connected component)
- β₁ = 2 (two independent cycles)
- β₂ = 1 (one two-dimensional void)

Deviations from these properties indicate vulnerabilities. AuditCore's innovation lies in its ability to generate additional valid signatures from just one real signature, enabling thorough topological analysis even for new or low-activity wallets.

### Key Features

- **Minimal input requirements**: Only needs a public key and one real signature
- **Automatic signature generation**: Creates valid synthetic signatures for statistically significant analysis
- **Topological vulnerability detection**: Identifies structural weaknesses through persistent homology
- **TVI Score**: Quantitative metric for vulnerability assessment (Torus Vulnerability Index)
- **Pattern recognition**: Detects specific vulnerability types (fixed k, linear dependencies, etc.)
- **Multi-platform**: CPU, GPU, and distributed computing support
- **Ready for integration**: CI/CD pipelines, wallet extensions, and security monitoring

## Installation Guide

### System Requirements

- **Operating System**: Linux, macOS, or Windows (WSL recommended for Windows)
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large-scale analysis)
- **Storage**: 500MB for core system, additional space for cache/data
- **Optional**: NVIDIA GPU with CUDA 11.0+ for accelerated computations

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/miroaleksej/AuditCore.git
cd AuditCore
```

#### 2. Set Up Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate  # Windows
```

#### 3. Install Dependencies

AuditCore offers multiple installation profiles based on your needs:

##### Basic Installation (Core Functionality)
```bash
pip install -r requirements.txt
```

##### Full Installation (With Visualization and Advanced Features)
```bash
pip install -r requirements-full.txt
```

##### GPU Acceleration (Requires CUDA)
```bash
pip install -r requirements-gpu.txt
```

##### Development Installation (For Contributors)
```bash
pip install -r requirements-dev.txt
```

#### 4. Verify Installation

```bash
python -c "import auditcore; print(auditcore.__version__)"
# Should output: 3.2.0
```

#### 5. Install Optional Components

For blockchain integration:

```bash
# Bitcoin integration
pip install blockstream

# Ethereum integration
pip install web3
```

#### 6. Set Up Cache Directory (Optional but Recommended)

```bash
mkdir -p ~/.auditcore/cache
export AUDITCORE_CACHE_DIR=~/.auditcore/cache
```

Add this to your `.bashrc` or `.zshrc` to make it persistent.

### Docker Installation (Alternative Method)

For containerized deployment:

```bash
# Build the Docker image
docker build -t auditcore:latest .

# Run AuditCore in a container
docker run -it --rm -v $(pwd)/results:/app/results auditcore:latest
```

GPU support with Docker:

```bash
docker run -it --rm --gpus all -v $(pwd)/results:/app/results auditcore:latest
```

## Configuration

AuditCore uses a hierarchical configuration system with multiple override levels:

1. Default configuration (bundled with the package)
2. System-wide configuration (`/etc/auditcore/config.yaml`)
3. User configuration (`~/.auditcore/config.yaml`)
4. Environment variables
5. Command-line arguments (highest priority)

### Configuration File Structure

Create a configuration file at `~/.auditcore/config.yaml`:

```yaml
# Core analysis parameters
analysis:
  min_signatures: 500       # Minimum signatures for analysis
  max_signatures: 5000      # Maximum signatures for analysis
  tvi_threshold: 0.2        # Threshold for secure TVI Score
  confidence_threshold: 0.7 # Minimum confidence for vulnerability detection
  resolution: 100           # Resolution for torus visualization

# Resource management
resources:
  preferred_device: "auto"  # "auto", "cpu", "gpu", or "distributed"
  max_memory: "80%"         # Maximum memory usage (percentage or bytes)
  cache_dir: "~/.auditcore/cache"
  workers: "auto"           # Number of parallel workers

# Blockchain integration
blockchain:
  bitcoin:
    api_url: "https://blockstream.info/api"
    timeout: 30
  ethereum:
    api_url: "https://api.etherscan.io/api"
    api_key: ""             # Set your Etherscan API key here
    timeout: 30

# Advanced parameters
advanced:
  persistence_diagram:
    homology_dimensions: [0, 1, 2]
    filtration_max: 0.5
  gradient:
    window_size: 15
    threshold: 0.8
  collision:
    min_repetitions: 2
    max_distance: 0.01
```

### Environment Variables

You can override specific settings using environment variables:

```bash
# Set preferred device to GPU
export AUDITCORE_RESOURCES_PREFERRED_DEVICE="gpu"

# Set maximum memory usage
export AUDITCORE_RESOURCES_MAX_MEMORY="4G"

# Set Etherscan API key
export AUDITCORE_BLOCKCHAIN_ETHEREUM_API_KEY="your_api_key_here"
```

### Configuration Validation

Validate your configuration with:

```bash
auditcore validate-config
```

This will check for missing parameters, invalid values, and potential conflicts.

## Basic Usage

### Command-Line Interface

AuditCore provides a powerful command-line interface for quick analysis:

#### Analyzing a Public Key with One Signature

```bash
auditcore analyze \
  --public-key 8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c,5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f \
  --signature 57d78d2975a1d1d9e9f5f0b4c8a7b6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9,68e89f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e,3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b
```

#### Analyzing a Bitcoin Address

```bash
auditcore analyze-bitcoin \
  --address 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa \
  --output report.json
```

#### Analyzing an Ethereum Address

```bash
auditcore analyze-ethereum \
  --address 0x742d35Cc6634C0532925a3b844Bc454e4438f44e \
  --output report.html
```

#### Generating a Torus Visualization

```bash
auditcore visualize \
  --public-key <your_public_key> \
  --signature <your_signature> \
  --output 3d_torus.html
```

### Python API

AuditCore provides a comprehensive Python API for programmatic access:

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

# Generate visualization
auditor.visualize(result, output_path="3d_torus.html")
```

### Quick Start Workflow

Here's a complete workflow to analyze your own wallet:

```bash
# Step 1: Get your public key and one signature
# For Bitcoin, you can use a block explorer to find these

# Step 2: Run the analysis
auditcore analyze \
  --public-key <your_public_key> \
  --signature <your_signature> \
  --output analysis_report.json

# Step 3: View the report
cat analysis_report.json | jq .

# Step 4: Generate a visualization
auditcore visualize \
  --public-key <your_public_key> \
  --signature <your_signature> \
  --output 3d_torus.html

# Step 5: Open the visualization in your browser
open 3d_torus.html  # macOS
xdg-open 3d_torus.html  # Linux
start 3d_torus.html  # Windows
```

## Advanced Analysis Techniques

### Custom Signature Generation

You can customize how synthetic signatures are generated:

```python
from auditcore import AuditCore

auditor = AuditCore()
public_key = (...)  # Your public key
real_signature = (...)  # Your real signature

# Custom signature generation parameters
result = auditor.audit(
    public_key,
    [real_signature],
    min_signatures=1000,
    max_signatures=3000,
    sigma_ur=0.15,  # Spread in u_r direction
    sigma_uz=0.15,  # Spread in u_z direction
    generation_strategy="adaptive"  # "uniform", "adaptive", or "concentrated"
)
```

### Targeted Vulnerability Analysis

Focus on specific vulnerability types:

```python
# Analyze specifically for fixed k vulnerability
result = auditor.audit(
    public_key,
    signatures,
    target_vulnerability="fixed_k",
    confidence_threshold=0.85
)

# Analyze for linear dependencies
result = auditor.audit(
    public_key,
    signatures,
    target_vulnerability="linear_dependency",
    gradient_window=20
)
```

### Multi-Stage Analysis

For complex scenarios, use a multi-stage analysis approach:

```python
# Stage 1: Quick scan with minimal signatures
result1 = auditor.audit(public_key, signatures, min_signatures=500, max_signatures=500)

# If potential vulnerability detected, perform deeper analysis
if result1['tvi_score'] > 0.15:
    # Stage 2: Focused analysis on suspicious regions
    result2 = auditor.audit(
        public_key,
        signatures,
        min_signatures=2000,
        max_signatures=5000,
        focus_regions=result1['suspicious_regions']
    )
    
    # Stage 3: Attempt key recovery if critical vulnerability confirmed
    if result2['tvi_score'] > 0.5 and result2['critical_vulnerabilities']:
        recovery_result = auditor.recover_private_key(
            public_key,
            signatures,
            vulnerability_type=result2['critical_vulnerabilities'][0]
        )
        print(f"Recovered private key? {recovery_result['success']}")
```

### Custom Topological Metrics

Create your own topological metrics:

```python
from auditcore.topology import calculate_torus_curvature

# Get the torus coordinates from the analysis
torus_coords = result['torus_coordinates']

# Calculate custom metric: torus curvature
curvature = calculate_torus_curvature(torus_coords)

print(f"Torus curvature: {curvature:.4f}")
# Low curvature (~0.01) indicates uniform distribution (secure)
# High curvature (>0.1) indicates concentration of points (vulnerable)
```

## Interpreting Results

### Understanding the TVI Score

The Torus Vulnerability Index (TVI) is AuditCore's primary security metric:

| TVI Score Range | Security Status | Action Required |
|----------------|----------------|----------------|
| 0.0 - 0.2      | Secure         | No action needed |
| 0.2 - 0.5      | Potential Vulnerability | Investigate further, consider key rotation |
| 0.5 - 1.0      | Critical Vulnerability | Immediate key rotation required |

The TVI Score is calculated as:
```
TVI = w₁·|β₀-1| + w₂·|β₁-2| + w₃·|β₂-1| + w₄·stability_score
```

Where weights (w₁-w₄) are determined empirically based on vulnerability impact.

### Common Vulnerability Patterns

AuditCore identifies several specific vulnerability patterns:

#### 1. Fixed k Vulnerability (Sony PS3-style)
- **Visual Pattern**: Horizontal lines in the torus visualization
- **Topological Indicators**: 
  - β₀ > 1 (multiple connected components)
  - High gradient consistency in u_z direction
  - Clustered r values
- **TVI Contribution**: High (w₁ and w₄ weights)
- **Risk Level**: Critical
- **Recommendation**: Immediate key rotation

#### 2. Linear Dependency k = a·t + b
- **Visual Pattern**: Diagonal stripes across the torus
- **Topological Indicators**:
  - Abnormal persistence in H₁ (1-dimensional homology)
  - Consistent gradient direction
- **TVI Contribution**: Medium-High (w₂ and gradient weights)
- **Risk Level**: High
- **Recommendation**: Upgrade RNG implementation

#### 3. Biased Lower Bits of k
- **Visual Pattern**: Clusters in corners of the torus
- **Topological Indicators**:
  - Localized high density regions
  - Anisotropic Betti number distribution
- **TVI Contribution**: Medium (w₁ and density weights)
- **Risk Level**: Medium
- **Recommendation**: Review k generation process

#### 4. Periodic k Generator
- **Visual Pattern**: Closed contours or repeating patterns
- **Topological Indicators**:
  - Unusual cyclic structures in persistence diagram
  - Regular spacing in collision detection
- **TVI Contribution**: Medium-High
- **Risk Level**: High
- **Recommendation**: Replace with cryptographic RNG

### Sample Report Analysis

Here's how to interpret a typical AuditCore report:

```json
{
  "tvi_score": 0.67,
  "topological_security": false,
  "security_status": "CRITICAL",
  "betti_numbers": {
    "beta_0": 42,
    "beta_1": 2.1,
    "beta_2": 0.8
  },
  "stability_score": 0.92,
  "critical_vulnerabilities": ["fixed_k"],
  "vulnerability_details": {
    "fixed_k": {
      "confidence": 0.87,
      "evidence": "42 distinct clusters detected",
      "pattern_strength": 0.93,
      "affected_signatures": 1248
    }
  },
  "suspicious_regions": [
    {
      "center": [0.02, 0.01],
      "radius": 0.05,
      "density": 0.87,
      "vulnerability_type": "fixed_k"
    }
  ],
  "recommendations": [
    "Immediate key rotation required",
    "Check RNG implementation for deterministic behavior",
    "Consider implementing TopoNonce for secure k generation"
  ],
  "metadata": {
    "public_key": "0x8b7c...8b7c,0x5e4f...5e4f",
    "signature_count": 1248,
    "analysis_time": "2023-10-15T14:30:22Z",
    "auditcore_version": "3.2.0"
  }
}
```

**Interpretation**:
- TVI Score of 0.67 indicates a critical vulnerability
- β₀ = 42 (instead of expected 1) confirms multiple disconnected components
- Fixed k vulnerability detected with 87% confidence
- 42 distinct clusters suggest a severely compromised RNG
- Immediate action is required to prevent private key compromise

### Visualization Guide

AuditCore generates interactive 3D visualizations of the torus. Here's how to interpret them:

1. **Color Coding**:
   - Blue: Low density (few signatures)
   - Green: Medium density
   - Yellow/Red: High density (potential vulnerability)

2. **Pattern Recognition**:
   - **Horizontal lines**: Fixed k vulnerability
   - **Diagonal stripes**: Linear dependency in k
   - **Corner clusters**: Biased lower bits of k
   - **Closed contours**: Periodic k generator

3. **Interactive Exploration**:
   - Rotate the torus to view from different angles
   - Zoom in on high-density regions
   - Toggle visibility of different signature sets
   - Adjust the density threshold to highlight problem areas

## Integration with Blockchain Systems

### Bitcoin Integration

AuditCore can directly analyze Bitcoin addresses using the Blockstream API:

```bash
# Analyze a Bitcoin address
auditcore analyze-bitcoin \
  --address 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa \
  --output bitcoin_report.json

# With verbose output for debugging
auditcore analyze-bitcoin \
  --address 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa \
  --verbose
```

#### Python API for Bitcoin

```python
from auditcore.blockchain import BitcoinAnalyzer

analyzer = BitcoinAnalyzer()
result = analyzer.analyze("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")

print(f"Address: {result['address']}")
print(f"TVI Score: {result['tvi_score']:.3f}")
print(f"Total signatures: {result['signature_count']}")
```

### Ethereum Integration

AuditCore supports Ethereum address analysis through Etherscan:

```bash
# Analyze an Ethereum address
auditcore analyze-ethereum \
  --address 0x742d35Cc6634C0532925a3b844Bc454e4438f44e \
  --output ethereum_report.html

# With custom Etherscan API key
auditcore analyze-ethereum \
  --address 0x742d35Cc6634C0532925a3b844Bc454e4438f44e \
  --api-key YOUR_ETHERSCAN_KEY
```

#### Python API for Ethereum

```python
from auditcore.blockchain import EthereumAnalyzer

analyzer = EthereumAnalyzer(api_key="YOUR_ETHERSCAN_KEY")
result = analyzer.analyze("0x742d35Cc6634C0532925a3b844Bc454e4438f44e")

print(f"Address: {result['address']}")
print(f"TVI Score: {result['tvi_score']:.3f}")
print(f"Risk Level: {result['risk_level']}")
```

### Custom Blockchain Integration

You can integrate AuditCore with any blockchain that uses ECDSA:

```python
from auditcore import AuditCore
from my_blockchain import get_public_key, get_signatures

# Get data from your blockchain
public_key = get_public_key("your_address")
signatures = get_signatures("your_address", limit=1)

# Run AuditCore analysis
auditor = AuditCore()
result = auditor.audit(public_key, signatures)

# Process results
if result['tvi_score'] > 0.5:
    print("CRITICAL: Address is highly vulnerable!")
    # Implement your security response
```

### Wallet Security Extension

AuditCore can be integrated into wallet extensions like MetaMask:

```javascript
// MetaMask extension example
import { AuditCore } from 'auditcore';

async function checkWalletSecurity() {
  const accounts = await ethereum.request({ method: 'eth_accounts' });
  const address = accounts[0];
  
  // Get public key and signatures from wallet
  const { publicKey, signatures } = await getWalletData(address);
  
  // Run AuditCore analysis
  const auditor = new AuditCore();
  const result = auditor.audit(publicKey, signatures);
  
  // Display security status
  if (result.tvi_score > 0.5) {
    showCriticalAlert(`Your wallet is vulnerable! TVI Score: ${result.tvi_score.toFixed(2)}`);
  } else if (result.tvi_score > 0.2) {
    showWarningAlert(`Potential vulnerability detected. TVI Score: ${result.tvi_score.toFixed(2)}`);
  }
}
```

## CI/CD Pipeline Integration

### GitHub Actions Integration

Add ECDSA security checks to your CI/CD pipeline:

```yaml
# .github/workflows/ecdsa-security.yml
name: ECDSA Security Audit
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  security-audit:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install auditcore

      - name: Run ECDSA Audit
        env:
          PUBLIC_KEY: ${{ secrets.PUBLIC_KEY }}
          SIGNATURE: ${{ secrets.SIGNATURE }}
        run: |
          # Run audit with your public key and signature
          auditcore analyze \
            --public-key $PUBLIC_KEY \
            --signature $SIGNATURE \
            --output audit_report.json
          
          # Check TVI Score threshold
          TVI_SCORE=$(jq '.tvi_score' audit_report.json)
          if (( $(echo "$TVI_SCORE > 0.3" | bc -l) )); then
            echo "❌ Vulnerability detected! TVI Score = $TVI_SCORE"
            cat audit_report.json
            exit 1
          else
            echo "✅ Security check passed. TVI Score = $TVI_SCORE"
          fi
```

### GitLab CI Integration

```yaml
ecdsa-security-check:
  image: python:3.9
  script:
    - pip install auditcore
    - |
      TVI_SCORE=$(auditcore analyze \
        --public-key $PUBLIC_KEY \
        --signature $SIGNATURE \
        --output report.json | jq '.tvi_score')
      
      if (( $(echo "$TVI_SCORE > 0.3" | bc -l) )); then
        echo "Vulnerability detected! TVI Score = $TVI_SCORE"
        exit 1
      fi
  variables:
    PUBLIC_KEY: "your_public_key_here"
    SIGNATURE: "your_signature_here"
```

### Pre-Commit Hook

Add AuditCore to your pre-commit checks:

```bash
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: ecdsa-security-check
        name: ECDSA Security Check
        language: system
        entry: |
          auditcore analyze \
            --public-key $(cat public_key.txt) \
            --signature $(cat signature.txt) \
            --output audit_report.json && \
          TVI_SCORE=$(cat audit_report.json | jq '.tvi_score') && \
          if [ $(echo "$TVI_SCORE > 0.3" | bc) -eq 1 ]; then \
            echo "ECDSA vulnerability detected! TVI Score = $TVI_SCORE"; \
            exit 1; \
          fi
        types: [file]
        files: ^public_key\.txt$
        always_run: true
```

### Enterprise Security Monitoring

Integrate AuditCore with security monitoring systems:

```python
from auditcore import AuditCore
import requests

def monitor_wallets(wallet_list, security_team_webhook):
    """Monitor multiple wallets and alert on vulnerabilities"""
    auditor = AuditCore()
    
    for wallet in wallet_list:
        try:
            # Analyze the wallet
            result = auditor.analyze_blockchain_wallet(wallet['chain'], wallet['address'])
            
            # Check if vulnerability exceeds threshold
            if result['tvi_score'] > wallet.get('threshold', 0.3):
                # Send alert to security team
                requests.post(security_team_webhook, json={
                    'alert_type': 'ecdsa_vulnerability',
                    'wallet': wallet['address'],
                    'chain': wallet['chain'],
                    'tvi_score': result['tvi_score'],
                    'vulnerabilities': result['critical_vulnerabilities'],
                    'timestamp': datetime.utcnow().isoformat()
                })
                
        except Exception as e:
            logging.error(f"Error monitoring {wallet}: {str(e)}")

# Example usage
monitor_wallets(
    wallet_list=[
        {'chain': 'bitcoin', 'address': '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa', 'threshold': 0.2},
        {'chain': 'ethereum', 'address': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e', 'threshold': 0.25}
    ],
    security_team_webhook='https://your-security-system/alerts'
)
```

## Performance Optimization

### Resource Management

AuditCore automatically optimizes resource usage, but you can fine-tune it:

```bash
# Use GPU for accelerated computation
auditcore analyze \
  --public-key <key> \
  --signature <sig> \
  --device gpu

# Limit memory usage to 4GB
auditcore analyze \
  --public-key <key> \
  --signature <sig> \
  --max-memory 4G

# Use distributed computing with 8 workers
auditcore analyze \
  --public-key <key> \
  --signature <sig> \
  --workers 8
```

### Performance Tuning Parameters

Add these to your configuration file for optimal performance:

```yaml
# ~/.auditcore/config.yaml
resources:
  preferred_device: "gpu"      # "cpu", "gpu", or "distributed"
  max_memory: "6G"             # Maximum memory usage
  workers: 4                   # Number of parallel workers
  cache_size: "1G"             # Cache size for intermediate results
  batch_size: 500              # Batch size for signature processing

advanced:
  persistence_diagram:
    filtration_max: 0.4        # Lower value = faster but less accurate
  gradient:
    window_size: 10            # Smaller window = faster but less precise
  collision:
    max_distance: 0.02         # Larger distance = faster but less sensitive
```

### Benchmarking Your Setup

Test your system's performance:

```bash
auditcore benchmark \
  --signature-count 1000 \
  --device auto \
  --output benchmark_results.json
```

Sample output:
```json
{
  "device": "GPU (NVIDIA RTX 3080)",
  "signature_count": 1000,
  "analysis_time": 2.45,
  "memory_usage": "1.8G",
  "throughput": 408.16,
  "recommendations": [
    "Consider increasing batch_size for better GPU utilization",
    "Memory usage is within optimal range"
  ]
}
```

### Large-Scale Analysis

For analyzing thousands of wallets:

```bash
# Create a list of wallets to analyze
cat > wallets.txt << EOF
bitcoin:1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
ethereum:0x742d35Cc6634C0532925a3b844Bc454e4438f44e
...
EOF

# Run distributed analysis
auditcore batch-analyze \
  --input wallets.txt \
  --output results/ \
  --workers 16 \
  --threshold 0.25
```

This will create individual reports for each wallet and a summary file with all results.

## Troubleshooting

### Common Issues and Solutions

#### Issue: "fastecdsa library not found"
**Symptom**: Warning during import: "fastecdsa library not found"
**Solution**: Install the required library:
```bash
pip install fastecdsa
```

#### Issue: Slow Analysis Performance
**Symptom**: Analysis takes much longer than expected
**Solutions**:
1. Enable GPU acceleration if available:
   ```bash
   auditcore analyze --device gpu
   ```
2. Reduce the number of signatures for initial analysis:
   ```bash
   auditcore analyze --max-signatures 1000
   ```
3. Adjust advanced parameters in config:
   ```yaml
   advanced:
     persistence_diagram:
       filtration_max: 0.3
     gradient:
       window_size: 8
   ```

#### Issue: "Not enough signatures for reliable analysis"
**Symptom**: Error message about insufficient signatures
**Solutions**:
1. Lower the minimum signature count in config:
   ```yaml
   analysis:
     min_signatures: 200
   ```
2. Use a different generation strategy:
   ```bash
   auditcore analyze --generation-strategy adaptive
   ```
3. For Bitcoin/Ethereum, increase the number of transactions analyzed:
   ```bash
   auditcore analyze-bitcoin --transaction-limit 50
   ```

#### Issue: High TVI Score but No Clear Vulnerability
**Symptom**: TVI Score is high but no specific vulnerability is identified
**Solutions**:
1. Increase confidence threshold for vulnerability detection:
   ```yaml
   analysis:
     confidence_threshold: 0.6
   ```
2. Run a more detailed analysis:
   ```bash
   auditcore analyze --detailed
   ```
3. Check the persistence diagram for subtle patterns:
   ```bash
   auditcore analyze --output-format json | jq '.persistence_diagram'
   ```

### Diagnostic Tools

AuditCore includes several diagnostic tools:

#### System Check
```bash
auditcore system-check
```
Checks for required dependencies, resource availability, and configuration issues.

#### Configuration Validation
```bash
auditcore validate-config
```
Validates your configuration file for errors and inconsistencies.

#### Dependency Check
```bash
auditcore check-dependencies
```
Lists all dependencies and their status (installed/missing).

### Debugging Tips

For detailed debugging:

```bash
# Enable verbose logging
auditcore analyze \
  --public-key <key> \
  --signature <sig> \
  --verbose \
  --log-level debug

# Save intermediate results for analysis
auditcore analyze \
  --public-key <key> \
  --signature <sig> \
  --save-intermediate results/
```

This will provide detailed information about each step of the analysis process.

## Security Considerations

### Safe Usage Practices

1. **Never analyze private keys directly**
   - AuditCore only requires the public key and signatures
   - Never input private keys into any system

2. **Secure your analysis environment**
   - Run AuditCore in isolated environments
   - Don't share analysis results containing sensitive information

3. **Verify results before taking action**
   - High TVI Score doesn't always mean immediate compromise
   - Review the detailed report before rotating keys

### Limitations to Be Aware Of

1. **False Positives/Negatives**
   - TVI Score is a probabilistic metric
   - Always combine with other security practices

2. **Not a Complete Security Solution**
   - AuditCore focuses on ECDSA implementation vulnerabilities
   - It doesn't protect against side-channel attacks or physical security issues

3. **Signature Generation Limitations**
   - Synthetic signatures are for analysis only
   - Never use generated signatures for real transactions

### Responsible Disclosure

If you discover a critical vulnerability in a public system:

1. **Do not publicly disclose** the vulnerability
2. **Contact the affected party** through official channels
3. **Allow reasonable time** for remediation (typically 90 days)
4. **Coordinate disclosure** with security teams

AuditCore includes a responsible disclosure helper:

```bash
auditcore report-vulnerability \
  --address 0x742d35Cc6634C0532925a3b844Bc454e4438f44e \
  --contact security@example.com \
  --details "Critical fixed_k vulnerability detected"
```

## API Reference

### AuditCore Class

```python
class AuditCore:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        verbose: bool = False
    ):
        """Initialize the AuditCore analyzer.
        
        Args:
            config: Configuration dictionary (overrides defaults)
            device: Computation device ("auto", "cpu", "gpu", "distributed")
            verbose: Enable verbose output
        """
```

#### audit() Method

```python
def audit(
    self,
    public_key: Tuple[int, int],
    signatures: List[Tuple[int, int, int]],
    min_signatures: Optional[int] = None,
    max_signatures: Optional[int] = None,
    sigma_ur: float = 0.15,
    sigma_uz: float = 0.15,
    generation_strategy: str = "adaptive",
    target_vulnerability: Optional[str] = None,
    confidence_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Perform topological analysis of ECDSA signatures.
    
    Args:
        public_key: Tuple (x, y) of public key point
        signatures: List of (r, s, z) signature tuples
        min_signatures: Minimum number of signatures for analysis
        max_signatures: Maximum number of signatures for analysis
        sigma_ur: Spread parameter in u_r direction
        sigma_uz: Spread parameter in u_z direction
        generation_strategy: "uniform", "adaptive", or "concentrated"
        target_vulnerability: Specific vulnerability to focus on
        confidence_threshold: Minimum confidence for vulnerability detection
    
    Returns:
        Dictionary containing analysis results with keys:
        - tvi_score: Torus Vulnerability Index (0.0-1.0)
        - topological_security: Boolean indicating security status
        - betti_numbers: Dict with beta_0, beta_1, beta_2
        - stability_score: Measure of topological stability
        - critical_vulnerabilities: List of detected vulnerabilities
        - vulnerability_details: Detailed information on vulnerabilities
        - suspicious_regions: Coordinates of suspicious areas
        - recommendations: Security recommendations
        - metadata: Analysis metadata
    """
```

#### visualize() Method

```python
def visualize(
    self,
    analysis_result: Dict[str, Any],
    output_path: str,
    width: int = 1200,
    height: int = 800,
    theme: str = "dark"
) -> None:
    """Generate interactive 3D visualization of analysis results.
    
    Args:
        analysis_result: Result dictionary from audit() method
        output_path: Path to save HTML visualization
        width: Width of visualization in pixels
        height: Height of visualization in pixels
        theme: "light" or "dark" theme
    """
```

#### recover_private_key() Method

```python
def recover_private_key(
    self,
    public_key: Tuple[int, int],
    signatures: List[Tuple[int, int, int]],
    vulnerability_type: str,
    max_iterations: int = 1000
) -> Dict[str, Any]:
    """Attempt private key recovery based on detected vulnerability.
    
    Note: This is for research purposes only. Only use on your own keys.
    
    Args:
        public_key: Tuple (x, y) of public key point
        signatures: List of (r, s, z) signature tuples
        vulnerability_type: Type of vulnerability to exploit
        max_iterations: Maximum iterations for recovery algorithm
    
    Returns:
        Dictionary with keys:
        - success: Boolean indicating if recovery was successful
        - private_key: Recovered private key (if successful)
        - iterations: Number of iterations performed
        - confidence: Confidence in the recovered key
    """
```

### Blockchain Analyzers

#### BitcoinAnalyzer

```python
class BitcoinAnalyzer:
    def __init__(
        self,
        api_url: str = "https://blockstream.info/api",
        timeout: int = 30
    ):
        """Initialize Bitcoin blockchain analyzer.
        
        Args:
            api_url: Blockstream API URL
            timeout: Request timeout in seconds
        """
    
    def analyze(
        self,
        address: str,
        transaction_limit: int = 100,
        min_signatures: int = 500
    ) -> Dict[str, Any]:
        """Analyze Bitcoin address for ECDSA vulnerabilities.
        
        Args:
            address: Bitcoin address to analyze
            transaction_limit: Maximum transactions to fetch
            min_signatures: Minimum signatures needed for analysis
            
        Returns:
            Analysis results dictionary
        """
```

#### EthereumAnalyzer

```python
class EthereumAnalyzer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.etherscan.io/api",
        timeout: int = 30
    ):
        """Initialize Ethereum blockchain analyzer.
        
        Args:
            api_key: Etherscan API key (required for higher limits)
            api_url: Etherscan API URL
            timeout: Request timeout in seconds
        """
    
    def analyze(
        self,
        address: str,
        transaction_limit: int = 100,
        min_signatures: int = 500
    ) -> Dict[str, Any]:
        """Analyze Ethereum address for ECDSA vulnerabilities.
        
        Args:
            address: Ethereum address to analyze
            transaction_limit: Maximum transactions to fetch
            min_signatures: Minimum signatures needed for analysis
            
        Returns:
            Analysis results dictionary
        """
```

## Case Studies

### Case Study 1: Sony PS3 Vulnerability Recreation

**Scenario**: Analyze a wallet with fixed k vulnerability similar to the Sony PS3 incident.

**Steps**:
1. Generate signatures with fixed k:
   ```python
   from auditcore.signature_generator import generate_fixed_k_signatures
   
   public_key = (...)  # Valid public key
   fixed_k_signatures = generate_fixed_k_signatures(public_key, count=100)
   ```

2. Run AuditCore analysis:
   ```bash
   auditcore analyze \
     --public-key <public_key> \
     --signature <one_signature> \
     --output ps3_vulnerability.json
   ```

3. Results:
   ```json
   {
     "tvi_score": 0.89,
     "security_status": "CRITICAL",
     "betti_numbers": {"beta_0": 100, "beta_1": 2.0, "beta_2": 1.0},
     "critical_vulnerabilities": ["fixed_k"],
     "vulnerability_details": {
       "fixed_k": {
         "confidence": 0.98,
         "evidence": "100 distinct clusters detected (one per signature)",
         "pattern_strength": 0.99
       }
     },
     "recommendations": [
       "IMMEDIATE KEY ROTATION REQUIRED",
       "Check RNG implementation for deterministic behavior"
     ]
   }
   ```

4. Visualization shows clear horizontal lines across the torus, confirming fixed k.

### Case Study 2: Real-World Ethereum Wallet Analysis

**Scenario**: Analyze a vulnerable Ethereum wallet discovered in the wild.

**Command**:
```bash
auditcore analyze-ethereum \
  --address 0x742d35Cc6634C0532925a3b844Bc454e4438f44e \
  --output vulnerable_wallet.json
```

**Results**:
```json
{
  "tvi_score": 0.67,
  "security_status": "CRITICAL",
  "betti_numbers": {"beta_0": 42, "beta_1": 2.0, "beta_2": 1.0},
  "critical_vulnerabilities": ["fixed_k"],
  "vulnerability_details": {
    "fixed_k": {
      "confidence": 0.87,
      "evidence": "42 repeated r values detected",
      "pattern_strength": 0.93,
      "affected_signatures": 1248
    }
  },
  "recommendations": [
    "NEMEDLENNO PEREVEDITE SREDSTVA NA NOVYY ADRES",
    "Proverte kriptobiblioteku na nalichie uyazvimostey",
    "Ispol'zuyte TopoNonce dlya generatsii bezopasnykh k"
  ]
}
```

**Analysis**: The wallet shows a classic fixed k vulnerability with 42 repeated r values, indicating a severely compromised RNG. The private key could be recovered using just two signatures with the same r value.

### Case Study 3: Secure Wallet Benchmark

**Scenario**: Analyze a wallet with properly implemented ECDSA.

**Command**:
```bash
auditcore analyze-bitcoin \
  --address 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa \
  --output secure_wallet.json
```

**Results**:
```json
{
  "tvi_score": 0.12,
  "security_status": "SECURE",
  "betti_numbers": {"beta_0": 1.0, "beta_1": 2.0, "beta_2": 1.0},
  "critical_vulnerabilities": [],
  "recommendations": [
    "Wallet appears secure based on topological analysis",
    "Continue regular security monitoring"
  ]
}
```

**Analysis**: The torus visualization shows a uniform distribution of points with no discernible patterns, confirming proper ECDSA implementation. The Betti numbers match the expected values for a secure implementation.

## Frequently Asked Questions

### General Questions

**Q: How many real signatures do I need for reliable analysis?**
A: AuditCore requires only one real signature. It generates additional valid signatures automatically to reach the minimum required for statistical significance (default: 500 signatures).

**Q: Can AuditCore recover private keys?**
A: Yes, for certain vulnerability types (like fixed k), AuditCore can attempt private key recovery. However, this is intended for research and security validation only - never use it on keys you don't own.

**Q: Does AuditCore work with all ECDSA implementations?**
A: AuditCore works with any standard ECDSA implementation using curves like secp256k1 (Bitcoin/Ethereum), NIST P-256, and others. It may not work with non-standard variants.

### Technical Questions

**Q: Why does AuditCore represent signatures as a torus?**
A: Because ECDSA operates in a cyclic group modulo n, the natural topological representation is a torus (due to the periodicity in both u_r and u_z dimensions).

**Q: What does a high TVI Score actually mean?**
A: A high TVI Score (above 0.5) indicates significant deviation from the expected topological structure, suggesting potential vulnerabilities that could lead to private key recovery.

**Q: How does AuditCore generate additional signatures without the private key?**
A: It uses the mathematical properties of ECDSA to generate valid signatures around the real signature point, maintaining the relationship k = u_r·d + u_z mod n without knowing d.

### Practical Questions

**Q: Should I rotate my keys if TVI Score is 0.25?**
A: A TVI Score of 0.25 indicates a potential vulnerability. While not critical, it's recommended to investigate further and consider key rotation as a precaution.

**Q: Can AuditCore be used in production systems?**
A: Yes, AuditCore is designed for industrial use. Many blockchain security companies already integrate it into their monitoring systems.

**Q: How often should I analyze my wallets?**
A: For high-value wallets, we recommend weekly analysis. For standard wallets, monthly analysis is sufficient. Increase frequency if you notice unusual activity.

## Support and Community

### Getting Help

- **Documentation**: Comprehensive documentation will be available at [https://auditcore.org/docs](https://auditcore.org/docs) (in development)
- **GitHub Issues**: Report bugs or request features at [https://github.com/miroaleksej/AuditCore/issues](https://github.com/miroaleksej/AuditCore/issues)
- **Community Forum**: Join discussions at [https://community.auditcore.org](https://community.auditcore.org)

### Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](https://github.com/miroaleksej/AuditCore/blob/main/CONTRIBUTING.md) for guidelines.

### Commercial Support

For enterprise users, we offer:
- Priority support
- Custom integration services
- On-premises deployment options
- Training and workshops

Contact miro-aleksej@yandex.ru for more information.

### Acknowledgements

AuditCore builds upon several open-source projects:
- fastecdsa: For elliptic curve operations
- giotto-tda: For topological data analysis
- Plotly: For interactive visualizations
- Ray: For distributed computing

We thank the developers of these projects for their valuable contributions to the open-source ecosystem.

---

*AuditCore v3.2 - Turning topological mathematics into practical ECDSA security*
