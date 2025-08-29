# Contributing to AuditCore

Thank you for your interest in contributing to AuditCore! This document outlines the guidelines and processes for contributing to the project. By participating, you agree to abide by our [Code of Conduct](https://github.com/miroaleksej/AuditCore/blob/main/Doc/Conduct.md).

## Table of Contents
- [Reporting Issues](#reporting-issues)
- [Suggesting Features](#suggesting-features)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Documentation Guidelines](#documentation-guidelines)
- [License Considerations](#license-considerations)

## Reporting Issues

Before reporting a new issue, please:
1. Search [existing issues](https://github.com/miroaleksej/AuditCore/issues) to check if it's already been reported
2. Verify you're using the latest version of AuditCore

When reporting an issue:
- Provide a clear, descriptive title
- Include steps to reproduce the problem
- Specify your environment (OS, Python version, dependencies)
- Include relevant logs or error messages
- For security vulnerabilities, follow our [security disclosure process](#security-disclosure)

### Security Disclosure

If you've discovered a security vulnerability, please **do not** create a public issue. Instead, email security@*** with:
- A detailed description of the vulnerability
- Steps to reproduce
- Your suggested fix (optional)
- Your contact information

We will acknowledge receipt within 48 hours and work with you to resolve the issue responsibly.

## Suggesting Features

We welcome feature suggestions! When proposing a new feature:
1. Explain the problem you're trying to solve
2. Describe your proposed solution
3. Include any relevant research or alternatives you've considered
4. Note if you're willing to implement the feature yourself

For major architectural changes, consider opening a discussion first to align with the project vision.

## Development Setup

To set up a development environment:

```bash
# Clone the repository
git clone https://github.com/miroaleksej/AuditCore.git
cd AuditCore

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (for code quality)
pre-commit install
```

### Running Tests

AuditCore uses pytest for testing. To run the test suite:

```bash
pytest tests/ --cov=auditcore
```

For topological analysis tests specifically:

```bash
pytest tests/topological/ --verbose
```

## Coding Standards

AuditCore follows these coding standards:

- **PEP 8** for code style
- **Type hints** for all public functions
- **Docstrings** in Google format for all modules, classes, and public methods
- **Meaningful test names** that describe what's being tested
- **No hardcoded values** - use constants from `auditcore.constants`
- **Modular design** - each module should have a single responsibility

Example of a well-documented function:

```python
def analyze_topology(signatures: List[Tuple[int, int, int]], 
                    n: int) -> Dict[str, float]:
    """Computes topological invariants from ECDSA signature space.
    
    Args:
        signatures: List of (r, s, z) signature tuples
        n: Order of the elliptic curve subgroup
        
    Returns:
        Dictionary containing Betti numbers and TVI score:
        {
            "beta_0": float,  # Number of connected components
            "beta_1": float,  # Number of independent cycles
            "beta_2": float,  # Number of voids
            "tvi_score": float  # Torus Vulnerability Index
        }
        
    Raises:
        ValueError: If signatures list is empty or contains invalid values
        RuntimeError: If topological computation fails
    """
```

## Testing Requirements

All contributions must include appropriate tests:
- **Unit tests** for new functionality
- **Integration tests** for module interactions
- **Regression tests** for bug fixes

Test coverage should not decrease. We aim for ≥85% test coverage across the codebase.

When submitting a PR:
1. Ensure all existing tests pass
2. Add tests for new functionality
3. Verify coverage hasn't decreased (`pytest --cov=auditcore`)

## Pull Request Process

1. **Fork** the repository and create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement** your changes following the coding standards

3. **Write tests** for your changes

4. **Update documentation** as needed

5. **Commit** your changes with a descriptive message:
   ```
   feat(topological): add adaptive sigma calculation for signature generation
   
   - Implemented adaptive sigma adjustment based on signature density
   - Added tests for high-density and low-density scenarios
   - Updated documentation for SignatureGenerator
   ```

6. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** with:
   - Clear description of changes
   - Reference to related issue (if applicable)
   - Test results
   - Any additional context needed for review

8. **Respond to review comments** promptly

We aim to review PRs within 7 business days.

## Documentation Guidelines

AuditCore uses Sphinx for documentation. When contributing:

- Update docstrings for any modified functionality
- Add new documentation for major features in `docs/`
- Use clear examples where appropriate
- Maintain consistent terminology
- Avoid jargon when possible, or define it when necessary

For module documentation, include:
- Purpose and functionality
- Key algorithms used
- Input/output specifications
- Example usage
- Limitations and edge cases

## License Considerations

By contributing to AuditCore, you agree that your contributions will be licensed under the [MIT License](LICENSE).

All new files must include the MIT license header:

```python
# Copyright (c) 2023 AuditCore Contributors
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
```

## Additional Resources

- [Project Architecture Overview](ARCHITECTURE.md)
- [Mathematical Foundations](MATHEMATICS.md)
- [Issue Templates](.github/ISSUE_TEMPLATE)
- [PR Template](.github/PULL_REQUEST_TEMPLATE.md)

Thank you for contributing to AuditCore! Your efforts help improve ECDSA security analysis for the entire community.
