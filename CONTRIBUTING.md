# Contributing to Squat Analyzer

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Abhishekgupta1223/squat-analyzer.git
cd squat-analyzer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/ tests/

# Type checking
mypy src/
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use descriptive variable names

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request

## Commit Message Format

```
<type>: <short summary>

<detailed description if needed>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Adding New Biomechanical Rules

1. Create a new rule class in `src/squat_analyzer/analysis/biomechanics.py`
2. Inherit from `BiomechanicsRule` base class
3. Implement `evaluate()` method
4. Add tests in `tests/test_biomechanics.py`
5. Register in `BiomechanicsEngine`

## Reporting Issues

- Use GitHub Issues
- Include reproduction steps
- Attach relevant logs or screenshots
- Specify your environment (OS, Python version)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
