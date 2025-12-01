# Contributing to Unified ORCHID Database Builder

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Unified ORCHID Database Builder.

## How to Contribute

### Reporting Issues

If you encounter a bug or have a feature request:

1. **Check existing issues** to see if it's already been reported
2. **Create a new issue** with a clear title and description
3. **Include relevant information**:
   - Python version
   - Operating system
   - Error messages or unexpected behavior
   - Steps to reproduce the issue

### Suggesting Enhancements

We welcome suggestions for improvements:

1. **Open an issue** with the label "enhancement"
2. **Describe the enhancement** clearly
3. **Explain the use case** and why it would be valuable
4. **Provide examples** if applicable

### Submitting Code Changes

1. **Fork the repository**
2. **Create a new branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the code style guidelines below
4. **Test your changes** thoroughly
5. **Commit your changes** with clear, descriptive messages:
   ```bash
   git commit -m "Add feature: description of what you added"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a pull request** with a clear description of your changes

## Code Style Guidelines

### Python Code Style

- Follow **PEP 8** style guidelines
- Use **descriptive variable names**
- Add **docstrings** to functions and classes
- Keep functions **focused and modular**
- Use **type hints** where appropriate

### Example:
```python
def compute_temporal_features(data: pd.DataFrame, 
                              patient_id: str, 
                              variable: str) -> dict:
    """
    Compute temporal statistics for a clinical variable.
    
    Args:
        data: DataFrame containing time-series measurements
        patient_id: Unique patient identifier
        variable: Name of the clinical variable
        
    Returns:
        Dictionary of temporal features (mean, slope, etc.)
    """
    # Implementation here
    pass
```

### Documentation

- Update **README.md** if you add new features
- Add **inline comments** for complex logic
- Update **requirements.txt** if you add dependencies

## Areas for Contribution

We welcome contributions in these areas:

### Feature Engineering
- Additional organ-specific features
- New temporal statistics
- Clinical risk scores (KDPI, MELD, etc.)

### Performance Optimization
- Further speed improvements
- Memory efficiency
- Parallel processing enhancements

### Data Quality
- Additional validation checks
- Improved imputation strategies
- Enhanced missingness analysis

### Documentation
- Usage examples
- Tutorials
- API documentation

### Testing
- Unit tests
- Integration tests
- Validation against known results

## Questions?

If you have questions about contributing, feel free to:
- Open an issue with the "question" label
- Contact Noah at noah@2460.life

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome diverse perspectives
- Focus on constructive feedback
- Prioritize the community's best interests

### Unacceptable Behavior

- Harassment or discriminatory language
- Personal attacks
- Publishing others' private information
- Other conduct inappropriate in a professional setting

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Thank you for helping improve this project! Your contributions support fairness-aware machine learning research in organ donation.
