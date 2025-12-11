# E.R.O.S Testing Guide

## Overview

The E.R.O.S testing suite is organized according to best practices with comprehensive coverage measurement and reporting.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests (isolated component testing)
│   ├── core/                   # Core system components
│   ├── agents/                 # Agent implementations
│   ├── dashboard/              # Dashboard components
│   ├── visualization/          # Visualization components
│   ├── tools/                  # Tool system
│   ├── monitoring/             # Monitoring and logging
│   ├── logos/                  # Logos layer (reasoning)
│   ├── pathos/                 # Pathos layer (emotional)
│   ├── memory/                 # Memory system
│   └── ethos/                  # Ethos framework (ethics)
└── integration/                # Integration tests (system-level testing)
    ├── test_integration.py
    ├── test_thread_integration.py
    └── test_thread_management.py
```

## Running Tests

### Quick Commands

```bash
# Run all tests with coverage
make test

# Run only unit tests
make test-unit

# Run only integration tests
make test-integration

# Generate HTML coverage report
make coverage-html

# Run tests quickly
make test-fast
```

### Using the Test Runner

```bash
# Run all tests with coverage
python run_tests.py

# Run only unit tests
python run_tests.py --unit

# Run only integration tests
python run_tests.py --integration

# Run without coverage (faster)
python run_tests.py --no-cov

# Generate HTML coverage report
python run_tests.py --html

# Verbose output
python run_tests.py --verbose
```

### Direct pytest Usage

```bash
# Run all tests
pytest

# Run specific test directory
pytest tests/unit/core/

# Run specific test file
pytest tests/unit/logos/test_logos_layer.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run with specific markers
pytest -m unit
pytest -m integration
pytest -m slow
```

## Coverage Requirements

- **Minimum Coverage**: 80% (enforced by pytest configuration)
- **Coverage Reports**: Generated in multiple formats
  - Terminal output with missing lines
  - HTML report in `htmlcov/` directory
  - XML report for CI/CD integration

## Test Categories

### Unit Tests
- Test individual components in isolation
- Use mocks and fixtures for dependencies
- Fast execution (< 1 second per test)
- Located in `tests/unit/`

### Integration Tests
- Test component interactions
- May use real dependencies
- Slower execution allowed
- Located in `tests/integration/`

### Property-Based Tests
- Use Hypothesis for property-based testing
- Marked with `@pytest.mark.property`
- Generate random test cases

## Test Markers

```python
@pytest.mark.unit          # Unit test
@pytest.mark.integration   # Integration test
@pytest.mark.property      # Property-based test
@pytest.mark.slow          # Slow running test
```

## Writing Tests

### Test File Naming
- Unit tests: `test_<component_name>.py`
- Integration tests: `test_<feature>_integration.py`
- Follow the source code structure

### Test Function Naming
```python
def test_<function_name>_<scenario>():
    """Test description"""
    pass

def test_<function_name>_<scenario>_<expected_result>():
    """Test description"""
    pass
```

### Using Fixtures

```python
def test_agent_initialization(agent_config):
    """Test agent initializes correctly with config"""
    agent = AutonomousAgent(agent_config)
    assert agent.is_initialized()

def test_pathos_state_update(sample_pathos_state):
    """Test pathos state updates correctly"""
    # Use the fixture
    assert sample_pathos_state.internal_reward == 0.5
```

## Coverage Configuration

Coverage is configured in `.coveragerc`:

```ini
[run]
source = src
omit = 
    */tests/*
    */test_*
    */__pycache__/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:

[html]
directory = htmlcov
title = E.R.O.S Coverage Report
```

## Continuous Integration

Tests run automatically on:
- Push to main/develop branches
- Pull requests
- Multiple Python versions (3.9, 3.10, 3.11)

See `.github/workflows/test.yml` for CI configuration.

## Best Practices

### Test Organization
1. **Mirror source structure**: Test organization should match `src/` structure
2. **Separate concerns**: Unit tests vs integration tests
3. **Use descriptive names**: Test names should describe the scenario

### Test Quality
1. **Test one thing**: Each test should verify one specific behavior
2. **Use fixtures**: Share common setup code via fixtures
3. **Mock dependencies**: Unit tests should not depend on external systems
4. **Assert clearly**: Use descriptive assertion messages

### Performance
1. **Fast unit tests**: Keep unit tests under 1 second
2. **Mark slow tests**: Use `@pytest.mark.slow` for longer tests
3. **Parallel execution**: Tests should be independent and parallelizable

### Coverage
1. **Aim for 80%+**: Maintain high coverage but focus on quality
2. **Test edge cases**: Cover error conditions and boundary cases
3. **Exclude appropriately**: Use `pragma: no cover` sparingly

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `src/` is in Python path
2. **Fixture not found**: Check fixture is in `conftest.py` or imported
3. **Coverage too low**: Add tests for uncovered code paths
4. **Slow tests**: Use mocks to avoid real I/O operations

### Debug Mode

```bash
# Run with verbose output and no capture
pytest -v -s

# Run specific test with debugging
pytest -v -s tests/unit/core/test_agent.py::test_agent_initialization

# Drop into debugger on failure
pytest --pdb
```

## Reporting Issues

When tests fail:
1. Check the error message and traceback
2. Verify test environment setup
3. Run individual failing tests with `-v -s`
4. Check coverage report for missing test cases
5. Review recent code changes that might affect tests