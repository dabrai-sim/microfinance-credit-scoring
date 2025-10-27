"""
Pytest configuration and shared fixtures
"""
import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))


@pytest.fixture(autouse=True)
def reset_matplotlib():
    """Reset matplotlib backend after each test"""
    import matplotlib
    matplotlib.use('Agg')
    yield
    matplotlib.pyplot.close('all')
