import sys


def test_python_version_is_supported():
    assert sys.version_info >= (3, 11)


def test_core_dependencies_import():
    import matplotlib
    import pandas
    import pydantic
    import streamlit

    assert matplotlib.__version__
    assert pandas.__version__
    assert pydantic.__version__
    assert streamlit.__version__
