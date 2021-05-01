import pytest

def pytest_generate_tests(metafunc):
    if "parallel" in metafunc.fixturenames:
        metafunc.parametrize("parallel", [
            pytest.param(True, id="parallel=True"),
            pytest.param(False, id="parallel=False")
        ])