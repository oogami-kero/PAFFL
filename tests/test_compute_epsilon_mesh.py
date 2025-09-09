import os
import sys
import types
from unittest import mock

prv_accountant_stub = types.SimpleNamespace(
    Accountant=object(),
    prv=types.SimpleNamespace(GaussianMechanism=object()),
    accountant=types.SimpleNamespace(Domain=object(), PRVAccountant=object()),
    discretisers=types.SimpleNamespace(ExplicitDomain=object()),
)
sys.modules['prv_accountant'] = prv_accountant_stub

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dp_utils import compute_epsilon


def failing_accountant(*args, **kwargs):
    raise RuntimeError('fail')


class LowLevelStub:
    def compute_epsilon(self, steps):
        return 0.25


with mock.patch('prv_accountant.Accountant', side_effect=failing_accountant), \
     mock.patch('prv_accountant.prv.GaussianMechanism'), \
     mock.patch('prv_accountant.accountant.Domain'), \
     mock.patch('prv_accountant.discretisers.ExplicitDomain'), \
     mock.patch('prv_accountant.accountant.PRVAccountant', return_value=LowLevelStub()):
    eps = compute_epsilon(10, 1.0, 1e-5, accountant='prv')
    print('lowlevel_eps', eps)


def stub_accountant(*args, **kwargs):
    assert kwargs.get('mesh_size') == 0.01

    class Stub:
        def compute_epsilon(self, steps):
            return 0.5

    return Stub()


with mock.patch('prv_accountant.Accountant', side_effect=stub_accountant):
    eps = compute_epsilon(10, 1.0, 1e-5, accountant='prv', mesh_size=0.01)
    print('custom_eps', eps)
