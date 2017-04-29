# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

import itertools

from climin import RmsProp

from .losses import LogisticRegression,ComplexQuadratic
from .common import continuation


def test_rmsprop_lr():
    obj = LogisticRegression()
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = RmsProp(obj.pars, obj.fprime, 0.01, 0.9, args=args)
    for i, info in enumerate(opt):
        print(obj.f(opt.wrt, obj.X, obj.Z))
        if i > 3000:
            break
    assert obj.solved(0.15), 'did not find solution'


def test_rmsprop_continue():
    obj = LogisticRegression(n_inpt=2, n_classes=2)
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = RmsProp(
        obj.pars, obj.fprime, step_rate=0.01, momentum=.9, decay=0.9,
        args=args)

    continuation(opt)

def test_rmsprop_cquadratic():
    obj = ComplexQuadratic(10)
    opt = RmsProp(obj.pars, obj.fprime, 0.01, 0.9,momentum=0.9)
    for i, info in enumerate(opt):
        print(obj.f(opt.wrt))
        if i > 10000:
            break
    assert obj.solved(0.15), 'did not find solution'

test_rmsprop_cquadratic()
