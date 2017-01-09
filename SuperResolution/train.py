# -*- coding: utf-8 -*-
import caffe
import sys

solver_path = 'examples/SRCNN/SRCNN_solver.prototxt'
solver = caffe.SGDSolver(solver_path)
sys.stdout = solver.solve()