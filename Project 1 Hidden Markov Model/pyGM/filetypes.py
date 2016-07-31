"""
pyGM/filetypes.py

Read methods for graphical model file types (UAI, WCSP, etc.)

Version 0.0.1 (2015-09-28)
(c) 2015 Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

import numpy as np;
from sortedcontainers import SortedSet as sset;
from .factor import *


def readFileByTokens(path, specials=[]):
  """Helper function for file IO"""
  with open(path, 'r') as fp:
    buf = []
    while True:
      ch = fp.read(1)
      if ch == '':
        break
      elif ch in specials:
        if buf:
          yield ''.join(buf)
        buf = []
        yield ch
      elif ch.isspace():
        if buf:
          yield ''.join(buf)
          buf = []
      else:
        buf.append(ch)



class FileTokenizer:
  """Helper function for file IO"""
  def __init__(self, path):
    self.name = path
    self.fh = open(path)
    self.eof = False
  def __enter__(self):
    return self
  def __exit__(self,type,value,traceback):
    self.close()
  def close(self):
    self.fh.close()
    self.eof = True
  def next(self):
    buf = []
    while not self.eof:
      ch = self.fh.read(1)
      if (ch == ''):
        self.eof=True
        break
      if (ch.isspace() and buf):
        break
      else:
        buf.append(ch)
    return ''.join(buf)

            
"""
with FileTokenizer('tst.txt') as tok:
  while not tok.eof:
    print tok.next()
"""



def readUai10(filename):
  """Read in a collection (list) of factors specified in UAI 2010 format"""
  dims = []
  i = 0
  cliques = []
  factors = []
  evid = {}

  gen = readFileByTokens(filename)
  type = gen.next()       # type = Bayes,Markov,Sparse,etc
  nVar = int(gen.next())
  for i in range(nVar):
    dims.append( int(gen.next()) )
  nCliques = int(gen.next())
  for c in range(nCliques):
    cSize = int(gen.next())
    cliques.append([])
    for i in range(cSize):
      v = int(gen.next())
      cliques[-1].append( Var(v,dims[v]) )
    #print cliques[-1]
  for c in range(nCliques):
    tSize = int(gen.next())
    vs = VarSet(cliques[c])
    assert( tSize == vs.nrStates() )
    factors.append(Factor(vs))
    factorSize = tuple(d for d in (v.states for v in cliques[c])) if len(cliques[c]) else (1,)
    pi = list(map(lambda x:vs.index(x), cliques[c])) 
    ipi = list(pi)
    for j in range(len(pi)):
      ipi[pi[j]] = j
    #print 'Building %s : %s,%s : %s'%(cliques[c],factorSize,vs,tSize)
    for tup in np.ndindex(factorSize):  # automatic uai order?
      tok = gen.next()
      #print "%s => %s: %s"%(tup,tuple(tup[ipi[j]] for j in range(len(ipi))),tok)
      if (tok == '('):    # check for "sparse" (run-length) representation
        run, comma, val, endparen = gen.next(), gen.next(), gen.next(), gen.next()
        assert(comma == ',' and endparen==')')
        for r in range(run):
          mytup = tuple(tup[ipi[j]] for j in range(len(ipi)))
          factors[-1][mytup] = float(val)
      else:               # otherwise just a list of values in the table
        mytup = tuple(tup[ipi[j]] for j in range(len(ipi)))
        factors[-1][mytup] = float(tok)
  
  # TODO: read evidence
  # condition on evidence
  # return graphmodel object? 
  return factors

# TODO: implement
def readErgo(filename):
  return
def readWCSP(filename):
  return



