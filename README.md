
A runnable corpus-based text generator in which next-token candidates are
scored by an n-gram language model and filtered/reweighted by several
non-isomorphic process fragments.

Conceptual mapping:
  F       reference next-token distribution from the n-gram model
  g_k     fragment-specific transforms of candidate/context features
  I_k     local approximate inverse/lift (not a global inverse)
  lambda  smooth adiabatic continuation from exploratory to selective scoring
  D       proposed "diopathane" consistency score
  tau     maximum permitted D score for a candidate

This is an experimental algorithmic analogy, not an established physical
model. "Diopathane" is a user-defined name for the consistency constraint.

Examples:
  python nonisomorphic_adiabatic_textgen.py corpus.txt --prompt "once upon a" --tokens 80
  python nonisomorphic_adiabatic_textgen.py --prompt "The machine" --tokens 60
  python nonisomorphic_adiabatic_textgen.py corpus.txt --order 4 --beam 8 --tau 0.65

Only dependency: NumPy
Install: python -m pip install numpy
