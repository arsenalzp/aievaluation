"""Multi-turn agent evaluation for the EvalRun agent runner.

Layering (see README, "Multi-turn agent evaluation"):

* ``models`` / ``canonical`` / ``dataset`` / ``checks`` / ``gate`` are framework neutral and
  use only the standard library.
* ``phoenix_adapter`` and ``deepeval_provider`` are the only modules that know about Phoenix /
  OpenInference and DeepEval respectively.
"""
