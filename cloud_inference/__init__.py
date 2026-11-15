"""PrintDiag cloud inference service package.

Runs on the GPU host behind an SSH tunnel, bound to 127.0.0.1 only, and serves
text generation for the Base model or the Base+LoRA adapter. It never talks to
the Marlin Simulator and never exposes device control.
"""
