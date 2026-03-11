# Development notes for `Jessamine.jl`

## 2024-08-07

### New features

- Added code to do symbolic calculations using `SymPy.jl` as well as `Symbolics.jl`
- Since modules `SymPy` and `Symbolics` are now in use, functions with the same name in both, like `simplify`, are ambiguous.
So you have to call `Symbolics.simplify(...)` and `SymPy.simplify(...)`
- Added hop mutations: Each instruction has some probability of hopping to the end of another instruction block.
The default probability is 0, so this *shouldn't* change anything but you never know.

### Breaking changes

- Changed `num_generations` to `max_generations`
- Changed `num_epochs` to `max_epochs`
- Added option to stop at a given time (deadline)
- Added option to stop when a signal is received on a `Channel`
- Added a `condition` field to `Population` to indicate whether its evolution is still in progress, or why it stopped.
- Refactored `Operations.jl` to make it easier to add unary functions like `sin`, `exp`, etc.

### Behind the scenes

- Rewrote test scripts to make debugging easier
