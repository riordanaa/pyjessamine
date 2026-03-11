#!/usr/bin/env -S julia --project=@.

using Pkg
using PkgHelpers

# Set up the [compat] section of Project.toml
# so that each package is marked as requiring the currently installed
# major and minor versions
# as in "~1.2", meaning that only patchlevel updates are allowed
PkgHelpers.freeze(Pkg; relaxed=true)