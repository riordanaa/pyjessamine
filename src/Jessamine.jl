module Jessamine

using Dates
using Distributions
using LinearAlgebra
using Logging
using LoggingExtras
using MLJ
using MLJLinearModels
using MLJModelInterface
using NamedTupleTools
using Optim
using Optimization
using OptimizationOptimJL
const _USE_PYCALL = !haskey(ENV, "JESSAMINE_NO_PYCALL")
if _USE_PYCALL
    using PyCall
end
using Random
using SciMLBase
using Serialization
using Symbolics
if _USE_PYCALL
    using SymPy
    using SymPyCore
end
using Tables
using TermInterface
using Base.Threads

include("LoggingTools.jl")
include("TablesUtil.jl")
include("GenomeCore.jl")
include("RandomDuplicateDelete.jl")
include("Mutation.jl")
include("Recombination.jl")
include("Operations.jl")
include("Evolution.jl")
include("SymbolicForm.jl")
if _USE_PYCALL
    include("SymPyForm.jl")
end
include("AbstractModelLayer.jl")
include("AbstractLinearModelLayer.jl")
include("RidgeLayer.jl")
include("MachineLayer.jl")
include("MLJInside.jl")
include("MLJOutside.jl")
include("TimeSeriesLayer.jl")
include("TimeSeriesRidgeLayer.jl")
include("PythonInterface.jl")

end
