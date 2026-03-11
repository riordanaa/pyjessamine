using Jessamine
using Random

X = randn(100, 2)
y = X[:, 1].^2 + 2*X[:, 2]

try
    result = jessamine_fit(X, y;
        output_size=3, scratch_size=3, parameter_size=1,
        num_time_steps=2, max_epochs=2, num_to_keep=5,
        num_to_generate=10, max_time=120, random_seed=42,
        verbosity=0)
    println("Success! Rating: ", result.rating)

    y_pred = jessamine_predict(result, X)
    println("Predictions length: ", length(y_pred))

    sym_str = jessamine_symbolic_string(result)
    println("Symbolic: ", sym_str)

    c = jessamine_complexity(result)
    println("Complexity: ", c)
catch e
    println("Error: ")
    showerror(stdout, e, catch_backtrace())
end
