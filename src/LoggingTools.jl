export fancy_interactive_logging

"""
    fancy_interactive_logging(level = Logging.Info)

Set the global logger that filters out messages that aren't from
the main module or Jessamine, and that prepends the current time
to each message. Messages are only logged if they are at or above
the specified level.
"""
function fancy_interactive_logging(level = Logging.Info)
    global original_logger
    if isnothing(original_logger)
        original_logger = global_logger()
    end
    message_improver = TransformerLogger(original_logger) do m
        msg = "$(now()): $(m.message)"
        return (
            level = m.level,
            message = msg,
            _module = m._module,
            group = m.group,
            id = m.id,
            file = m.file,
            line = m.line,
            kwargs = m.kwargs
        )
    end
    filtered_logger = ActiveFilteredLogger(message_improver) do log_args
        ((log_args._module == Main
          || log_args._module == Jessamine
          || log_args._module == @__MODULE__)
         &&
         log_args.level >= level)
    end
    # The default loggers block Debug messages unless JULIA_DEBUG is set.
    if level == Logging.Debug && !haskey(ENV, "JULIA_DEBUG")
        ENV["JULIA_DEBUG"] = "Main,Jessamine,$(@__MODULE__)"
    end
    global_logger(filtered_logger)
end

original_logger = nothing