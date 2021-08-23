module LearnedJacobi

function step_iteration!(unext, uprev, rhs, stepsize, parameters)
    wL, wC, wR, wF = parameters
    numpts = length(unext)
    for i = 2:numpts-1
        unext[i] =
            wL * uprev[i-1] +
            wC * uprev[i] +
            wR * uprev[i+1] +
            wF * stepsize^2 * rhs[i]
    end
end

function run_parametric_iterations(
    bcleft,
    bcright,
    rhs,
    stepsize,
    numiter,
    parameters,
)
    T = eltype(parameters)
    numpts = length(rhs)

    u0 = zeros(T, numpts)
    u0[1] = bcleft
    u0[numpts] = bcright

    uprev = copy(u0)
    unext = copy(u0)

    for iter = 1:numiter
        step_iteration!(unext, uprev, rhs, stepsize, parameters)
        uprev .= unext
    end
    return unext
end

function error_for_parameters(
    bcleft,
    bcright,
    rhs,
    stepsize,
    numiter,
    uexact,
    parameters,
)

    itersol = run_parametric_iterations(
        bcleft,
        bcright,
        rhs,
        stepsize,
        numiter,
        parameters,
    )
    err = sum((uexact - itersol) .^ 2)
    return err
end

function run_parametric_iterations_all_rhs(
    bcleft,
    bcright,
    stepsize,
    numiter,
    parameters,
)
    T = eltype(parameters)

    solution = zeros(T, numpts, numpts - 2)
    solution[1, :] .= bcleft
    solution[end, :] .= bcright

    rhs = diagm(ones(numpts))[:, 2:numpts-1]
    rows, cols = size(rhs)

    for col = 1:cols
        u0 = solution[:, col]
        solution[:, col] .= run_parametric_iterations(
            u0,
            rhs[:, col],
            stepsize,
            numiter,
            parameters,
        )
    end
    return solution
end

function error_for_parameters_all_rhs(
    bcleft,
    bcright,
    stepsize,
    numiter,
    uexact,
    parameters,
)

    itersol = run_parametric_iterations_all_rhs(
        bcleft,
        bcright,
        stepsize,
        numiter,
        parameters,
    )
    err = sum((itersol - uexact) .^ 2)
    return err
end

# end module
end
# end module
