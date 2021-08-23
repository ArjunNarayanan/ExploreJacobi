using ForwardDiff
using LinearAlgebra
include("poisson_linear_system.jl")
include("gradient_descent.jl")

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

function run_parametric_iterations(u0, rhs, stepsize, numiter, parameters)
    T = eltype(parameters)
    uprev = T.(u0)
    unext = T.(u0)
    for iter = 1:numiter
        step_iteration!(unext, uprev, rhs, stepsize, parameters)
        uprev .= unext
    end
    return unext
end

function error_for_parameters(u0, rhs, stepsize, numiter, uexact, parameters)

    itersol = run_parametric_iterations(u0, rhs, stepsize, numiter, parameters)
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

function solve_direct(operator, rhs, bcleft, bcright, stepsize)
    numpts = length(rhs)
    copyrhs = copy(rhs)
    copyrhs[1] = bcleft / stepsize^2
    copyrhs[numpts] = bcright / stepsize^2
    sol = stepsize^2 * (operator \ copyrhs)
    return sol
end

function solve_direct_all_rhs(operator, bcleft, bcright, stepsize)
    ndofs, ndofs = size(operator)
    rhs = diagm(ones(ndofs))[:, 2:ndofs-1]
    rhs[1, :] .= bcleft / stepsize^2
    rhs[ndofs, :] .= bcright / stepsize^2
    sol = stepsize^2 * (operator \ rhs)
    return sol
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


numelements = 32
parameters = [0.5, 0.0, 0.5, 0.5]
# parameters = [0.,1.,0.,1.]
numiter = numelements^2

xL, xR = -1.0, 1.0
bcleft, bcright = 0.0, 0.0
# wavenumber = pi / 2

stepsize = (xR - xL) / numelements
numpts = numelements + 1
xrange = range(xL, stop = xR, length = numpts)

op = -poisson_linear_system(numpts)
exactsol = solve_direct_all_rhs(op, bcleft, bcright, stepsize)
itersol = run_parametric_iterations_all_rhs(
    bcleft,
    bcright,
    stepsize,
    numiter,
    parameters,
)


err0 = error_for_parameters_all_rhs(
    bcleft,
    bcright,
    stepsize,
    numiter,
    exactsol,
    parameters,
)

func(x) = error_for_parameters_all_rhs(
    bcleft,
    bcright,
    stepsize,
    numiter,
    exactsol,
    x,
)
grad(x) = ForwardDiff.gradient(func, x)

newparams = run_gradient_descent_iterations(func, grad, parameters, 1.0, 50)

err1 = error_for_parameters_all_rhs(
    bcleft,
    bcright,
    stepsize,
    numiter,
    exactsol,
    newparams,
)
