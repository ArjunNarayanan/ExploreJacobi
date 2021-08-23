using ForwardDiff
using LinearAlgebra
include("poisson_linear_system.jl")
include("gradient_descent.jl")
include("learned_jacobi.jl")


function optimize_parameters(
    numelements,
    wavenumber,
    guessparams;
    lr = 1.0,
    gditers = 20,
)
    numiter = numelements^2

    xL, xR = -1.0, 1.0
    bcleft, bcright = 0.0, 0.0

    stepsize = (xR - xL) / numelements
    numpts = numelements + 1
    xrange = range(xL, stop = xR, length = numpts)

    rhs = cos.(wavenumber * xrange)

    op = -poisson_linear_system(numpts)
    exactsol = solve_direct(op, rhs, bcleft, bcright, stepsize)

    func(x) = LearnedJacobi.error_for_parameters(
        bcleft,
        bcright,
        rhs,
        stepsize,
        numiter,
        exactsol,
        x,
    )
    grad(x) = ForwardDiff.gradient(func, x)

    optimumparams = run_gradient_descent_iterations(func, grad, p0, lr, gditers)

    itersol = LearnedJacobi.run_parametric_iterations(
        bcleft,
        bcright,
        rhs,
        stepsize,
        numiter,
        optimumparams,
    )

    return optimumparams, itersol, exactsol
end


numelements = 32
# jacobiparams = [0.5, 0.0, 0.5, 0.5]
p0 = [0.0, 0.0, 0.0, 0.0]
wavenumber = pi / 2

optimumparams, itersol, exactsol = optimize_parameters(numelements,wavenumber,p0)
