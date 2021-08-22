using LinearAlgebra
include("poisson_linear_system.jl")
include("simpson_integrate.jl")

function poisson_error(numelements)
    xL = -1.0
    xR = +1.0
    wavenumber = pi / 2
    bcleft = bcright = 0.0

    numpts = numelements + 1
    stepsize = (xR - xL) / numelements
    xrange = range(xL, stop = xR, length = numpts)

    rhs = cos.(wavenumber * xrange)
    op = -poisson_linear_system(numpts)
    rhs[1] = bcleft / stepsize^2
    rhs[end] = bcright / stepsize^2
    sol = stepsize^2 * (op \ rhs)
    exactsolution =
        1.0 / wavenumber^2 * (cos.(wavenumber * xrange) .- cos(wavenumber))
    err = simpson_integrate(sol - exactsolution, stepsize)
    return err
end

function convergence_rate(err,dx)
    return diff(log.(err)) ./ diff(log.(dx))
end

powers = 1:10
numelements = 2 .^ powers
stepsize = 2.0 ./ numelements
err = poisson_error.(numelements)
rate = convergence_rate(err,stepsize)

# using Plots
# plot(stepsize,err,xscale = :log, yscale = :log)
# scatter!(stepsize,err)
