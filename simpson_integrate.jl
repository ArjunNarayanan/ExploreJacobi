function simpson_integrate(u, stepsize)
    numpts = length(u)
    @assert iseven(numpts - 1)
    mid = div(numpts,2)
    integral = 0.0
    for i = 1:mid
        integral += stepsize / 3.0 * (u[2i-1] + 4.0 * u[2i] + u[2i+1])
    end
    return integral
end
