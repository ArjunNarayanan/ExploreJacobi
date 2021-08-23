function run_gradient_descent_iterations(func, grad, x0, lr, numiter)
    xprev = copy(x0)
    xnext = copy(x0)
    for iter = 1:numiter
        f = func(xprev)
        g = grad(xprev)

        xnext .= xprev - lr * f / (g' * g) * g
        xprev .= xnext
    end
    return xnext
end
