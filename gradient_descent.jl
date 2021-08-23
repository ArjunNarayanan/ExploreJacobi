function run_gradient_descent_iterations(func, grad, x0, lr, numiter; rtol = 1e5)
    xprev = copy(x0)
    xnext = copy(x0)

    g = grad(xprev)
    gnorm = g'*g
    iter = 1
    tol = rtol*eps()

    while gnorm > tol && iter < numiter
        f = func(xprev)
        g = grad(xprev)
        gnorm = g'*g

        xnext .= xprev - lr * (f / gnorm) * g
        xprev .= xnext

        iter += 1
    end
    return xnext
end
