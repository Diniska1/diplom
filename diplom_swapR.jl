using LinearAlgebra
using FileIO, JLD2

eps = 1e-2

function A_scal(A, a, b)
    return (A * a)' * b
end

function print_matrix(A, msg="")
    println("Matrix ", msg)
    display(A)
end

function qr_gram_schmidt(A, B, inner_product::Function)
    m, n = size(A)
    Q = zeros(ComplexF64, m, n)
    R = zeros(ComplexF64, n, n)

    for j = 1:n
        v = A[:, j]  

        for i = 1:j-1
            R[i, j] = inner_product(B, Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        end

        R[j, j] = sqrt(real(inner_product(B, v, v)))
        Q[:, j] = v / R[j, j]
    end

    return Q, R
end

function check(T)               # find index small diag elem
    n = size(T, 1)
    ind = n + 1
    for i in 1:n
        if abs(T[i, i]) < eps / 10
            ind = i
            break
        end
    end
    return ind
end

function check_orth(A, P, P_prev)
    d1 = diag(P_prev' * A * P_prev)
    d1 = [1 / sqrt(d1[i]) for i in 1:size(d1, 1)]
    D1 = zeros(size(d1, 1), size(d1, 1))

    d2 = diag(P' * A * P)
    d2 = [1 / sqrt(d2[i]) for i in 1:size(d2, 1)]
    D2 = zeros(size(d2, 1), size(d2, 1))

    for i in size(D1, 1)
        D1[i, i] = d1[i]
    end
    for i in size(D2, 1)
        D2[i, i] = d2[i]
    end
    
    T = D2 * P' * A * P_prev * D1
    res = maximum(abs.(T))

    return res
end

function norms2(R)
    norms = Float64[]

    for i in 1:size(R, 2)
        push!(norms, sqrt(R[:, i]' * R[:, i]))
    end    

    return norms
end


function traspose_perm_matrix(a)
    n = size(a, 1)
    b = zeros(Int, n)
    for i in 1:n
        b[a[i]] = i
    end
    return b
end

function scale_columns(A, b)
    """
    res is matrix A: A[:, i] = A[:, i] * b[i] 
    """
    return A .* reshape(b, (1, length(b)))
end

function swap_converged_cols(R, X, B, perm, k, S, normsR)
    i = 1
    while i < k
        if !(i in S) && normsR[i] < eps
            # println("converged column: ", i)
            index = findfirst(x -> x == k, S)
            if !isnothing(index)
                S[index] = i
            end

            R[:, i], R[:, k] = R[:, k], R[:, i]
            B[:, i], B[:, k] = B[:, k], B[:, i]
            X[:, i], X[:, k] = X[:, k], X[:, i]
            perm[i], perm[k] = perm[k], perm[i]
            normsR[i], normsR[k] = normsR[k], normsR[i]
            k -= 1
        else
            i += 1
        end  
    end
    return R, X, B, perm, k, S
end

function swap_back(X, B, perm)
    i = 1
    while i < length(perm)
        if perm[perm[i]] == perm[i]
            i += 1
        else
            B[:, perm[perm[i]]], B[:, perm[i]] = B[:, perm[i]], B[:, perm[perm[i]]]
            X[:, perm[perm[i]]], X[:, perm[i]] = X[:, perm[i]], X[:, perm[perm[i]]]
            perm[perm[i]], perm[i] = perm[i], perm[perm[i]]
        end
    end
    return X, B
end

function BCG(A, B)
    n = size(A, 1)
    n1, m = size(B) # amount right sides
    if n1 != n
        println("DimensionMismatch, rows A != rows B")
        exit()
    end
    r = 10          # block size
    k = m           # amount not converged columns

    norms2_B = Float64[]
    norms2_B_div = Float64[]
    for i in 1:size(B, 2)
        t = sqrt(B[:, i]' * B[:, i])
        push!(norms2_B, t)
        push!(norms2_B_div, 1 / t)
    end    

    B = scale_columns(B, norms2_B_div)

    X = zeros(ComplexF64, n, m)     # starting value
    R = B - A * X               # n x m

    F = qr(R, Val(true)) 
    S = copy(F.p[1:r])
    P = copy(R[:, S])                   # n x r
    P_prev = copy(P)

    converged_columns = Int32[]
    perm = collect(1:m)
    c_norms = Float64[]
    V = Matrix{Float64}(undef, n, 0)

    iteration = 1
    maxiter = 2000
    while iteration < maxiter
        normsR = norms2(R[:, 1:k], norms2_B)
        if iteration % 25 == 0
            println("iteration = $iteration")
            println("cnorm B - AX: ", maximum(abs.(R)))
            println("max(2norm(R[i])): ", maximum(normsR))
        end

        # if iteration > 2 && iteration % 2 == 0
        #     push!(c_norms, check_orth(A, P, P_prev))
        #     P_prev = P
        # end

        AP = A * P;
        PAP = P' * AP;
        PAP = (PAP + PAP') / 2;
        L = cholesky(PAP).L;

        alpha = L' \ (L \ (P' * (R[:, 1:k])))           # r x m
        X[:, 1:k] = X[:, 1:k] + P * alpha                   # n x m
        R[:, 1:k] = R[:, 1:k] - AP * alpha               # n x m

        
        if (maximum(normsR)) < eps
            break
        end
        R, X, B, perm, k, S = swap_converged_cols(R, X, B, perm, k, S, normsR)

        _, TMP = qr(R[:, S])
        F = qr(TMP, Val(true))                 # PQR
        M = F.p
        T = F.R

        ind = check(T)                              # find index small elem on diag(T)
        if ind == size(T, 1) + 1                    # if all elements on diag(T) is large
            beta = L' \ (L \ (AP' * R[:,S]))
            Pn = R[:, S] - P * beta

            if isempty(V) == false
                AV = A * V
                gamma = (AV' * V) \ (AV' * R[:, S])
                Pn = Pn - V * gamma
            end

            beta = L' \ (L \ (AP' * Pn))
            Pn -= P * beta

            if isempty(V) == false
                AV = A * V
                gamma = (AV' * V) \ (AV' * Pn)
                Pn -= V * gamma
            end
            P = Pn;

        else
            print("\nCONVERGE: (iter = $iteration)")

            add_used_cols = [ S[F.p[i]] for i in ind:size(F.p, 1) ]
            converged_columns = [ converged_columns ; add_used_cols ]                      # indexes of columns that converged
            others_cols = setdiff( 1:k, [ converged_columns ; S[ F.p[1:ind-1] ] ] )         # indexes of cols that in R and not in converged_columns and not in current block
            
            if isempty(others_cols) == true && isempty(F.p[1:ind-1]) == true                                # stopping criteria
                println("!!! others_cols is empty !!!")                     # if all indexes of cols in converged_columns than all converged than break 
                break
            end

            take_cols = min(size(T, 1) - ind + 1, size(others_cols, 1))     # amount columns to take in block
            print(" take_cols = $take_cols ")

            _, TMP = qr(R[:, others_cols])
            F = qr(TMP, Val(true))                            # PQR
            S_cap = others_cols[ F.p[1:take_cols] ]
            S = [ S[M[1:ind-1]] ; S_cap]
            
            M_transposed = traspose_perm_matrix(M)
            Q_cap, R_cap = qr_gram_schmidt(P[:, M_transposed], A, A_scal) # QR decomp with A-scalar product

            P_cap = Q_cap[:, 1:take_cols]
            P_wave = Q_cap[:, (take_cols+1):size(Q_cap, 2)]

            nu = ((P_cap' * A) * P_cap) \ ((P_cap' * A) * P_wave);
            P_wave -= P_cap * nu;

            V = hcat(V, P_wave)                                          # column concatenation

            AV = A * V
            VAV = V' * AV

            beta = ((P_cap' * A) * P_cap) \ ((P_cap' * A) * R[:, S])
            gamma = (VAV) \ (AV' * R[:, S])
            P = R[:, S] - P_cap * beta - V * gamma

            beta = ((P_cap' * A) * P_cap) \ ((P_cap' * A) * P)
            gamma = (VAV) \ (AV' * P)
            P -= P_cap * beta + V * gamma

            
            println("end")
            #println("cnorm VAP: ",maximum(abs.(AV' * P)))

        end
        
        iteration += 1
    end
    if iteration == maxiter
        println("!! Reached max iteration !!")
    end
    if isempty(c_norms) == false
        println("max(c_norms of P_{k+2}' A P_k): ", maximum(c_norms))
    end
    X, B = swap_back(X, B, perm)
    # X = scale_columns(X, norms2_B)
    
    return X, iteration
end


f = FileIO.load("matrix.jld2");
A = f["A"];
B = f["B"];

println("n = ", size(A, 1))
println("right sides = ", size(B, 2))

println("function started")

X, iter = BCG(A, B)

println("iterations = $iter")
println("C_norm of B - AX is ", maximum(abs.(B - A*X)))
println("max(2norm(B - AX)) is ", maximum(norms2(B - A*X)))
