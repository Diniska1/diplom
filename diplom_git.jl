using LinearAlgebra

eps = 1e-5

function A_scal(A, a, b)
    return (A * a)' * b
end

function print_matrix(A, msg="")
    println("Matrix ", msg)
    display(A)
end

function qr_gram_schmidt(A::Matrix{Float64}, B, inner_product::Function)
    m, n = size(A)
    Q = zeros(Float64, m, n)
    R = zeros(Float64, n, n)

    for j = 1:n
        v = A[:, j]  

        for i = 1:j-1
            R[i, j] = inner_product(B, Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        end

        R[j, j] = sqrt(inner_product(B, v, v))
        Q[:, j] = v / R[j, j]
    end

    return Q, R
end

function check(T)               # find index small diag elem
    n = size(T, 1)
    ind = n + 1
    for i in 1:n
        if abs(T[i, i]) < eps
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

function traspose_perm_matrix(a)
    n = size(a, 1)
    b = zeros(Int, n)
    for i in 1:n
        b[a[i]] = i
    end
    return b
end

function BCG(A, B)
    n = size(A, 1)
    n1, m = size(B) # amount right sides
    if n1 != n
        println("DimensionMismatch, rows A != rows B")
        exit()
    end
    r = 3          # block size

    X = zeros(n, m)     # starting value
    R = B - A * X               # n x m

    F = qr(R, Val(true)) 
    S = copy(F.p[1:r])
    P = copy(R[:, S])                   # n x r
    P_prev = copy(P)

    used_columns = Int32[]
    c_norms = Float64[]

    V = Matrix{Float64}(undef, n, 0)

    iteration = 1
    maxiter = 1000
    while iteration < maxiter

        if iteration > 2 && iteration % 2 == 0
            push!(c_norms, check_orth(A, P, P_prev))
            P_prev = P
        end

        # if iteration % 100 == 0 && iteration > 2
        #     println("iter = $iteration, max c_norms is ",maximum(c_norms))
        # end    

        
        if isapprox(det(P' * A * P), 0.0)
            print_matrix(P' * A * P, "P' * A * P")
            println("P'AP SINGULAR")
            exit()
        end
        alpha = (P' * A * P) \ P' * R       # r x m
        X = X + P * alpha                   # n x m
        R = R - A * P * alpha               # n x m

        F = qr(R[:,S], Val(true))                   # PQR
        M = F.p
        T = F.R

        ind = check(T)                              # find index small elem on diag(T)
        if ind == size(T, 1) + 1                    # if all elements on diag(T) is large
            beta = (P' * A * P) \ P' * A * R[:,S]
            P = R[:, S] - P * beta

            # if isempty(V) == false
            #     gamma = V' * A * R[:, S]
            #     P = P - V * gamma
            # end

        else

            add_used_cols = [ S[F.p[i]] for i in ind:size(F.p, 1) ]
            used_columns = [ used_columns ; add_used_cols ]                      # indexes of columns that converged
            others_cols = setdiff( 1:m, [ used_columns ; S[ F.p[1:ind-1] ] ] )   # indexes of cols that in R and not in used_columns and not in current block
            
            if isempty(others_cols) == true && isempty(F.p[1:ind-1]) == true                                # stopping criteria
                println("!!! others_cols is empty !!!")                     # if all indexes of cols in used_columns than all converged than break 
                break
            end

            take_cols = min(size(T, 1) - ind + 1, size(others_cols, 1))     # amount columns to take in block

            F = qr(R[:, others_cols], Val(true))                            # PQR
            S_cap = others_cols[ F.p[1:take_cols] ]
            S = [ S[M[1:ind-1]] ; S_cap]
            
            M_transposed = traspose_perm_matrix(M)
            Q_cap, R_cap = qr_gram_schmidt(P[:, M_transposed], A, A_scal) # QR decomp with A-scalar product

            P_cap = Q_cap[:, 1:take_cols]
            P_wave = Q_cap[:, (take_cols+1):size(Q_cap, 2)]

            V = hcat(V, P_wave)                                          # column concatenation

            beta = P_cap' * A * R[:, S]
            gamma = V' * A * R[:, S]
            P = R[:, S] - P_cap * beta - V * gamma
        end
        
        iteration += 1
    end
    if iteration == maxiter
        println("!! Reached max iteration !!")
    end
    if isempty(c_norms) == false
        println("max(c_norms of P_{k+2}' A P_k): ", maximum(c_norms))
    end
    return X, iteration
end

n = 200
A = float(zeros(n,n))
for i in 1:n
    if i != 1
        A[i - 1, i] = i - 1
        A[i, i - 1] = i - 1
    end
    if i > 3
        A[i - 3, i] = 1
        A[i, i - 3] = 1
    end
    A[i, i] = 2 * i + 1
end
if isposdef(A) == false
    println("Matrix A is NOT positive definitive or not symmertic\n")
    exit()
end
B = float(hcat(collect(1:n), circshift(collect(1:n), 1),  circshift(collect(1:n), 2), circshift(collect(1:n), 3), circshift(collect(1:n), 4)))


X, iter = BCG(A, B)

println("iterations = $iter")
println("C_norm of B - AX is ", maximum(abs.(B - A*X)))
