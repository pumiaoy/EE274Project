import cvxpy as cp
import numpy as np
import scipy as scipy

# Fix random number generator so we can repeat the experiment.
np.random.seed(0)

# Dimension of matrix.
n = 10

# Number of samples, y_i
N = 1000

# Create sparse, symmetric PSD matrix S
A = np.random.randn(n, n)  # Unit normal gaussian distribution.
A[scipy.sparse.rand(n, n, 0.85).todense().nonzero()] = 0  # Sparsen the matrix.
Strue = A.dot(A.T) + 0.05 * np.eye(n)  # Force strict pos. def.

# Create the covariance matrix associated with S.
R = np.linalg.inv(Strue)

# Create samples y_i from the distribution with covariance R.
y_sample = scipy.linalg.sqrtm(R).dot(np.random.randn(n, N))

# Calculate the sample covariance matrix.
Y = np.cov(y_sample)
# The alpha values for each attempt at generating a sparse inverse cov. matrix.
alphas = [10, 2, 1]

# Empty list of result matrixes S
Ss = []

# Solve the optimization problem for each value of alpha.
for alpha in alphas:
    # Create a variable that is constrained to the positive semidefinite cone.
    S = cp.Variable(shape=(n,n), PSD=True)

    # Form the logdet(S) - tr(SY) objective. Note the use of a set
    # comprehension to form a set of the diagonal elements of S*Y, and the
    # native sum function, which is compatible with cvxpy, to compute the trace.
    # TODO: If a cvxpy trace operator becomes available, use it!
    obj = cp.Maximize(cp.log_det(S) - sum([(S*Y)[i, i] for i in range(n)]))

    # Set constraint.
    constraints = [cp.sum(cp.abs(S)) <= alpha]

    # Form and solve optimization problem
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.CVXOPT)
    if prob.status != cp.OPTIMAL:
        raise Exception('CVXPY Error')

    # If the covariance matrix R is desired, here is how it to create it.
    R_hat = np.linalg.inv(S.value)

    # Threshold S element values to enforce exact zeros:
    S = S.value
    S[abs(S) <= 1e-4] = 0

    # Store this S in the list of results for later plotting.
    Ss += [S]

    print('Completed optimization parameterized by alpha = {}, obj value = {}'.format(alpha, obj.value))