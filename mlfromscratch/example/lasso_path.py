import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path
from sklearn import datasets

X, y = datasets.load_diabetes(return_X_y=True)

X /= X.std(axis=0)
eps = 5e-3

print("Computing regularization path using the lasso...")
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=False) 
neg_log_alphas_lasso = -np.log10(alphas_lasso)

#print alphas_lasso
#print coefs_lasso

plt.figure(1)
for coef in coefs_lasso:
    plt.plot(neg_log_alphas_lasso, coef)
plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso Paths')
plt.axis('tight')
plt.show()

