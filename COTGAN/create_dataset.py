import numpy as np
import numpy.linalg as la
from scipy.sparse import spdiags
import torch

from tqdm import trange
class DatasetTwoCollidingSolitons():
    def __init__(self, eta, gamma, t_max, P, M, N, lower, upper):
        self.eta = eta
        self.gamma = gamma
        self.t_max = t_max
        self.P = P
        self.M = M
        self.N = N
        self.lower = lower
        self.upper = upper

        # format filename to not include decimals
        self.filename = f"eta={eta}_gamma={gamma}_tmax={t_max}_P={P}_N={N}_M={M}_lower={lower}_upper={upper}".replace('.', 'p') + ".npy"

        self.x, self.dx = self.grid(P, M) # spatial grid
        self.t, self.dt = self.grid(t_max, N) # temporal grid

        self.D1, self.D2 = self.difference_matrices(P,M)

        #self.g = lambda x, t: 0
        self.f = lambda u, t: -np.matmul(self.D1, .5*self.eta*u**2 + self.gamma**2*np.matmul(self.D2,u))# + self.g(x, t)
        self.Df = lambda u: -np.matmul(self.D1, self.eta*np.diag(u) + self.gamma**2*self.D2)

        self.data = None
        #self.data = self.load_data()

    def grid(self, P, M):
        dx = P/M
        x = np.linspace(0, P-dx, M)
        return x, dx

    def initial_condition_kdv(self, x, k1, k2, eta):
        M = x.size
        P = int((x[-1]-x[0])*M/(M-1))

        d1 = .3
        d2 = .5
        sech = lambda a: 1/np.cosh(a) # sech isn't defined in NumPy
        u0 = 0
        u0 += (-6./-eta)*2 * k1**2 * sech(np.abs(k1 * ((x+P/2-P*d1) % P - P/2)))**2
        u0 += (-6./-eta)*2 * k2**2 * sech(np.abs(k2 * ((x+P/2-P*d2) % P - P/2)))**2
        u0 = np.concatenate([u0[M:], u0[:M]], axis=-1)
        return u0

    def difference_matrices(self, P, M):
        dx = P/M
        e = np.ones(M) # unit vector of length M
        # 1st order central difference matrix:
        D1 = .5/dx*spdiags([e,-e,e,-e], np.array([-M+1,-1,1,M-1]), M, M).toarray()
        # 2nd order central difference matrix:
        D2 = 1/dx**2*spdiags([e,e,-2*e,e,e], np.array([-M+1,-1,0,1,M-1]), M, M).toarray()
        return D1, D2

    def midpoint_method(self, u, un, t, f, Df , dt ,M, tol, max_iter):
        '''
        Integrating one step of the ODE u_t = f, from u to un,
        with the implicit midpoint method
        Using Newton's method to find un
        '''
        I = np.eye(M)
        F = lambda u_hat: 1/dt*(u_hat-u) - f((u+u_hat)/2, t+.5*dt)
        J = lambda u_hat: 1/dt*I - 1/2*Df((u+u_hat)/2)
        err = la.norm(F(un))
        it = 0
        while err > tol:
            un = un - la.solve(J(un),F(un))
            err = la.norm(F(un))
            it += 1
            if it > max_iter:
                print("Newton's method didn't converge after {} iterations".format(max_iter))
                break
        return un

    def create_single_sample(self, k1, k2):
        u0 = self.initial_condition_kdv(self.x, k1, k2, self.eta)
        u = np.zeros((self.N, self.M))
        u[0] = u0
        for n in range(self.N-1):
            u[n+1] = self.midpoint_method(u[n], u[n], self.t[n], self.f, self.Df, self.dt, self.M, 1e-8, 100)
        return u

    def create_dataset(self, N_samples):
        self.data = np.zeros((N_samples, self.N, self.M))
        k1 = self.lower + (self.upper-self.lower) * np.random.rand(N_samples)
        k2 = self.lower + (self.upper-self.lower) * np.random.rand(N_samples)
        for i in trange(N_samples, desc="Creating dataset"):
            self.data[i] = self.create_single_sample(k1[i], k2[i])

    def save_data(self):
        try:
            np.save(self.filename, self.data)
            print(f"Saved data to {self.filename}")
        except:
            raise FileExistsError(f"Couldn't save data to {self.filename}")

    def load_data(self):
        try:
            data = np.load(self.filename)
            print(f"Successfully loaded data from {self.filename}")
            self.data = data
        except:
            raise FileNotFoundError(f"Couldn't load data from {self.filename}")

if __name__ == "__main__":
    # Parameters
    eta = 6.0
    gamma = 1.0

    t_max = 10
    c = 2  # the speed and two times the height of the wave
    P = 50  # period
    M = 360  # number of spatial points,
    N = 360  # = 360 # number of temporal points
    lower, upper = .2, .7  # lower and upper bounds for wave init heights

    dataset = DatasetTwoCollidingSolitons(eta, gamma, t_max, P, M, N, lower, upper)
    dataset.create_dataset(5)
    dataset.save_data()