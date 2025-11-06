from src.methods import fit_mle, fit_parametricEB, fit_semiBayes
from src.dgps import DatasetGenerator
import tqdm


simple_generator = DatasetGenerator(n=20, tau_0=0.2, tau_1=0.2, sigma2=1.0, rho=0.5)

n_sims = 500
with tqdm.tqdm(total=n_sims) as pbar:
    for _ in range(n_sims):
        X, y, beta = simple_generator(N=100)
        model = fit_mle(X, y)
        eb = fit_parametricEB(model)
        sb = fit_semiBayes(model)
        pbar.update(1)
