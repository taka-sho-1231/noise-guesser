import numpy as np

class Diffuser:
    def __init__(self, num_timesteps: int = 1000, 
                 beta_start: float = 0.0001, 
                 beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        self.betas = np.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)
    
    def add_noise(self, x_0, t, return_x_T=False):
        T = self.num_timesteps
        assert 1 <= t <= T

        t_idx = t - 1  # alpha_bars[0] is for t=1
        alpha_bar = self.alpha_bars[t_idx]  # (N,)
        alpha_bar = alpha_bar.reshape(1,1,1)  # (N, 1, 1, 1)
        
        noise = np.random.randn(*x_0.shape).astype(np.float32)
        x_t = np.sqrt(alpha_bar) * x_0 + np.sqrt(1 - alpha_bar) * noise
        
        if not return_x_T:
            return x_t, noise
        else:
            alpha_bar_T = self.alpha_bars[-1]
            alpha_bar_T = alpha_bar_T.reshape(1, 1, 1)
            x_T = np.sqrt(alpha_bar_T) * x_0 + np.sqrt(1 - alpha_bar_T) * noise
            return x_t, x_T, noise
        
diffuser = Diffuser()