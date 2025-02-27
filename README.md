# Multivariate Variational Mode Decomposition (MVMD)

This code provides an implementation of the Multivariate Variational Mode Decomposition (MVMD) algorithm (1). MVMD is used to decompose multivariate or multichannel data sets, enabling the analysis of complex signals by separating them into modes.

## Features

- Decompose a multivariate signal into their intrinsic oscillatory modes
- Option to initialize center frequencies in different ways (constant, linear, or exponential)
- Verbose mode to print convergence information

## Requirements
- NumPy

## Installation

Clone the repository:

```bash
git clone https://github.com/Dmocrito/mvmd.git
```

## Usage

Import the `mvmd` function from `mvmd.py` and apply it to your multivariate signal:

```python
from mvmd import mvmd

# Example signal
signal = np.random.rand(3, 100)  # 3 channels, 100 samples

# Decompose the signal
modes, modes_hat, omega = mvmd(signal, num_modes=3, alpha=2000, tolerance=1e-3)
```

## Parameters

- `signal` : ndarray
  - Input multivariate signal to be decomposed (channels x samples).
- `num_modes` : int
  - Number of modes to be recovered.
- `alpha` : float
  - Bandwidth constraint parameter.
- `tolerance` : float
  - Stopping criterion for the dual ascent.

**Optional parameters**
- `init` : int
  - Initialization method for center frequencies:
    - 0: All initial frequency guesses start at 0.
    - 1: All initial frequency guesses are set uniformly.
    - 2: All initial frequency guesses are set exponentially.
- `tau` : float
  - Time-step of the dual ascent (use 0 for noise-slack).
- `verbose` : bool
  - If True, print information about the convergence of the algorithm.

## Returns

- `modes` : ndarray
  - The collection of decomposed modes (K x C x T).
- `modes_hat` : ndarray
  - Spectra of the modes (K x C x F).
- `omega` : ndarray
  - Estimated mode center-frequencies (iter x K).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## References

(1) N. Rehman and H. Aftab (2019) "Multivariate Variational Mode Decomposition," IEEE Transactions on Signal Processing.