import numpy as np

def mvmd(signal, K, alpha, tol=1e-3, init=0, tau=1e-2, DC=False):
    """
    Multivariate Variational Mode Decomposition (MVMD)
    
    The function MVMD applies the "Multivariate Variational Mode Decomposition (MVMD)" algorithm to multivariate or multichannel data sets from [1].
    
    Parameters:
    -----------
    signal : ndarray
        Input multivariate signal to be decomposed (channels x samples).
            It assumes that: channels < samples 
    K : int
        Number of modes to be recovered.
    alpha : float
        Bandwidth constraint parameter.
    tol : float
        Stopping criterion for the dual ascent
    init : int
        Initialization method for center frequencies:
            - 0: All omegas start at 0.
            - 1: All omegas are initialized uniformly.
            - 2: All omegas are initialized randomly.
    tau : float
        Time-step of the dual ascent (use 0 for noise-slack).
    DC : bool
        If True, the first mode is kept at DC (0 frequency).
    
    Returns:
    --------
    u : ndarray
        The collection of decomposed modes (K x C x T).
    u_hat : ndarray
        Spectra of the modes (K x C x N).
    omega : ndarray
        Estimated mode center-frequencies (N_iter x K).

    
    This is a Python implementation of the algorithm described in:
    -----------------------------------------------------------------
    [1] N. Rehman and H. Aftab (2019) Multivariate Variational Mode Decomposition, IEEE Transactions on Signal Processing
    """
    # Variables
    N = 500 # Maximum number of iterations

    # Check diminsions (transpose if nescessary)
    if signal.shape[0] < signal.shape[1]: # It assumes that: channels < samples
        C, T = signal.shape
    else:
        T, C = signal.shape
        signal = signal.T

    #--- Preprocessing steps ---
    fs = 1.0 / T  # Sampling frequency

    # Mirroring - To avoid edge interferring effects
    #     > mMm 14-Oct-2016 - I optimize this part into a single line by using 
    #     the np.pad function instead of using indices.
    f = np.pad(signal, ((0, 0), (T//2, T - T//2)), mode='symmetric')

    # Time domain and frequencies
    T = f.shape[1]
    t_points = np.arange(1, T + 1) / T
    f_points = t_points - 0.5 - 1.0 / T

    # Construct and center f_hat
    f_hat = np.fft.fftshift(np.fft.fft(f, axis=1), axes=1)
    f_hat_plus = f_hat.copy()
    f_hat_plus[:, :T // 2] = 0

    u_hat_new = np.zeros((C, T), dtype=complex) 
    u_hat_plus = np.zeros((K, C, T), dtype=complex)
    omega_plus = np.zeros((N, K))

    # Initialize omegas
    if init == 1:
        omega_plus[0, :] = (0.5 / K) * np.arange(K)
    elif init == 2:
        omega_plus[0, :] = np.sort(
            np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(K))
        )
    else:
        omega_plus[0, :] = 0

    if DC:
        omega_plus[0, 0] = 0

    # Start with empty dual variables
    lambda_hat = np.zeros((N, C, T), dtype=complex)
    uDiff = tol + np.finfo(float).eps # Stopping criterion
    n = 0  # Loop counter
    sum_uk = np.zeros((C, T), dtype=complex)  # Accumulator

    # Main loop of MVMD
    while uDiff > tol and n < N - 1:
        sum_uDiff = 0.

        # Loop over the modes
        for k in range(K):
            # Update mode accumulator
            sum_uk = np.sum(np.delete(u_hat_plus, k, axis=0), axis=0)

            # Update spectrum of mode through Wiener filter of residuals
            numerator = f_hat_plus - sum_uk - 0.5*lambda_hat[n, :, :]
            denominator = 1 + alpha * (f_points - omega_plus[n, k])**2

            # Update new mode
            u_hat_new = numerator / denominator

            # Update modes
            sum_uDiff += np.sum(np.abs(u_hat_new - u_hat_plus[k, :, :]) ** 2)
            u_hat_plus[k, :, :] = u_hat_new

            # Update center frequencies
            if not DC or k > 0:
                module_u_hat = np.abs(u_hat_plus[k, :, T // 2:]) ** 2

                numerator = np.sum(np.dot(module_u_hat, f_points[T // 2:]))
                denominator = np.sum(module_u_hat)

                omega_plus[n + 1, k] = numerator / denominator

        # Dual ascent
        lambda_hat[n+1, :, :] = (
            lambda_hat[n, :, :] + tau * (np.sum(u_hat_plus, axis=0) - f_hat_plus)
        )
         
        # Loop counter update
        n += 1

        # Convergence check
        uDiff = (1.0 / T) * sum_uDiff
        

    # Post-processing
    omega = omega_plus[:n, :] / fs

    # Order the results of omoga list, based on the final result
    idx = np.argsort(omega[-1, :])
    omega = omega[:, idx]

    # Signal reconstruction
    u_hat_full = np.zeros((K, C, T), dtype=complex)
    T2 = T // 2

    u_hat_full[:, :, T2:] = u_hat_plus[:, :, T2:]
    u_hat_full[:, :, T2-1:0:-1] = np.conj(u_hat_plus[:, :, T2+1:])
    u_hat_full[:, :, 0] = np.conj(u_hat_plus[:, :, -1])

    # Compute the modes
    u = np.zeros((K, C, u_hat_full.shape[2]))
    for k in range(K):
        for c in range(C):
            u_temp = np.fft.ifft(np.fft.ifftshift(u_hat_full[k, c, :]))
            u[k, c, :] = np.real(u_temp)

    # Order modes
    u = u[idx, :, :]

    # Remove mirror part
    T4 = T // 4
    u = u[:, :, T4:3 * T4]

    # Recompute spectra
    u_hat = np.zeros((K, C, u.shape[2]), dtype=complex)
    for k in range(K):
        for c in range(C):
            u_hat[k, c, :] = np.fft.fftshift(np.fft.fft(u[k, c, :]))
            
    return u, u_hat, omega

