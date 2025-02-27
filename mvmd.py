import numpy as np

def mvmd(signal, num_modes, alpha, tolerance=1e-3, init=0, tau=1e-2, verbose=False):
    """
    Multivariate Variational Mode Decomposition (MVMD)

    The function MVMD applies the "Multivariate Variational Mode Decomposition (MVMD)" algorithm to multivariate or multichannel data sets from [1].

    Parameters:
    -----------
    signal : ndarray
        Input multivariate signal to be decomposed (channels x samples)
    num_modes : int
        Number of modes to be recovered.
    alpha : float
        Bandwidth constraint parameter.
    tolerance : float
        Stopping criterion for the dual ascent
    init : int
        Initialization method for center frequencies:
            - 0: All omegas start at 0.
            - 1: All omegas are initialized lineary distributed.
            - 2: All omegas are initialized exponentially distributed.
    tau : float
        Time-step of the dual ascent (use 0 for noise-slack).
    verbose : bool
        Print information about the convergence of the algorithm.

    Returns:
    --------
    modes : ndarray
        The collection of decomposed modes (K x C x T).
    modes_hat : ndarray
        Spectra of the modes (K x C x F).
    omega : ndarray
        Estimated mode center-frequencies (iter x K).

    
    This is a Python implementation of the algorithm described in:
    -----------------------------------------------------------------
    [1] N. Rehman and H. Aftab (2019) Multivariate Variational Mode Decomposition, IEEE Transactions on Signal Processing
    """
    # Variables
    max_iter = 500 # Maximum number of iterations

    # Extract dimensions -> assumes that the signal is shaped channels x time-points
    if signal is None or not isinstance(signal, np.ndarray) or signal.ndim != 2:
        raise ValueError("Signal must be a non-empty 2D numpy array (channels x t-points)" )

    num_channels, num_tpoints = signal.shape

    # Show dimensionality of the problem
    if verbose:
        print(
            f'___Parameters___ \n'
            f'Signal - Channels: {num_channels} Timepoints: {num_tpoints} \n'
            f' Model - Number of modes: {num_modes} ')

    # Initialize omegas
    omega_list = np.zeros((max_iter + 1, num_modes))

    if init == 1:  # Linear
        omega_list[0, :] = np.linspace(0, 0.5, num_modes)
    elif init == 2:  # Exponential
        omega_list[0, :] = 0.5 * np.logspace(-3, 0, num_modes)
    else:  # constant
        omega_list[0, :] = 0

    #--- Frequency domain ---
    sampling_freq = 1.0 / num_tpoints  # Sampling frequency
    num_fpoints = num_tpoints + 1
    f_points = np.linspace(0,0.5, num_fpoints)

    signal_hat = _to_freq_domain(signal)

    modes_hat = np.zeros((num_modes, num_channels, num_fpoints), dtype=complex)


    #=== MVDM algorithm ===
    # Start with empty dual variables
    lambda_hat = np.zeros((max_iter + 1, num_channels, num_fpoints), dtype=complex)
    residual_diff = tolerance + np.finfo(float).eps # Stopping criterion
    n = 0  # Loop counter

    while n < max_iter and residual_diff > tolerance:
        residual_diff = 0

        # Loop over the modes
        for k in range(num_modes):
            #--- Mode update ---
            aux_mode_hat = np.copy(modes_hat[k, :, :])
            modes_hat[k, :, :] = 0 # Remove cotribution from the previous iteration

            # Update mode
            modes_hat[k, :, :] = signal_hat - np.sum(modes_hat, axis=0) - 0.5*lambda_hat[n, :, :]
            modes_hat[k, :, :] /= 1 + alpha * (f_points - omega_list[n, k]) ** 2

            # Update residual
            residual_diff += np.sum(np.abs(modes_hat[k, :, :] - aux_mode_hat) ** 2)

            #--- Update central frequencies ---
            module_mode_hat = np.abs(modes_hat[k, :, :]) ** 2

            omega_list[n + 1, k] = np.sum(np.dot(module_mode_hat, f_points))
            omega_list[n + 1, k] /= np.sum(module_mode_hat)

            # Dual ascent
            lambda_hat[n+1, :, :] = (
                lambda_hat[n, :, :] + tau * (np.sum(modes_hat, axis=0) - signal_hat)
            )
         
        # Loop counter update
        n += 1

        # Convergence modulation
        residual_diff /= num_tpoints
        

    # Post-processing
    omega = omega_list[:n, :] / sampling_freq

    # Order the frequency list based on teh last results
    idx = np.argsort(omega[-1, :])
    omega = omega[:, idx]

    # Signal reconstruction
    modes_list = np.array([_to_time_domain(m_hat) for m_hat in modes_hat])

    # Order modes
    modes_list = modes_list[idx, :, :]
            
    return modes_list, modes_hat, omega


def _to_freq_domain(signal, pad_mode='symmetric'):
    # Dimension
    tpoints = signal.shape[1]

    if pad_mode is None:
        full_signal = signal
    else:
        full_signal = np.pad(signal, pad_width=((0, 0), (tpoints // 2, tpoints - tpoints // 2)), mode=pad_mode)

    signal_hat = np.fft.fft(full_signal, axis=1)[:, :tpoints + 1]

    return signal_hat


def _to_time_domain(signal_hat, extended=False):
    channels, fpoints = signal_hat.shape
    red_ft = fpoints - 1

    # Construct Hermitian-symmetric assuming the signal is real
    full_signal_hat = np.zeros((channels, 2 * red_ft), dtype=complex)

    full_signal_hat[:, red_ft:] = signal_hat[:, :red_ft]
    full_signal_hat[:, :red_ft] = np.conj(signal_hat[:, red_ft:0:-1])

    # Inverse FFT to reconstruct the time-domain signal
    shifted_full_signal_hat = np.fft.ifftshift(full_signal_hat, axes=1)

    signal = np.real(np.fft.ifft(shifted_full_signal_hat, axis=1))

    if not extended:
        signal = signal[:, (red_ft // 2):(3 * red_ft // 2)]

    return signal