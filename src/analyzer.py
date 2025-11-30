import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlate2d
from scipy.fft import fft2, fftshift, fft, rfft

class Analyzer:
    def __init__(self):
        pass

    # --- Metric 1: Mexican Hat (Autocorrelation) ---
    def compute_autocorrelation(self, data, mode='2d'):
        """
        Detects Short-Range (+) and Long-Range (-) feedback.
        """
        if mode == '2d':
            # For Gray-Scott and DDPM Images
            # Normalize
            data = data - np.mean(data)
            corr = correlate2d(data, data, boundary='symm', mode='same')
            # Normalize to 0-1
            corr = corr / corr.max()
            
            # Take a 1D slice through the center to visualize the "Hat"
            mid_y = corr.shape[0] // 2
            slice_1d = corr[mid_y, :]
            return slice_1d
            
        elif mode == '1d':
            # For LSTM Time Series
            data = data.flatten()
            data = data - np.mean(data)
            corr = correlate(data, data, mode='same')
            corr = corr / corr.max()
            return corr

    # --- Metric 2: Spectral Analysis ---
    def compute_psd(self, data, mode='2d'):
        """
        Finds the characteristic frequency (Pattern Size).
        """
        if mode == '2d':
            f = fft2(data)
            fshift = fftshift(f)
            magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
            
            # Radial Average (to get a 1D plot of Frequency Power)
            center = np.array(data.shape) // 2
            y, x = np.indices(data.shape)
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            r = r.astype(int)
            
            tbin = np.bincount(r.ravel(), magnitude.ravel())
            nr = np.bincount(r.ravel())
            radial_profile = tbin / (nr + 1e-8)
            return radial_profile
            
        elif mode == '1d':
            # Standard FFT for LSTM
            f = rfft(data.flatten())
            magnitude = np.abs(f)
            return magnitude

    # --- Metric 3: Stability/Flux ---
    def compute_flux(self, history_array):
        """
        history_array: [Time, Dimensions...]
        """
        flux = []
        for i in range(1, len(history_array)):
            # L2 Norm of the difference
            diff = np.linalg.norm(history_array[i] - history_array[i-1])
            flux.append(diff)
        return flux

    # --- MASTER PLOTTER ---
    def compare_models(self, 
                       gs_final, gs_history, 
                       lstm_final, lstm_history, 
                       ddpm_final, ddpm_history):
        
        fig, ax = plt.subplots(3, 3, figsize=(18, 12))
        plt.suptitle("Comparative Analysis: Reaction-Diffusion vs LSTM vs DDPM", fontsize=16)
        
        # Row 1: The Resulting Patterns
        ax[0,0].imshow(gs_final, cmap='jet')
        ax[0,0].set_title("Gray-Scott (Ground Truth)")
        
        # For LSTM, we plot the sequence
        ax[0,1].plot(lstm_final) 
        ax[0,1].set_title("LSTM Sequence")
        
        ax[0,2].imshow(ddpm_final, cmap='gray')
        ax[0,2].set_title("DDPM Generated")

        # Row 2: The "Mexican Hat" (Autocorrelation)
        # This proves the Activator/Inhibitor mechanism
        ac_gs = self.compute_autocorrelation(gs_final, '2d')
        ac_lstm = self.compute_autocorrelation(lstm_final, '1d')
        ac_ddpm = self.compute_autocorrelation(ddpm_final, '2d')

        ax[1,0].plot(ac_gs, color='green')
        ax[1,0].set_title("GS: Spatial Correlation\n(Look for dip below 0)")
        ax[1,0].axhline(0, color='black', linestyle='--')
        
        # Zoom in on center for LSTM
        center = len(ac_lstm)//2
        ax[1,1].plot(ac_lstm[center-50:center+50], color='blue')
        ax[1,1].set_title("LSTM: Temporal Correlation")
        ax[1,1].axhline(0, color='black', linestyle='--')

        ax[1,2].plot(ac_ddpm, color='red')
        ax[1,2].set_title("DDPM: Spatial Correlation")
        ax[1,2].axhline(0, color='black', linestyle='--')

        # Row 3: Flux Dynamics (History of creation)
        flux_gs = self.compute_flux(gs_history)
        flux_lstm = self.compute_flux(lstm_history)
        flux_ddpm = self.compute_flux(ddpm_history)

        ax[2,0].plot(flux_gs, color='green')
        ax[2,0].set_title("GS Flux (Settling to Steady State)")
        
        ax[2,1].plot(flux_lstm, color='blue')
        ax[2,1].set_title("LSTM Flux (Limit Cycle/Stable)")
        
        ax[2,2].plot(flux_ddpm, color='red')
        ax[2,2].set_title("DDPM Flux (Structure Formation Spike)")
        # Reverse DDPM x-axis because t goes N -> 0
        ax[2,2].invert_xaxis() 

        plt.tight_layout()
        plt.show()

class RDComparator:
    def __init__(self):
        pass

    # --- HELPER: Autocorrelation ---
    def compute_acf(self, signal, mode='2d'):
        """Computes Autocorrelation.
        Signal: 1D (LSTM gates) or 2D (Images)
        """
        signal = np.array(signal)
        signal = signal - np.mean(signal) # Zero center
        
        if mode == '2d':
            # For Gray Scott and DDPM images
            corr = correlate2d(signal, signal, boundary='symm', mode='same')
        else:
            # For LSTM time-series
            corr = correlate(signal, signal, mode='same')
            
        # Normalize
        return corr / np.max(corr)

    # --- HELPER: Power Spectral Density ---
    def compute_psd(self, signal, mode='2d'):
        if mode == '2d':
            f = fft2(signal)
            fshift = fftshift(f)
            magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
            # Radial average could be done here, but 2D spectrum is fine for viz
            return magnitude
        else:
            f = fft(signal)
            fshift = fftshift(f)
            return np.abs(fshift)

    # --- 1. EXPOSE REACTION & DIFFUSION ---
    def expose_mechanics(self, gs_data, lstm_gates, ddpm_history):
        """
        Visualizes the raw 'Activator' and 'Inhibitor' signals side-by-side.
        """
        fig, ax = plt.subplots(2, 3, figsize=(18, 10))
        plt.suptitle("Exposing Reaction (Activator) vs Diffusion (Inhibitor)", fontsize=16)

        # --- GRAY SCOTT (Ground Truth) ---
        # Activator: U component (Concentration)
        # Inhibitor: V component (Concentration)
        ax[0, 0].imshow(gs_data['U'], cmap='inferno')
        ax[0, 0].set_title("GS: Activator (U)")
        ax[1, 0].imshow(gs_data['V'], cmap='viridis')
        ax[1, 0].set_title("GS: Inhibitor (V)")

        # --- LSTM (Temporal) ---
        # Activator: Input Gate (i) - The impulse
        # Inhibitor: Forget Gate (f) - The decay
        # We take a slice of the hidden units (e.g., unit 0)
        i_gate = lstm_gates['activator'][:, 0].flatten() # Time series
        f_gate = lstm_gates['inhibitor'][:, 0].flatten()
        
        ax[0, 1].plot(i_gate, 'r-')
        ax[0, 1].set_title("LSTM: Input Gate (Reaction)")
        ax[1, 1].plot(f_gate, 'b-')
        ax[1, 1].set_title("LSTM: Forget Gate (Diffusion)")

        # --- DDPM (Generative) ---
        # Activator: The magnitude of the update step (The U-Net's "Push")
        # Inhibitor: The Noise level (Variance) at that step
        # We calculate the per-step flux for the activator
        
        ddpm_hist = np.array(ddpm_history).squeeze() # [Steps, H, W]
        # Flux = |x_t - x_{t+1}|
        flux = [np.mean(np.abs(ddpm_hist[i] - ddpm_hist[i+1])) for i in range(len(ddpm_hist)-1)]
        # Noise schedule (Idealized linear decay for viz)
        noise_level = np.linspace(1, 0, len(flux))
        
        ax[0, 2].plot(flux, 'g-')
        ax[0, 2].set_title("DDPM: Structuring Flux (Reaction)")
        ax[1, 2].plot(noise_level, 'k--')
        ax[1, 2].set_title("DDPM: Noise Schedule (Diffusion)")
        
        plt.tight_layout()
        plt.show()

    # --- 2. COMPARE PATTERN FORMATION (METRICS) ---
    def compare_patterns(self, gs_img, lstm_seq, ddpm_img):
        """
        Compares the spatial/temporal signatures.
        """
        fig, ax = plt.subplots(2, 3, figsize=(18, 10))
        plt.suptitle("Pattern Formation Metrics: PSD & Autocorrelation", fontsize=16)

        # 1. Gray Scott
        ax[0,0].imshow(self.compute_psd(gs_img, '2d'), cmap='jet')
        ax[0,0].set_title("GS: Power Spectrum (Frequency)")
        ax[1,0].imshow(self.compute_acf(gs_img, '2d'), cmap='coolwarm')
        ax[1,0].set_title("GS: Autocorrelation (Range)")

        # 2. LSTM (Converted to Spectrogram for visual similarity)
        # We treat the sequence as a 1D signal
        ax[0,1].plot(self.compute_psd(lstm_seq, '1d'))
        ax[0,1].set_title("LSTM: Power Spectrum (Frequency)")
        ax[1,1].plot(self.compute_acf(lstm_seq, '1d'))
        ax[1,1].set_title("LSTM: Autocorrelation (Memory Length)")

        # 3. DDPM
        ax[0,2].imshow(self.compute_psd(ddpm_img, '2d'), cmap='jet')
        ax[0,2].set_title("DDPM: Power Spectrum")
        ax[1,2].imshow(self.compute_acf(ddpm_img, '2d'), cmap='coolwarm')
        ax[1,2].set_title("DDPM: Autocorrelation")

        plt.show()