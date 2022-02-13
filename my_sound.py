import os
import pdb
import numpy as np
import yaml


# #=========================================================
# # Input Arguments
# #=========================================================
# with open("parameters.yml", 'r') as stream:
#     D = yaml.safe_load(stream)
#
# for key in D:
#     globals() [str(key)] = D[key]
#     print('{}: {}'.format(str(key), D[key]))
#     # transforms key-names from dictionary into global variables, then assigns those variables their respective key-values
# ========================================================
# CLASS - Pressure Wave
# ========================================================
class PressureWave:
    def __init__(self, amp, freq, t0):
        self.amplitude = amp
        self.frequency = freq
        self.time_init = t0
    # ----------------------------------------------------------
    def get_pressure(self, t):
        duration = t - self.time_init
        assert duration >= 0
        return self.amplitude * np.sin(2 * np.pi * self.frequency * (-duration))
    # ----------------------------------------------------------
    def get_pressure_dot(self, t):
        duration = t - self.time_init
        assert duration >= 0
        return - self.amplitude * 2 * np.pi * self.frequency * np.cos(2 * np.pi * self.frequency * (-duration))
    # ---------------------------------------------------------
    # default values for 'speed' and 'alpha' store constants for the simulation
    def get_attenuation(self, t, speed=360, alpha=0.0022):
        duration = t - self.time_init
        assert duration >= 0
        return np.exp(- alpha * self.frequency ** 2 * speed * duration)


# ========================================================
# CLASS - Sound Wave (multiple pressure waves)
# ========================================================
class SoundWave:
    def __init__(self, amp_range, freq_range, num_waves):
        assert type(num_waves) is int
        assert num_waves >= int(1)
        self.n_waves = num_waves
        self.amp_range = amp_range
        self.freq_range = freq_range
        self.waves = self.generate_waves()
        print("__init__(SoundWave)")
        print(f"amp_range = {amp_range}")

    # ----------------------------------------------------------------
    def generate_waves(self):
        amp_min, amp_max = self.amp_range
        assert amp_max >= amp_min
        freq_min, freq_max = self.freq_range
        assert freq_max >= freq_min
        # -----------------------------------------
        amp_samples = self.log_normal_normalize(amp_min, amp_max)
        print("generate_waves():")
        print(f"amp_samples = {amp_samples}")
        freq_samples = self.log_normal_interval(freq_min, freq_max)
        t_samples = - np.random.uniform(0, 1, self.n_waves) / freq_samples  # random portion of T = 1/freq
        # -----------------------------------------
        sound = list()
        for j in range(self.n_waves):
            sound.append(PressureWave(amp_samples[j], freq_samples[j], t_samples[j]))
        return sound

    # ----------------------------------------------------------------
    def pressure(self, t):
        assert t >= 0
        sum_pos = sum_neg = 0.0
        sum_abs = 0.0
        # --------------------------------
        for wave in self.waves:
            #print(f"wave.amplitude = {wave.amplitude}")
            temp = wave.get_pressure(t)
            sum_abs = sum_abs + abs(temp)
            if temp >= 0.0:
                sum_pos += temp  # sums all positive pressure to avoid catastrophic cancellation
            else:
                sum_neg += temp  # sums all negative pressures to avoid catastrophic cancellation
        print("pressure(t):")
        print(f"mean(abs(P)) = {sum_abs/self.n_waves}")
        return sum_neg + sum_pos
    # --------------------------------------------------------------------
    def pressure_dot(self, t):
        assert t >= 0
        sum_pos = sum_neg = 0.0
        # --------------------------------
        for wave in self.waves:
            Pdot = wave.get_pressure_dot(t)
            if Pdot >= 0:
                sum_pos += Pdot  # sums all positive pressure to avoid catastrophic cancellation
            else:
                sum_neg += Pdot  # sums all negative pressures to avoid catastrophic cancellation

        return (sum_pos + sum_neg)

    # --------------------------------------------------------------------
    def log_normal_normalize(self, A, B):
        # RETURN: random sample following a log_normal distribution that sums to x in (A,B)
        # A: minimum sum of samples
        # B: maximum sum of samples
        # ------------------------------------------
        assert B >= A
        mu, sigma = 0.0, 1.0
        log_rand = np.random.lognormal(mu, sigma, self.n_waves)
        log_rand = log_rand / np.sum(log_rand)
        return log_rand * np.random.uniform(A,B)  # normalized and then scaled to desired random sum
    # --------------------------------------------------------------------
    def log_normal_interval(self, A, B):
        # RETURN: random sample following a log_normal distribution mapped to the interval {A < x < B}
        # A: minimum of any sample
        # B: maximum of any sample
        # note: 3 is the statistically apparent max of log_normal(0,1)
        # ------------------------------------------
        assert B >= A
        mu, sigma = 0.0, 1.0
        log_rand = np.random.lognormal(mu, sigma, self.n_waves)
        max_rand = np.max([3, np.max(log_rand)])  # it is possible to generate a random value above 3 (default)
        return np.ones(self.n_waves) * A + log_rand * (B - A) / max_rand
        # ------------------------------------------
        # derivation of interval scaling:
        # (1) { 0 < r < 3 } --> { 0 < r/3 < 1 }  from the log_normal distribution
        # (2) { A < x < B } --> { 0 < (x-A)/(B-A) < 1 }  from the desired distribution
        # (1) & (2)  x-A = r/3 (B-A)


# ========================================================
# UTILITY FUNCTIONS: archived for later use
# ========================================================
def chop_to_1(x):
    return round(x, -int(np.floor(np.log10(abs(x)))) + 0)


# ========================================================
def get_natural():
    global R0, S_hat, u, Ca, exponent, S_hat, p0, rho, S, pv
    global freq_range, dt, EPS, amp_range
    global amp_range, amp_min, amp_max
    EPS = np.finfo(float).eps
    global time_constant
    time_constant = R0 * (rho / p0) ** (1 / 2)
    global Re
    v = u / rho
    if (v < EPS):
        Re = np.inf
    else:
        Re = (R0 / v) * (p0 / rho) ** (1 / 2)
    global S_hat
    S_hat = p0 * R0 / S
    global Ca
    Ca = (p0 - pv) / p0
    #--------------------------------------------------
    freq_natural = 3 * exponent * Ca + 2 * (3 * exponent - 1) / (S_hat * R0)
    freq_natural = np.sqrt(freq_natural / (R0 ** 2))
    T_natural = 1 / freq_natural
    print(f"T_natural = {T_natural}")
    dt = chop_to_1(T_natural)
    print(f"dt = {dt}")
    global freq_range
    freq_range = [ freq_min * freq_natural, freq_max * freq_natural ]
    amp_range = [amp_min, amp_max]
