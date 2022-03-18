
import numpy as np

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
# CLASS: Sound Wave (composed of multiple pressure waves)
# ========================================================
class SoundWave:
    def __init__(self, amp_range, freq_range, num_waves):
        assert type(num_waves) == int
        assert num_waves >= int(1)
        assert np.size(amp_range) == 2
        assert np.size(freq_range) == 2
        # -----------------------------------------
        self.n_waves = num_waves
        self.amp_range = amp_range
        self.freq_range = freq_range
        self.waves = self.generate_waves()
        #print("__init__(SoundWave)")
        #print(f"amp_range = {amp_range}")

    # ----------------------------------------------------------------
    def generate_waves(self):
        assert self.amp_range[1] >= self.amp_range[0]
        assert self.freq_range[1] >= self.freq_range[0]
        # -----------------------------------------
        amp_samples = self.poisson_normalize(self.amp_range[0], self.amp_range[1], self.n_waves)
        amp_samples = np.sort(amp_samples) # increasing order, smallest to largest
        freq_samples = self.lognormal_interval(self.freq_range[0], self.freq_range[1], self.n_waves)
        #freq_samples = np.sort(freq_samples)
        #freq_samples = freq_samples[::-1]  # decreasing order, largest to smallest
        #print(f"sum of amps before = {np.sum(amp_samples)}")
        # -----------------------------------------
        waves_list = list()
        for j in range(self.n_waves):
            waves_list.append(self.PressureWave(amp_samples[j], freq_samples[j]))
        return waves_list
    # ----------------------------------------------------------------
    def pressure(self, t):
        assert t >= 0.0
        assert np.size(self.waves) > 0
        sum_pos = 0.0
        sum_neg = 0.0
        # ---------------------------------
        #if np.size(self.waves) == 0:
        #    self.generate_waves()
        # --------------------------------
        for j in range(np.size(self.waves)):
            temp = self.waves[j].get_pressure(t)
            #sum_abs = sum_abs + abs(temp)
            if temp >= 0.0:
                sum_pos += temp  # sums all positive pressure to avoid catastrophic cancellation
            else:
                sum_neg += temp  # sums all negative pressures to avoid catastrophic cancellation
        #print("pressure(t):")
        #print(f"mean(|pressure|) = {sum_abs/self.n_waves}")
        # --------------------------------
        return sum_neg + sum_pos
    # --------------------------------------------------------------------

    def pressure_dot(self, t):
        assert t >= 0.0
        assert np.size(self.waves) > 0
        sum_pos = 0.0
        sum_neg = 0.0
        # --------------------------------
        #if np.size(self.waves) == 0:
        #    self.generate_waves(self)
        # --------------------------------
        for j in range(np.size(self.waves)):
            temp = self.waves[j].get_pressure_dot(t)
            if temp >= 0:
                sum_pos += temp  # sums all positive pressure to avoid catastrophic cancellation
            else:
                sum_neg += temp  # sums all negative pressures to avoid catastrophic cancellation
        # --------------------------------
        return sum_pos + sum_neg

    # --------------------------------------------------------------------
    @staticmethod
    def lognormal_normalize(A, B, N):
        # RETURN: random sample following a log_normal distribution that sums to 'x' in the interval {A < x < B}
        # A: minimum sum of samples
        # B: maximum sum of samples
        # ------------------------------------------
        assert B >= A
        mu, sigma = 0.0, 1.0
        r = np.random.lognormal(mu, sigma, N)
        r = r / np.sum(r)
        return r * np.random.uniform(A,B)  # normalized and then scaled to desired random sum
    # --------------------------------------------------------------------
    @staticmethod
    def lognormal_interval(A, B, N):
        # RETURN: random sample following a log_normal distribution mapped to the interval {A < x < B}
        # A: minimum of any sample
        # B: maximum of any sample
        # note: 4 is the "apparent" max of log_normal(0,1)
        # ------------------------------------------
        assert B >= A
        mu, sigma = 0.0, 1.0
        r = np.random.lognormal(mu, sigma, N)
        max_r = np.max([4,np.max(r)])
        return np.ones(N) * A + r * (B - A) / max_r
        # ------------------------------------------
        # derivation of interval scaling:
        # (1) { 0 < r < M } --> { 0 < r/M < 1 }  the random distribution
        # (2) { A < x < B } --> { 0 < (x-A)/(B-A) < 1 }  the desired distribution
        # (1) & (2)  x-A = r/M (B-A)

    @staticmethod
    def poisson_normalize(A, B, N):
        # RETURN: random sample following a poisson distribution that sums to 'x' in the interval {A < x < B}
        # A: minimum sum of samples
        # B: maximum sum of samples
        # ------------------------------------------
        assert B >= A
        mu = 3.0
        r = np.random.poisson(mu, N)
        r = r / np.sum(r)
        return r * np.random.uniform(A, B)  # normalized and then scaled to desired random sum

    # --------------------------------------------------------------------
    @staticmethod
    def poisson_interval(A, B, N):
        # RETURN: random sample following a poisson distribution mapped to the interval {A < x < B}
        # A: minimum of any sample
        # B: maximum of any sample
        # note: 9 is the "apparent" max of poisson(3)
        # ------------------------------------------
        assert B >= A
        mu = 3.0
        r = np.random.poisson(mu, N)
        max_r = np.max([9, np.max(r)])
        return np.ones(N) * A + r * (B - A) / max_r
        # ------------------------------------------
        # derivation of interval scaling:
        # (1) { 0 < r < M } --> { 0 < r/M < 1 }  the random distribution
        # (2) { A < x < B } --> { 0 < (x-A)/(B-A) < 1 } the desired distribution
        # (1) & (2)  x-A = r/M (B-A)

    # ========================================================
    # SUB-CLASS: Pressure Wave
    # ========================================================
    class PressureWave:
        __slots__ = ('amplitude','frequency','time_init')

        # ----------------------------------------------------------
        def __init__(self, amp, freq, t0=None):
            self.amplitude = amp
            self.frequency = freq
            if t0 == None:
                self.time_init = (-1) * np.random.uniform(0, 1) * (1 / freq)
            else:
                self.time_init = t0

        # ----------------------------------------------------------
        def get_pressure(self, t):
            duration = t - self.time_init
            assert duration >= 0
            return self.amplitude * np.sin(2 * np.pi * self.frequency * duration)
            #return pressure

        # ----------------------------------------------------------
        def get_pressure_dot(self, t):
            duration = t - self.time_init
            assert duration >= 0
            return self.amplitude * 2 * np.pi * self.frequency * np.cos(2 * np.pi * self.frequency * duration)
            #return pressure_dot

        # ---------------------------------------------------------

# ========================================================
# UTILITY FUNCTIONS: archived for later use
# ========================================================