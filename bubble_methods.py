
# ## adapted by Scott Sims 05/11/2022
import numpy as np
from scipy.optimize import fsolve

#=========================================================
# Function: Calculates number of steps for total duration 
#=========================================================
def get_num_steps(dt, model_steps, k_max, period_min, n_periods):
    max_steps = np.int64(np.round(model_steps * 2**k_max)) # max slice size
    return np.int64(np.round( max_steps * np.ceil( n_periods * period_min / (dt * max_steps)) ))

#==========================================
#  CLASS BUBBLE
#==========================================
class Bubble:
    def __init__(self, parameters):

        self.R_eq = parameters['R_eq']
        self.P_eq = parameters['P_eq']
        self.P_vap = parameters['P_vap']
        self.poly_index = parameters['poly_index']
        self.surface_tension = parameters['surface_tension']
        self.density = parameters['density']
        self.viscosity_dynamic = parameters['viscosity_dynamic']
        self.time_constant = self.R_eq * (self.density/self.P_eq)**(1/2)
        self.Ca = (self.P_eq - self.P_vap) / self.P_eq
        self.S = self.P_eq * self.R_eq / self.surface_tension
        self.viscosity_kinematic = self.viscosity_dynamic / self.density
        #---------------------------------------------------
        if (self.viscosity_kinematic < np.finfo(float).eps):
            self.reynolds = np.inf
        else:
            self.reynolds = (self.R_eq / self.viscosity_kinematic) * (self.P_eq / self.density) ** (1/2)
        #---------------------------------------------------
        self.freq_natural = np.sqrt( 3*self.poly_index*self.Ca + 2*(3 * self.poly_index - 1)/self.S )
        self.T_natural = 1 / self.freq_natural

    #========================================================
    # RAYLEIGH-PLESSET ODE (nondimensionalized)
    #========================================================
    def rhs_rp(self, t, y, sound):
        # R, Rdot = y[0], y[1]
        Cp = sound.pressure(t)
        #---------------------------------------------------
        # SYSTEM OF ODEs, the 1st and 2nd derivatives
        #----------------------------------------------------
        ydot = np.zeros(2)
        ydot[0] = y[1]
        ydot[1] = -(3 / 2) * (y[1] ** 2) - (4 / self.reynolds) * y[1] / y[0] - (2 / self.S) * (1 / y[0]) + (2 / self.S + self.Ca) * ( 1 / (y[0] ** (3*self.poly_index))) - (Cp + self.Ca)
        ydot[1] = ydot[1] / y[0]
        #---------------------------------------------------
        return ydot



# ========================================================
# CLASS: Sound Wave (composed of multiple pressure waves)
# ========================================================
class SoundWave:
    def __init__(self, amp_range, freq_range, num_waves, time_align=None):
        assert type(num_waves) == int
        assert num_waves >= int(1)
        assert np.size(amp_range) == 2
        assert np.size(freq_range) == 2
        # -----------------------------------------
        self.time_align = time_align
        self.n_waves = num_waves
        self.amp_range = amp_range
        self.freq_range = freq_range
        self.waves = self.generate_waves()
        # -----------------------------------------
        # check initial pressure and correct if magnitude too large
        # -----------------------------------------

    # ----------------------------------------------------------------
    def generate_waves(self):
        assert self.amp_range[1] >= self.amp_range[0]
        assert self.freq_range[1] >= self.freq_range[0]
        # -----------------------------------------
        amp_samples = self.uniform_normalized(self.amp_range[0], self.amp_range[1], self.n_waves)
        #amp_samples = self.lognormal_normalized(self.amp_range[0], self.amp_range[1], self.n_waves)
        amp_samples = np.sort(amp_samples) # increasing order, smallest to largest
        amp_samples = amp_samples[::-1] # reverse order, largest to smallest
        #freq_samples = np.random.uniform(self.freq_range[0], self.freq_range[1], self.n_waves)
        freq_samples = self.uniform_interval(self.freq_range[0], self.freq_range[1], self.n_waves)
        idx_max = np.argmax(freq_samples)
        idx = np.max([idx_max-2, 0])
        temp = freq_samples[idx_max]
        freq_samples[idx_max] = freq_samples[idx]
        freq_samples[idx] = temp
        #freq_samples = np.sort(freq_samples)
        #freq_samples = freq_samples[::-1]
        # -----------------------------------------
        waves_list = []
        for j in range(self.n_waves):
            waves_list.append(self.PressureWave(amp_samples[j], freq_samples[j]))
        # -----------------------------------------
        return waves_list

    # ----------------------------------------------------------------
    def pressure(self, t):
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
        assert np.size(self.waves) > 0
        sum_pos = 0.0
        sum_neg = 0.0
        # --------------------------------
        #if np.size(self.waves) == 0:
        #    print("no waves found. please initialize structure SoundWave")
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
    def lognormal_normalized(A, B, N):
        # RETURN: random sample following a log_normal distribution that sums to 'x', random in the interval {A < x < B}
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
        # (1) { 0 < r < max_r } --> { 0 < r/max_r < 1 }  the random distribution
        # (2) { A < x < B } --> { 0 < (x-A)/(B-A) < 1 }  the desired distribution
        # (1) & (2)  x-A = r/max_r (B-A)

    @staticmethod
    def poisson_normalized(A, B, N):
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
        # (1) { 0 < r < max_r } --> { 0 < r/max_r < 1 }  the random distribution
        # (2) { A < x < B } --> { 0 < (x-A)/(B-A) < 1 } the desired distribution
        # (1) & (2)  x-A = r/max_r (B-A)
     # --------------------------------------------------------------------
    @staticmethod
    def uniform_normalized(A, B, N):
        # RETURN: random sample following a uniform distribution that sums to 'x' in the interval {A < x < B}
        # A: minimum sum of samples
        # B: maximum sum of samples
        # ------------------------------------------
        assert B >= A
        r = np.random.uniform(0.3, 1, N)
        r = r * np.random.uniform(A,B) / np.sum(r)
        return r
        # expect_a = N/2 
        # scale = np.max([expect_a, sum_a*5*expect_a/3])
        # return scale * alpha
    #--------------------------------------------------------------------
    @staticmethod
    def uniform_interval(A, B, N):
        # RETURN: random sample following a uniform distribution
        # A: minimum sum of samples
        # B: maximum sum of samples
        # ------------------------------------------
        assert B >= A
        return np.random.uniform(A, B, N)  # normalized and then scaled to desired random sum
    # ========================================================
    # SUB-CLASS: Pressure Wave
    # ========================================================
    class PressureWave:
        __slots__ = ('amplitude','freq','phase')

        # ----------------------------------------------------------
        def __init__(self, amp, freq, phase=None):
            self.amplitude = amp
            self.freq = freq
            if phase == None: # if t0 not given, then randomly initialize within some interval {-T < t0 < 0} where T = 1/freq
                self.phase = np.random.uniform(0, 2*np.pi)
            else:
                self.phase = phase

        # ----------------------------------------------------------
        def get_pressure(self, t):
            # pressure at time 't'
            return self.amplitude * np.sin(2 * np.pi * self.freq * t + self.phase)
            #return pressure

        # ----------------------------------------------------------
        def get_pressure_dot(self, t):
            # time derivative of pressure at time 't'
            return self.amplitude * 2 * np.pi * self.freq * np.cos(2 * np.pi * self.freq * t + self.phase)
            #return pressure_dot

        # ---------------------------------------------------------
