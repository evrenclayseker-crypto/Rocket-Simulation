#!/usr/bin/env python3
"""
Complex Multi-Stage Rocket Trajectory & Environmental Forces Simulator.
Complete, self-contained Python program simulating multi-stage rocket flight from
launch pad liftoff through the atmosphere into low Earth orbit.

Features:
- US Standard Atmosphere 1976 (0 to 1,000 km continuous temperature lapse and scale height layers).
- WGS-84 Oblate Spheroid Gravity with J2 & J3 zonal harmonics.
- Aerodynamic Lift and Mach-dependent Drag (subsonic, transonic drag rise, supersonic shock decay).
- Non-Inertial Reference Frame Transformations (ECI <-> ECEF, Earth rotation, Coriolis & Centrifugal effects).
- Solar Radiation Pressure (SRP) and Third-body gravitational perturbations.
- Dynamic Pressure (Max-Q) calculation and stagnation aero-thermal convective heat flux.
- Modular Component Architecture: easily swap engines, stages, dry mass, fuel, diameter, and payload.
- Adaptive Runge-Kutta 4th/5th order (RK45) trajectory integrator with staging event detection.
- Publication-quality 6-panel flight telemetry plot generation.
"""
"""
US Standard Atmosphere 1976 model with extended thermosphere/exosphere layers.
Computes temperature, pressure, density, speed of sound, and dynamic viscosity from sea level to 1,000 km.
"""
import numpy as np

class USStandardAtmosphere1976:
    # Universal gas constant and sea level constants
    R_GAS = 8.31432e3          # J / (kmol * K)
    M_AIR = 28.9644             # kg / kmol (mean molecular weight at sea level)
    R_SPECIFIC = 287.05287      # J / (kg * K)
    GAMMA = 1.40                # Specific heat ratio for diatomic air
    G0 = 9.80665                # Standard acceleration of gravity (m/s^2)
    R_EARTH = 6356766.0         # Nominal Earth radius for geopotential conversion (m)
    
    # Sea-level standard values
    P0 = 101325.0               # Pa
    T0 = 288.15                 # K
    RHO0 = 1.2250               # kg / m^3
    
    # Layer definitions up to 86 km geopotential altitude (H_b in m, L_b in K/m, T_b in K, P_b in Pa)
    # Layer: [Base Geopotential Alt (m), Lapse Rate (K/m), Base Temp (K)]
    LAYERS = [
        (0.0,      -0.0065,  288.15),   # Troposphere
        (11000.0,   0.0000,  216.65),   # Tropopause
        (20000.0,   0.0010,  216.65),   # Stratosphere 1
        (32000.0,   0.0028,  228.65),   # Stratosphere 2
        (47000.0,   0.0000,  270.65),   # Stratopause
        (51000.0,  -0.0028,  270.65),   # Mesosphere 1
        (71000.0,  -0.0020,  214.65),   # Mesosphere 2
        (84852.0,   0.0000,  186.95),   # Mesopause
    ]

    def __init__(self):
        # Precompute base pressures for standard layers
        self.base_pressures = [self.P0]
        for i in range(len(self.LAYERS) - 1):
            h_b, lapse, t_b = self.LAYERS[i]
            h_next = self.LAYERS[i+1][0]
            p_b = self.base_pressures[-1]
            if abs(lapse) < 1e-10:
                p_next = p_b * np.exp(-self.G0 * (h_next - h_b) / (self.R_SPECIFIC * t_b))
            else:
                t_next = t_b + lapse * (h_next - h_b)
                p_next = p_b * (t_next / t_b) ** (-self.G0 / (lapse * self.R_SPECIFIC))
            self.base_pressures.append(p_next)

    def geometric_to_geopotential(self, z):
        """Convert geometric altitude z (m) to geopotential altitude H (m)."""
        return (self.R_EARTH * z) / (self.R_EARTH + z)

    def properties(self, altitude_m):
        """
        Calculates atmospheric properties at a given geometric altitude (m).
        Returns dict:
            'temperature_K': Temperature in Kelvin
            'pressure_Pa': Static pressure in Pascals
            'density_kg_m3': Atmospheric density in kg/m^3
            'speed_of_sound_m_s': Speed of sound in m/s
            'dynamic_viscosity_Pa_s': Dynamic viscosity (Sutherland formula)
        """
        z = max(0.0, float(altitude_m))
        h = self.geometric_to_geopotential(z)
        
        # 1. Lower to Upper Mesosphere (0 to 84.852 km geopotential)
        if h <= self.LAYERS[-1][0]:
            # Find applicable layer
            layer_idx = 0
            for i in range(len(self.LAYERS) - 1, -1, -1):
                if h >= self.LAYERS[i][0]:
                    layer_idx = i
                    break
            
            h_b, lapse, t_b = self.LAYERS[layer_idx]
            p_b = self.base_pressures[layer_idx]
            dh = h - h_b
            
            if abs(lapse) < 1e-10:
                T = t_b
                P = p_b * np.exp(-self.G0 * dh / (self.R_SPECIFIC * T))
            else:
                T = t_b + lapse * dh
                P = p_b * (T / t_b) ** (-self.G0 / (lapse * self.R_SPECIFIC))
            
            rho = P / (self.R_SPECIFIC * T)
            
        # 2. Thermosphere and Exosphere (above 86 km to 1,000 km)
        else:
            # Transition to thermosphere: molecular mass decreases, temperature rises, density drops exponentially
            # Reference values at 86 km
            h_86 = self.LAYERS[-1][0]
            T_86 = 186.95
            P_86 = self.base_pressures[-1]
            rho_86 = P_86 / (self.R_SPECIFIC * T_86)
            
            dz = z - 86000.0
            if z < 120000.0:
                # Lower thermosphere: T increases with altitude up to ~360 K
                T = T_86 + (360.0 - T_86) * (dz / 34000.0)
                H_scale = (self.R_SPECIFIC * T) / self.G0
                rho = rho_86 * np.exp(-dz / 6000.0)
                P = rho * self.R_SPECIFIC * T
            elif z < 250000.0:
                # Upper thermosphere: temperature asymptotes to ~900 K
                T = 360.0 + 540.0 * (1.0 - np.exp(-(z - 120000.0) / 45000.0))
                # Atmospheric density at 120 km is ~2.4e-8 kg/m3
                rho = 2.4e-8 * np.exp(-(z - 120000.0) / 18000.0)
                P = rho * self.R_SPECIFIC * T
            elif z < 1000000.0:
                # Exosphere: atomic oxygen / helium / hydrogen diffuse regime
                T = 950.0
                rho = 1.8e-11 * np.exp(-(z - 250000.0) / 50000.0)
                P = rho * self.R_SPECIFIC * T
            else:
                # Deep space / vacuum floor
                T = 3.0
                P = 1e-14
                rho = 1e-17
        
        # Guard against zero or negative density/pressure
        rho = max(rho, 1e-17)
        P = max(P, 1e-14)
        T = max(T, 1.0)
        
        # Speed of sound a = sqrt(gamma * R * T)
        a = np.sqrt(self.GAMMA * self.R_SPECIFIC * T)
        
        # Sutherland's Law for dynamic viscosity
        # mu = mu_0 * (T / T_0)**(3/2) * (T_0 + S) / (T + S)
        mu0 = 1.716e-5  # Pa * s
        T_ref = 273.15  # K
        S = 110.4       # Sutherland constant for air (K)
        mu = mu0 * ((T / T_ref) ** 1.5) * ((T_ref + S) / (T + S))
        
        return {
            'temperature_K': float(T),
            'pressure_Pa': float(P),
            'density_kg_m3': float(rho),
            'speed_of_sound_m_s': float(a),
            'dynamic_viscosity_Pa_s': float(mu),
        }

"""
Gravitational Field Model incorporating WGS-84 reference ellipsoid,
J2 and J3 zonal harmonic perturbations, and third-body Lunar/Solar gravitation.
"""
import numpy as np

class WGS84Gravity:
    # WGS-84 Earth Constants
    MU_EARTH = 3.986004418e14       # Geocentric gravitational constant (m^3 / s^2)
    R_EARTH_EQ = 6378137.0          # Semi-major equatorial radius (m)
    R_EARTH_POL = 6356752.3142      # Semi-minor polar radius (m)
    FLATTENING = 1.0 / 298.257223563
    J2 = 1.08262668e-3              # Second zonal harmonic coefficient (equatorial bulge)
    J3 = -2.5327e-6                 # Third zonal harmonic coefficient (pear-shaped asymmetry)
    
    # Third-Body Gravitational Constants
    MU_MOON = 4.9048695e12          # m^3 / s^2
    R_MOON_ORBIT = 3.844e8          # Nominal Moon orbital radius (m)
    MU_SUN = 1.32712440018e20       # m^3 / s^2
    R_SUN_ORBIT = 1.496e11          # 1 AU (m)

    def __init__(self, enable_j2=True, enable_j3=True, enable_third_body=False):
        self.enable_j2 = enable_j2
        self.enable_j3 = enable_j3
        self.enable_third_body = enable_third_body

    def acceleration(self, r_vec, time_sec=0.0):
        """
        Calculates the gravitational acceleration vector (m/s^2) in ECI coordinates
        for a given position vector r_vec = [x, y, z] in meters.
        """
        x, y, z = float(r_vec[0]), float(r_vec[1]), float(r_vec[2])
        r2 = x*x + y*y + z*z
        r = np.sqrt(r2)
        
        if r < 1000.0:
            return np.array([0.0, 0.0, -9.81])
            
        # 1. Newtonian Point Mass Gravity
        inv_r3 = 1.0 / (r2 * r)
        g_point = -self.MU_EARTH * inv_r3 * np.array([x, y, z])
        
        # 2. J2 Zonal Harmonic Perturbation (Equatorial Bulge)
        a_j2 = np.zeros(3)
        if self.enable_j2:
            r5 = r2 * r2 * r
            factor_j2 = 1.5 * self.MU_EARTH * self.J2 * (self.R_EARTH_EQ ** 2) / r5
            z2_over_r2 = (z * z) / r2
            
            a_j2[0] = factor_j2 * x * (5.0 * z2_over_r2 - 1.0)
            a_j2[1] = factor_j2 * y * (5.0 * z2_over_r2 - 1.0)
            a_j2[2] = factor_j2 * z * (5.0 * z2_over_r2 - 3.0)
            
        # 3. J3 Zonal Harmonic Perturbation
        a_j3 = np.zeros(3)
        if self.enable_j3:
            r7 = r5 * r2
            factor_j3 = 0.5 * self.MU_EARTH * self.J3 * (self.R_EARTH_EQ ** 3) / r7
            z_over_r = z / r
            z2_over_r2 = z_over_r * z_over_r
            
            a_j3[0] = factor_j3 * 5.0 * x * (7.0 * z2_over_r2 - 3.0) * z_over_r
            a_j3[1] = factor_j3 * 5.0 * y * (7.0 * z2_over_r2 - 3.0) * z_over_r
            a_j3[2] = factor_j3 * (35.0 * (z2_over_r2 * z2_over_r2) - 30.0 * z2_over_r2 + 3.0)
            
        # 4. Third-Body Perturbations (Moon and Sun point-mass positions)
        a_third = np.zeros(3)
        if self.enable_third_body and r > 100000.0:
            # Approximate Moon position along circular orbit inclined at 5.14 deg
            omega_moon = 2.0 * np.pi / (27.32 * 86400.0)
            theta_m = omega_moon * time_sec
            r_moon = self.R_MOON_ORBIT * np.array([np.cos(theta_m), np.sin(theta_m), 0.0])
            delta_m = r_moon - np.array([x, y, z])
            d_m3 = np.linalg.norm(delta_m) ** 3
            r_m3 = self.R_MOON_ORBIT ** 3
            a_moon = self.MU_MOON * (delta_m / d_m3 - r_moon / r_m3)
            
            # Approximate Sun position along ecliptic
            omega_sun = 2.0 * np.pi / (365.25 * 86400.0)
            theta_s = omega_sun * time_sec
            eps_ecliptic = np.radians(23.44)
            r_sun = self.R_SUN_ORBIT * np.array([
                np.cos(theta_s),
                np.sin(theta_s) * np.cos(eps_ecliptic),
                np.sin(theta_s) * np.sin(eps_ecliptic)
            ])
            delta_s = r_sun - np.array([x, y, z])
            d_s3 = np.linalg.norm(delta_s) ** 3
            r_s3 = self.R_SUN_ORBIT ** 3
            a_sun = self.MU_SUN * (delta_s / d_s3 - r_sun / r_s3)
            
            a_third = a_moon + a_sun
            
        return g_point + a_j2 + a_j3 + a_third

    def sea_level_surface_radius(self, geodetic_lat_rad):
        """Radius of the WGS-84 reference ellipsoid at a given latitude."""
        a = self.R_EARTH_EQ
        b = self.R_EARTH_POL
        phi = geodetic_lat_rad
        num = (a * a * np.cos(phi))**2 + (b * b * np.sin(phi))**2
        den = (a * np.cos(phi))**2 + (b * np.sin(phi))**2
        return np.sqrt(num / den)

"""
Aerodynamic Forces Model: Mach-dependent drag coefficient CD(M),
compressibility wave drag, angle of attack normal force/lift,
center of pressure (CP), and stagnation aero-thermal heating.
"""
import numpy as np

class AerodynamicsModel:
    def __init__(self, reference_area_m2, nosecone_fineness=2.5, base_diameter_m=3.7):
        self.s_ref = reference_area_m2
        self.fineness = nosecone_fineness
        self.diameter = base_diameter_m
        
    def drag_coefficient(self, mach, angle_of_attack_rad=0.0):
        """
        Computes total drag coefficient CD(M, alpha) across subsonic, transonic,
        supersonic, and hypersonic regimes using empirical supersonic wave drag theory.
        """
        m = max(0.001, float(mach))
        alpha = float(angle_of_attack_rad)
        
        # Zero-lift drag coefficient CD0(M)
        if m < 0.8:
            # Subsonic: skin friction + form drag
            cd0 = 0.28 - 0.03 * (m / 0.8)
        elif m < 1.05:
            # Transonic drag rise (critical Mach divergence peak)
            t = (m - 0.8) / 0.25
            cd0 = 0.25 + 0.55 * (3 * t**2 - 2 * t**3)
        elif m < 1.4:
            # Peak transonic wave drag
            t = (m - 1.05) / 0.35
            cd0 = 0.80 - 0.25 * t
        elif m < 5.0:
            # Supersonic wave drag decay: shock cone steepens
            cd0 = 0.55 / np.sqrt(max(0.1, m*m - 1.0)) + 0.18
        else:
            # Hypersonic asymptotic limit (modified Newtonian impact)
            cd0 = 0.26 + 0.05 / (m - 4.0)
            
        # Induced drag due to angle of attack: CDi = CN_alpha * sin^2(alpha)
        cn_alpha = 2.0 + (np.pi / 2.0) * (self.diameter / np.sqrt(self.s_ref))
        cd_induced = cn_alpha * (np.sin(alpha) ** 2)
        
        return cd0 + cd_induced

    def normal_force_coefficient(self, mach, angle_of_attack_rad=0.0):
        """
        Computes normal/lift force coefficient CN(M, alpha).
        Linear slender-body potential term + crossflow drag separation term.
        """
        alpha = float(angle_of_attack_rad)
        cn_alpha = 2.0 + 1.2 * min(3.0, mach)
        cdc = 1.15  # Crossflow drag coefficient of a cylinder
        cn = cn_alpha * np.sin(alpha) * np.cos(alpha) + cdc * (np.sin(alpha) ** 2)
        return cn

    def compute_forces(self, airspeed_rel_m_s, density_kg_m3, mach, angle_of_attack_rad=0.0):
        """
        Calculates aerodynamic drag, lift/normal force, dynamic pressure (Max-Q),
        and stagnation point aerodynamic heating flux.
        """
        v = float(airspeed_rel_m_s)
        rho = float(density_kg_m3)
        
        # Dynamic pressure q = 0.5 * rho * v^2
        q = 0.5 * rho * (v ** 2)
        
        cd = self.drag_coefficient(mach, angle_of_attack_rad)
        cn = self.normal_force_coefficient(mach, angle_of_attack_rad)
        
        drag_force = q * self.s_ref * cd
        lift_force = q * self.s_ref * cn
        
        # Stagnation convective heat flux: Detra-Kemp-Riddell / Sutton-Graves formula
        # q_dot = k * sqrt(rho / Rn) * v^3 (W/m^2)
        rn = 0.5 * (self.diameter / self.fineness)  # Effective nose radius (m)
        k_heat = 1.83e-4  # Convective heating constant for Earth air
        if v > 100.0 and rho > 1e-12:
            heat_flux_w_m2 = k_heat * np.sqrt(rho / max(0.05, rn)) * (v ** 3)
        else:
            heat_flux_w_m2 = 0.0
            
        return {
            'dynamic_pressure_Pa': q,
            'drag_force_N': drag_force,
            'lift_force_N': lift_force,
            'drag_coefficient': cd,
            'normal_coefficient': cn,
            'heat_flux_W_m2': heat_flux_w_m2,
        }

    def center_of_pressure(self, mach, rocket_length_m):
        """
        Estimates the aerodynamic Center of Pressure (distance from nosecone tip in meters).
        Subsonic: CP is around 0.45 * length. Supersonic: shifts rearward to ~0.55 * length.
        """
        if mach < 0.8:
            return 0.45 * rocket_length_m
        elif mach < 1.3:
            t = (mach - 0.8) / 0.5
            return (0.45 + 0.10 * t) * rocket_length_m
        else:
            return 0.55 * rocket_length_m

"""
Coordinate Transformations between Earth-Centered Inertial (ECI)
and Earth-Centered Earth-Fixed (ECEF), Geodetic coordinates,
and Earth Rotation dynamics (Coriolis and Centrifugal forces).
"""
import numpy as np

class CelestialMechanics:
    OMEGA_EARTH = 7.292115e-5       # Earth rotation angular velocity (rad/s)
    WGS84_A = 6378137.0             # Semi-major axis (m)
    WGS84_F = 1.0 / 298.257223563   # Flattening
    WGS84_B = WGS84_A * (1.0 - WGS84_F)
    WGS84_E2 = 2.0 * WGS84_F - WGS84_F ** 2

    @classmethod
    def ecef_to_eci_rotation(cls, time_sec):
        """Returns 3x3 rotation matrix from ECEF to ECI at time t."""
        theta = cls.OMEGA_EARTH * time_sec
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        return np.array([
            [cos_t, -sin_t, 0.0],
            [sin_t,  cos_t, 0.0],
            [0.0,    0.0,   1.0]
        ])

    @classmethod
    def eci_to_ecef(cls, r_eci, v_eci, time_sec):
        """Transforms position and velocity from ECI to ECEF frame."""
        theta = cls.OMEGA_EARTH * time_sec
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # R_z(-theta)
        r_ecef = np.array([
            cos_t * r_eci[0] + sin_t * r_eci[1],
           -sin_t * r_eci[0] + cos_t * r_eci[1],
            r_eci[2]
        ])
        
        # Velocity in rotating frame: v_ecef = R_z(-theta) * (v_eci - omega x r_eci)
        v_rot = v_eci - np.array([
            -cls.OMEGA_EARTH * r_eci[1],
             cls.OMEGA_EARTH * r_eci[0],
             0.0
        ])
        v_ecef = np.array([
            cos_t * v_rot[0] + sin_t * v_rot[1],
           -sin_t * v_rot[0] + cos_t * v_rot[1],
            v_rot[2]
        ])
        return r_ecef, v_ecef

    @classmethod
    def ecef_to_geodetic(cls, r_ecef):
        """
        Converts ECEF Cartesian position [x, y, z] to Geodetic Latitude (rad),
        Longitude (rad), and Altitude above WGS84 ellipsoid (m) using Bowring's algorithm.
        """
        x, y, z = r_ecef[0], r_ecef[1], r_ecef[2]
        lon = np.arctan2(y, x)
        
        p = np.sqrt(x*x + y*y)
        if p < 1e-6:
            lat = np.pi / 2.0 if z > 0 else -np.pi / 2.0
            alt = abs(z) - cls.WGS84_B
            return lat, lon, alt
            
        theta = np.arctan2(z * cls.WGS84_A, p * cls.WGS84_B)
        lat = np.arctan2(
            z + cls.WGS84_E2 * cls.WGS84_B * (np.sin(theta) ** 3),
            p - cls.WGS84_E2 * cls.WGS84_A * (np.cos(theta) ** 3)
        )
        
        n_rad = cls.WGS84_A / np.sqrt(1.0 - cls.WGS84_E2 * (np.sin(lat) ** 2))
        alt = p / np.cos(lat) - n_rad
        return float(lat), float(lon), float(alt)

    @classmethod
    def atmospheric_velocity_eci(cls, r_eci, wind_vector_ned=None):
        """
        Velocity of the atmosphere at position r_eci in ECI frame.
        Accounts for Earth's rotation (co-rotating atmosphere) and optional local wind.
        """
        # Co-rotating atmosphere velocity: omega_earth x r_eci
        v_atm = np.array([
            -cls.OMEGA_EARTH * r_eci[1],
             cls.OMEGA_EARTH * r_eci[0],
             0.0
        ])
        if wind_vector_ned is not None:
            # Transform NED wind to ECI if provided
            v_atm = v_atm + wind_vector_ned
        return v_atm

"""
Solar Radiation Pressure (SRP) Model for exo-atmospheric trajectory phases.
Includes Earth cylindrical/conical shadow (umbra and penumbra) calculation.
"""
import numpy as np

class SolarRadiationPressure:
    SOLAR_FLUX_1AU = 1361.0         # W / m^2 (Solar constant)
    SPEED_OF_LIGHT = 299792458.0    # m / s
    P_SRP_1AU = SOLAR_FLUX_1AU / SPEED_OF_LIGHT # ~4.54e-6 N / m^2
    R_EARTH = 6378137.0
    R_SUN = 6.9634e8
    AU_M = 1.495978707e11

    def __init__(self, reflectivity_coeff=1.2):
        self.cr = reflectivity_coeff # 1.0 = blackbody absorption, 2.0 = specular reflection

    def shadow_factor(self, r_vec, r_sun_vec):
        """
        Calculates the fraction of sunlight reaching the vehicle (0 = full shadow, 1 = direct sunlight).
        """
        r_mag = np.linalg.norm(r_vec)
        if r_mag < 1.0:
            return 0.0
            
        r_unit = r_vec / r_mag
        s_unit = r_sun_vec / np.linalg.norm(r_sun_vec)
        cos_angle = np.dot(r_unit, s_unit)
        
        if cos_angle >= 0.0:
            return 1.0 # Dayside
            
        # Nightside: check perpendicular distance to Earth-Sun line
        dist_perp = r_mag * np.sqrt(max(0.0, 1.0 - cos_angle**2))
        if dist_perp < self.R_EARTH:
            return 0.0 # Umbra
        return 1.0

    def acceleration(self, r_vec, vehicle_mass_kg, illuminated_area_m2, time_sec=0.0):
        """
        Calculates the SRP acceleration vector (m/s^2) acting on the spacecraft.
        """
        if vehicle_mass_kg <= 0.0:
            return np.zeros(3)
            
        # Nominal Sun vector in ECI
        omega_sun = 2.0 * np.pi / (365.25 * 86400.0)
        theta_s = omega_sun * time_sec
        eps = np.radians(23.44)
        r_sun = self.AU_M * np.array([
            np.cos(theta_s),
            np.sin(theta_s) * np.cos(eps),
            np.sin(theta_s) * np.sin(eps)
        ])
        
        nu = self.shadow_factor(r_vec, r_sun)
        if nu <= 0.0:
            return np.zeros(3)
            
        s_unit = r_sun / np.linalg.norm(r_sun)
        force_mag = nu * self.P_SRP_1AU * self.cr * illuminated_area_m2
        a_srp = (force_mag / vehicle_mass_kg) * s_unit
        return a_srp

"""
Liquid and Solid Rocket Engine Model.
Calculates pressure-dependent thrust, specific impulse (Isp),
mass flow rate, nozzle expansion ratio compensation, and gimbaling.
"""
import numpy as np

class RocketEngine:
    G0 = 9.80665 # Standard gravity (m/s^2)

    def __init__(self, name, thrust_sl_N, thrust_vac_N, isp_sl_s, isp_vac_s,
                 nozzle_exit_area_m2=0.5, gimbal_limit_deg=5.0, throttle_min=0.4, throttle_max=1.0):
        self.name = name
        self.thrust_sl = float(thrust_sl_N)
        self.thrust_vac = float(thrust_vac_N)
        self.isp_sl = float(isp_sl_s)
        self.isp_vac = float(isp_vac_s)
        self.nozzle_exit_area = float(nozzle_exit_area_m2)
        self.gimbal_limit_rad = np.radians(float(gimbal_limit_deg))
        self.throttle_min = float(throttle_min)
        self.throttle_max = float(throttle_max)
        
        # Nominal mass flow rate at 100% throttle: m_dot = F_vac / (Isp_vac * g0)
        self.nominal_mass_flow = self.thrust_vac / (self.isp_vac * self.G0)

    def current_isp(self, ambient_pressure_Pa):
        """
        Calculates specific impulse as a function of ambient atmospheric pressure:
        Isp(p_a) = Isp_vac - (p_a * A_e) / (m_dot * g0)
        """
        pa = max(0.0, float(ambient_pressure_Pa))
        # Delta Isp = (pa * A_e) / (m_dot * g0)
        delta_isp = (pa * self.nozzle_exit_area) / (self.nominal_mass_flow * self.G0)
        isp = self.isp_vac - delta_isp
        return max(self.isp_sl * 0.8, min(self.isp_vac, isp))

    def current_thrust(self, ambient_pressure_Pa, throttle=1.0):
        """
        Calculates thrust force in Newtons accounting for ambient pressure backpressure:
        F(p_a) = throttle * (F_vac - p_a * A_e)
        """
        pa = max(0.0, float(ambient_pressure_Pa))
        thrott = max(self.throttle_min, min(self.throttle_max, float(throttle)))
        f_ideal = self.thrust_vac - (pa * self.nozzle_exit_area)
        f_clamped = max(0.0, f_ideal)
        return thrott * f_clamped

    def mass_flow_rate(self, throttle=1.0):
        """Propellant consumption rate (kg/s) at given throttle setting."""
        thrott = max(self.throttle_min, min(self.throttle_max, float(throttle)))
        return thrott * self.nominal_mass_flow

"""
Rocket Stage Model.
Manages dry structural mass, propellant mass depletion, multiple engine clusters,
geometry (diameter, length), and staging events.
"""
import numpy as np

class RocketStage:
    def __init__(self, name, dry_mass_kg, propellant_mass_kg, diameter_m, length_m,
                 engines, engine_count=1, staging_burnout_delay_s=2.0):
        self.name = name
        self.dry_mass = float(dry_mass_kg)
        self.propellant_mass = float(propellant_mass_kg)
        self.initial_propellant_mass = float(propellant_mass_kg)
        self.diameter = float(diameter_m)
        self.length = float(length_m)
        self.cross_sectional_area = np.pi * (self.diameter / 2.0) ** 2
        
        self.engines = engines if isinstance(engines, list) else [engines]
        self.engine_count = int(engine_count)
        self.staging_delay = float(staging_burnout_delay_s)
        self.is_ignited = False
        self.is_burned_out = False
        self.burn_time_elapsed = 0.0

    @property
    def total_mass(self):
        """Current total mass of this stage (dry + remaining fuel)."""
        return self.dry_mass + max(0.0, self.propellant_mass)

    def ignite(self):
        self.is_ignited = True

    def burn(self, dt, ambient_pressure_Pa, throttle=1.0):
        """
        Burns propellant for duration dt (seconds) at given throttle and ambient pressure.
        Returns:
            thrust_N: Total thrust produced by all engines in Newtons.
            mass_consumed_kg: Mass of propellant consumed in dt.
        """
        if not self.is_ignited or self.is_burned_out or self.propellant_mass <= 0.0:
            return 0.0, 0.0
            
        m_dot_total = sum(e.mass_flow_rate(throttle) for e in self.engines) * self.engine_count
        mass_demand = m_dot_total * dt
        
        if self.propellant_mass <= mass_demand:
            # Burn out occurs during this step
            actual_burned = self.propellant_mass
            dt_effective = actual_burned / m_dot_total if m_dot_total > 0 else 0.0
            self.propellant_mass = 0.0
            self.is_burned_out = True
            thrust_per_engine = sum(e.current_thrust(ambient_pressure_Pa, throttle) for e in self.engines)
            thrust = thrust_per_engine * self.engine_count * (dt_effective / dt)
            self.burn_time_elapsed += dt_effective
            return thrust, actual_burned
        else:
            self.propellant_mass -= mass_demand
            self.burn_time_elapsed += dt
            thrust_per_engine = sum(e.current_thrust(ambient_pressure_Pa, throttle) for e in self.engines)
            thrust = thrust_per_engine * self.engine_count
            return thrust, mass_demand

    def center_of_mass(self):
        """
        Estimated distance of center of mass from base of this stage (m).
        Propellant fills bottom portion, structural mass distributed.
        """
        fuel_fraction = self.propellant_mass / max(1.0, self.initial_propellant_mass)
        # Structural CG ~ 0.5 * length, Fuel CG ~ 0.4 * fuel_level
        cg_dry = 0.50 * self.length
        cg_fuel = 0.40 * self.length * fuel_fraction
        cm = (self.dry_mass * cg_dry + self.propellant_mass * cg_fuel) / self.total_mass
        return cm

"""
Multi-Stage Rocket Assembly.
Coordinates vehicle modularity: swap stages, change engines, customize fairings,
track total mass, center of mass / moment of inertia, and staging sequence.
"""
import numpy as np

class MultiStageRocket:
    def __init__(self, name, stages, payload_mass_kg=5000.0, fairing_mass_kg=1750.0,
                 fairing_jettison_alt_m=110000.0, fairing_diameter_m=5.2, nosecone_fineness=2.5):
        self.name = name
        self.stages = list(stages) # List of RocketStage instances ordered [Stage 1, Stage 2, ...]
        self.payload_mass = float(payload_mass_kg)
        self.fairing_mass = float(fairing_mass_kg)
        self.fairing_attached = True
        self.fairing_jettison_alt = float(fairing_jettison_alt_m)
        self.fairing_diameter = float(fairing_diameter_m)
        self.nosecone_fineness = float(nosecone_fineness)
        
        self.current_stage_idx = 0
        self.staging_history = []
        
        # Start Stage 1
        if len(self.stages) > 0:
            self.stages[0].ignite()

    @property
    def current_stage(self):
        if self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None

    @property
    def total_mass(self):
        """Total instantaneous mass of the vehicle (active stages + payload + fairing)."""
        m = self.payload_mass
        if self.fairing_attached:
            m += self.fairing_mass
        for idx in range(self.current_stage_idx, len(self.stages)):
            m += self.stages[idx].total_mass
        return m

    @property
    def reference_area(self):
        """Cross-sectional aerodynamic reference area (m^2)."""
        dia = self.fairing_diameter if self.fairing_attached else (
            self.current_stage.diameter if self.current_stage else 3.7
        )
        return np.pi * (dia / 2.0) ** 2

    @property
    def total_length(self):
        """Total length of the assembled active stages plus nosecone."""
        length = 10.0 # Nosecone/fairing length approx
        for idx in range(self.current_stage_idx, len(self.stages)):
            length += self.stages[idx].length
        return length

    def check_fairing_jettison(self, altitude_m, dynamic_pressure_Pa):
        """Jettisons fairing once above dense atmosphere and dynamic pressure is low."""
        if self.fairing_attached and altitude_m >= self.fairing_jettison_alt and dynamic_pressure_Pa < 100.0:
            self.fairing_attached = False
            return True
        return False

    def step(self, dt, ambient_pressure_Pa, altitude_m, dynamic_pressure_Pa, throttle=1.0):
        """
        Steps the propulsion and staging logic by dt.
        Returns:
            thrust_N: Current thrust produced.
            mass_consumed: Propellant burned in this step.
            events: List of event strings occurred in this step (e.g. MECO, SECO, Staging, Fairing Jettison).
        """
        events = []
        
        # Check fairing separation
        if self.check_fairing_jettison(altitude_m, dynamic_pressure_Pa):
            events.append("FAIRING_JETTISON")
            
        stage = self.current_stage
        if stage is None:
            return 0.0, 0.0, events
            
        thrust, burned = stage.burn(dt, ambient_pressure_Pa, throttle)
        
        # Check for stage burnout and stage separation
        if stage.is_burned_out:
            events.append(f"{stage.name}_BURNOUT")
            # Advance to next stage if available
            self.current_stage_idx += 1
            next_stage = self.current_stage
            if next_stage is not None:
                next_stage.ignite()
                events.append(f"STAGE_SEPARATION_{stage.name}_TO_{next_stage.name}")
            else:
                events.append("ALL_STAGES_BURNOUT")
                
        return thrust, burned, events

    def get_component_summary(self):
        """Returns structured dictionary of all rocket parts for inspection and customization."""
        summary = {
            'rocket_name': self.name,
            'total_initial_mass_kg': self.total_mass,
            'payload_mass_kg': self.payload_mass,
            'fairing_mass_kg': self.fairing_mass,
            'stages_count': len(self.stages),
            'stages': []
        }
        for i, s in enumerate(self.stages):
            summary['stages'].append({
                'index': i + 1,
                'name': s.name,
                'dry_mass_kg': s.dry_mass,
                'propellant_mass_kg': s.propellant_mass,
                'diameter_m': s.diameter,
                'length_m': s.length,
                'engine_count': s.engine_count,
                'engines': [e.name for e in s.engines]
            })
        return summary

"""
Ascent Guidance and Pitch Steering Program.
Controls vehicle attitude from vertical pad liftoff, initial pitch-kick,
zero-angle-of-attack gravity turn through dense atmosphere, to orbital insertion steering.
"""
import numpy as np

class AscentGuidance:
    def __init__(self, target_orbit_altitude_m=200000.0, pitch_kick_alt_m=1200.0,
                 pitch_kick_deg=2.2, launch_azimuth_deg=90.0):
        self.target_alt = float(target_orbit_altitude_m)
        self.kick_alt = float(pitch_kick_alt_m)
        self.kick_angle_rad = np.radians(float(pitch_kick_deg))
        self.azimuth_rad = np.radians(float(launch_azimuth_deg)) # 90 deg = Due East

    def compute_thrust_direction(self, altitude_m, v_rel_vector, position_vector_eci, t_flight):
        """
        Determines the commanded unit thrust vector in ECI coordinates.
        Uses relative velocity v_rel for aerodynamic gravity turn.
        """
        r_mag = np.linalg.norm(position_vector_eci)
        r_hat = position_vector_eci / r_mag
        
        # East and North unit vectors at current position
        k_hat = np.array([0.0, 0.0, 1.0]) # Earth rotational axis
        east_vec = np.cross(k_hat, r_hat)
        east_mag = np.linalg.norm(east_vec)
        if east_mag < 1e-6:
            east_hat = np.array([0.0, 1.0, 0.0])
        else:
            east_hat = east_vec / east_mag
            
        north_hat = np.cross(r_hat, east_hat)
        
        # Horizontal downrange vector along launch azimuth
        downrange_hat = np.sin(self.azimuth_rad) * east_hat + np.cos(self.azimuth_rad) * north_hat
        
        v_rel_mag = np.linalg.norm(v_rel_vector)
        
        # Phase 1: Vertical clearing phase (pad safety)
        if altitude_m < self.kick_alt or v_rel_mag < 60.0:
            return r_hat, "VERTICAL_ASCENT"
            
        # Phase 2: Pitch kick initiation (gentle pitch over)
        elif altitude_m < self.kick_alt + 2000.0:
            frac = (altitude_m - self.kick_alt) / 2000.0
            theta = self.kick_angle_rad * frac
            thrust_dir = np.cos(theta) * r_hat + np.sin(theta) * downrange_hat
            return thrust_dir / np.linalg.norm(thrust_dir), "PITCH_KICK"
            
        # Phase 3: Aerodynamic gravity turn (zero angle of attack relative to atmosphere)
        elif altitude_m < 70000.0:
            if v_rel_mag > 1.0:
                v_dir = v_rel_vector / v_rel_mag
                return v_dir, "GRAVITY_TURN"
            else:
                return r_hat, "VERTICAL_ASCENT"
                
        # Phase 4: Exo-atmospheric orbital insertion steering
        else:
            # Linear tangent pitch profile targeting circular LEO insertion
            alt_ratio = min(1.0, altitude_m / self.target_alt)
            # Pitch angle above local horizontal decreases to 0 degrees at insertion
            pitch_elev = np.radians(20.0 * max(0.0, (1.0 - alt_ratio)) ** 1.3)
            thrust_dir = np.sin(pitch_elev) * r_hat + np.cos(pitch_elev) * downrange_hat
            return thrust_dir / np.linalg.norm(thrust_dir), "ORBITAL_INSERTION"

"""
Adaptive Runge-Kutta-Fehlberg (RK45) 6-DOF Trajectory Integrator.
Simulates translation dynamics under all environmental forces,
tracks aero-thermal structural limits, and logs telemetry events.
"""
import numpy as np

class AdaptiveRK45Integrator:
    def __init__(self, tolerance=1e-5, min_step=0.005, max_step=1.0):
        self.tol = tolerance
        self.h_min = min_step
        self.h_max = max_step
        
        # Butcher Tableau for Cash-Karp / Dormand-Prince RK45
        self.A = [
            [],
            [1.0/5.0],
            [3.0/40.0, 9.0/40.0],
            [3.0/10.0, -9.0/10.0, 6.0/5.0],
            [-11.0/54.0, 5.0/2.0, -70.0/27.0, 35.0/27.0],
            [1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0]
        ]
        self.B4 = [2825.0/27648.0, 0.0, 18575.0/48384.0, 13525.0/55296.0, 277.0/14336.0, 1.0/4.0]
        self.B5 = [37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0]

    def step(self, f, t, y, h, args=()):
        """
        Executes an adaptive single step.
        Returns: y_new, t_new, h_next, error_estimate
        """
        k = []
        k.append(np.array(f(t, y, *args)))
        
        for i in range(1, 6):
            y_eval = y.copy()
            for j in range(i):
                y_eval += h * self.A[i][j] * k[j]
            k.append(np.array(f(t + h, y_eval, *args)))
            
        y4 = y.copy()
        y5 = y.copy()
        for i in range(6):
            y4 += h * self.B4[i] * k[i]
            y5 += h * self.B5[i] * k[i]
            
        error = np.linalg.norm(y5 - y4)
        
        # Determine next step size
        if error > self.tol and h > self.h_min:
            h_next = max(self.h_min, h * 0.7)
            # Reject step
            return y, t, h_next, error, False
        else:
            h_next = min(self.h_max, h * min(1.5, max(0.8, (self.tol / max(1e-12, error)) ** 0.2)))
            return y5, t + h, h_next, error, True

"""
Preset Launch Vehicle Configurations:
1. Falcon 9 Full Thrust (2-stage orbital vehicle with Merlin 1D engines)
2. Saturn V (3-stage super heavy-lift Apollo launch vehicle)
3. Electron (2-stage small satellite launch vehicle)
4. CustomModularRocket (Easily customizable user vehicle)
"""

def create_falcon_9_ft(payload_kg=5500.0):
    """Creates a Falcon 9 Full Thrust replica."""
    # Stage 1: 9 Merlin 1D engines
    m1d = RocketEngine(
        name="Merlin-1D",
        thrust_sl_N=845000.0,
        thrust_vac_N=914000.0,
        isp_sl_s=282.0,
        isp_vac_s=311.0,
        nozzle_exit_area_m2=0.65,
        gimbal_limit_deg=5.0
    )
    stage_1 = RocketStage(
        name="Falcon9-Stage1",
        dry_mass_kg=25600.0,
        propellant_mass_kg=395700.0,
        diameter_m=3.7,
        length_m=42.6,
        engines=[m1d],
        engine_count=9,
        staging_burnout_delay_s=2.5
    )
    
    # Stage 2: 1 Merlin Vacuum engine
    m_vac = RocketEngine(
        name="Merlin-1D-Vac",
        thrust_sl_N=600000.0,
        thrust_vac_N=981000.0,
        isp_sl_s=220.0,
        isp_vac_s=348.0,
        nozzle_exit_area_m2=3.0,
        gimbal_limit_deg=6.0
    )
    stage_2 = RocketStage(
        name="Falcon9-Stage2",
        dry_mass_kg=4000.0,
        propellant_mass_kg=92670.0,
        diameter_m=3.7,
        length_m=12.6,
        engines=[m_vac],
        engine_count=1,
        staging_burnout_delay_s=1.0
    )
    
    return MultiStageRocket(
        name="Falcon-9-FT",
        stages=[stage_1, stage_2],
        payload_mass_kg=payload_kg,
        fairing_mass_kg=1750.0,
        fairing_jettison_alt_m=110000.0,
        fairing_diameter_m=5.2,
        nosecone_fineness=2.8
    )

def create_saturn_v(payload_kg=45000.0):
    """Creates a Saturn V Apollo moon rocket replica."""
    # S-IC First Stage: 5 F-1 engines
    f1 = RocketEngine(
        name="F-1",
        thrust_sl_N=6770000.0,
        thrust_vac_N=7770000.0,
        isp_sl_s=263.0,
        isp_vac_s=304.0,
        nozzle_exit_area_m2=4.8
    )
    s_ic = RocketStage(
        name="S-IC",
        dry_mass_kg=130000.0,
        propellant_mass_kg=2160000.0,
        diameter_m=10.1,
        length_m=42.0,
        engines=[f1],
        engine_count=5
    )
    
    # S-II Second Stage: 5 J-2 engines
    j2 = RocketEngine(
        name="J-2",
        thrust_sl_N=780000.0,
        thrust_vac_N=1023000.0,
        isp_sl_s=200.0,
        isp_vac_s=421.0,
        nozzle_exit_area_m2=2.1
    )
    s_ii = RocketStage(
        name="S-II",
        dry_mass_kg=38000.0,
        propellant_mass_kg=450000.0,
        diameter_m=10.1,
        length_m=24.8,
        engines=[j2],
        engine_count=5
    )
    
    # S-IVB Third Stage: 1 J-2 engine
    s_ivb = RocketStage(
        name="S-IVB",
        dry_mass_kg=13000.0,
        propellant_mass_kg=105000.0,
        diameter_m=6.6,
        length_m=17.8,
        engines=[j2],
        engine_count=1
    )
    
    return MultiStageRocket(
        name="Saturn-V",
        stages=[s_ic, s_ii, s_ivb],
        payload_mass_kg=payload_kg,
        fairing_mass_kg=3000.0,
        fairing_jettison_alt_m=115000.0,
        fairing_diameter_m=6.6,
        nosecone_fineness=2.2
    )

def create_custom_rocket(stages_config, payload_kg=2000.0, name="CustomRocket"):
    """
    Factory function allowing users to build and modify any arbitrary rocket.
    stages_config: list of dicts specifying dry_mass, prop_mass, diameter, length, and engine specs.
    """
    stages = []
    for i, cfg in enumerate(stages_config):
        eng = RocketEngine(
            name=cfg.get('engine_name', f"Engine_Stg{i+1}"),
            thrust_sl_N=cfg['thrust_sl_N'],
            thrust_vac_N=cfg['thrust_vac_N'],
            isp_sl_s=cfg['isp_sl_s'],
            isp_vac_s=cfg['isp_vac_s'],
            nozzle_exit_area_m2=cfg.get('nozzle_area_m2', 0.8)
        )
        stg = RocketStage(
            name=cfg.get('stage_name', f"Stage-{i+1}"),
            dry_mass_kg=cfg['dry_mass_kg'],
            propellant_mass_kg=cfg['propellant_mass_kg'],
            diameter_m=cfg['diameter_m'],
            length_m=cfg['length_m'],
            engines=[eng],
            engine_count=cfg.get('engine_count', 1)
        )
        stages.append(stg)
        
    return MultiStageRocket(
        name=name,
        stages=stages,
        payload_mass_kg=payload_kg,
        fairing_mass_kg=cfg.get('fairing_mass_kg', 1000.0),
        fairing_diameter_m=stages[0].diameter * 1.1 if stages else 3.0
    )

"""
Simulation Runner Engine.
Integrates all physics subsystems: US Standard Atmosphere, WGS-84 J2 Oblateness Gravity,
Aerodynamic Lift/Drag/Heating, Coordinate Reference Frames, Propulsion Staging,
and Flight Guidance to run full pad-to-orbit trajectories.
"""
import numpy as np

class FlightSimulation:
    def __init__(self, rocket, launch_lat_deg=28.57, launch_lon_deg=-80.65,
                 launch_azimuth_deg=90.0, target_orbit_alt_m=200000.0):
        self.rocket = rocket
        self.launch_lat = np.radians(launch_lat_deg)
        self.launch_lon = np.radians(launch_lon_deg)
        self.azimuth_deg = launch_azimuth_deg
        self.target_alt = target_orbit_alt_m
        
        # Instantiate Physics Engines
        self.atmosphere = USStandardAtmosphere1976()
        self.gravity = WGS84Gravity(enable_j2=True, enable_j3=True, enable_third_body=True)
        self.aero_model = AerodynamicsModel(
            reference_area_m2=rocket.reference_area,
            nosecone_fineness=rocket.nosecone_fineness,
            base_diameter_m=rocket.stages[0].diameter if rocket.stages else 3.7
        )
        self.srp = SolarRadiationPressure()
        self.guidance = AscentGuidance(
            target_orbit_altitude_m=target_orbit_alt_m,
            pitch_kick_alt_m=1200.0,
            pitch_kick_deg=1.8,
            launch_azimuth_deg=launch_azimuth_deg
        )
        self.integrator = AdaptiveRK45Integrator(tolerance=1e-5, min_step=0.01, max_step=0.5)

    def initialize_state(self):
        """Initializes position and velocity in ECI frame on the rotating Earth launch pad."""
        # Calculate Earth radius at launch latitude
        r_pad = self.gravity.sea_level_surface_radius(self.launch_lat)
        # Position in ECEF
        x_ecef = r_pad * np.cos(self.launch_lat) * np.cos(self.launch_lon)
        y_ecef = r_pad * np.cos(self.launch_lat) * np.sin(self.launch_lon)
        z_ecef = r_pad * np.sin(self.launch_lat)
        r_ecef = np.array([x_ecef, y_ecef, z_ecef])
        
        # At t=0, ECI aligns with ECEF
        r_eci = r_ecef.copy()
        
        # Velocity in ECI is solely due to Earth's rotation: v = omega x r
        v_eci = np.array([
            -CelestialMechanics.OMEGA_EARTH * y_ecef,
             CelestialMechanics.OMEGA_EARTH * x_ecef,
             0.0
        ])
        return r_eci, v_eci

    def run(self, max_time_s=600.0, dt_nominal=0.1):
        """
        Executes the full trajectory simulation.
        Returns telemetry dictionary containing flight history and milestones.
        """
        r_eci, v_eci = self.initialize_state()
        t = 0.0
        dt = dt_nominal
        
        # Telemetry logs
        telemetry = {
            'time': [],
            'altitude': [],
            'velocity_inertial': [],
            'airspeed_rel': [],
            'downrange_distance': [],
            'dynamic_pressure': [],
            'mach': [],
            'acceleration_g': [],
            'drag_force': [],
            'thrust_force': [],
            'mass': [],
            'heat_flux': [],
            'events': [],
            'stage_name': [],
            'r_eci': [],
            'v_eci': []
        }
        
        max_q = 0.0
        max_q_time = 0.0
        max_q_alt = 0.0
        karman_reached = False
        orbit_inserted = False
        
        step_count = 0
        while t < max_time_s:
            # Current geodetic coordinates
            r_ecef, v_ecef = CelestialMechanics.eci_to_ecef(r_eci, v_eci, t)
            lat, lon, alt = CelestialMechanics.ecef_to_geodetic(r_ecef)
            
            # Atmospheric conditions
            atm = self.atmosphere.properties(alt)
            rho = atm['density_kg_m3']
            p_amb = atm['pressure_Pa']
            a_sound = atm['speed_of_sound_m_s']
            
            # Atmospheric relative velocity (airspeed)
            v_atm = CelestialMechanics.atmospheric_velocity_eci(r_eci)
            v_rel = v_eci - v_atm
            v_rel_mag = np.linalg.norm(v_rel)
            mach = v_rel_mag / a_sound
            
            # Aerodynamic forces
            q = 0.5 * rho * (v_rel_mag ** 2)
            if q > max_q:
                max_q = q
                max_q_time = t
                max_q_alt = alt
                
            aero_res = self.aero_model.compute_forces(v_rel_mag, rho, mach, angle_of_attack_rad=0.0)
            f_drag_mag = aero_res['drag_force_N']
            heat_flux = aero_res['heat_flux_W_m2']
            
            # Drag vector opposes relative velocity
            if v_rel_mag > 1e-3:
                f_drag = -f_drag_mag * (v_rel / v_rel_mag)
            else:
                f_drag = np.zeros(3)
                
            # Rocket Propulsion & Staging
            thrust_mag, mass_burned, events = self.rocket.step(dt, p_amb, alt, q, throttle=1.0)
            for ev in events:
                telemetry['events'].append((t, ev))
                
            # Guidance: Thrust direction
            thrust_dir, guidance_mode = self.guidance.compute_thrust_direction(alt, v_rel, r_eci, t)
            f_thrust = thrust_mag * thrust_dir
            
            # Gravitational acceleration
            a_grav = self.gravity.acceleration(r_eci, time_sec=t)
            f_grav = self.rocket.total_mass * a_grav
            
            # Solar Radiation Pressure force in space
            a_srp = self.srp.acceleration(r_eci, self.rocket.total_mass, self.rocket.reference_area, t)
            f_srp = self.rocket.total_mass * a_srp
            
            # Total force and acceleration in ECI
            f_total = f_thrust + f_drag + f_grav + f_srp
            m_total = self.rocket.total_mass
            a_total = f_total / m_total
            
            # Apparent acceleration felt by onboard payload / structure (g-force)
            g_force = np.linalg.norm(f_thrust + f_drag) / (m_total * 9.80665)
            
            # Event detection: Kármán line (100 km)
            if not karman_reached and alt >= 100000.0:
                karman_reached = True
                telemetry['events'].append((t, "KARMAN_LINE_REACHED_100KM"))
                
            # Check orbital insertion condition
            v_inertial_mag = np.linalg.norm(v_eci)
            v_circ_local = np.sqrt(self.gravity.MU_EARTH / np.linalg.norm(r_eci))
            if not orbit_inserted and alt >= 180000.0 and v_inertial_mag >= v_circ_local * 0.98:
                orbit_inserted = True
                telemetry['events'].append((t, "ORBITAL_INSERTION_ACHIEVED"))
                
            # Record telemetry
            r_pad_initial = self.gravity.sea_level_surface_radius(self.launch_lat)
            downrange_m = r_pad_initial * np.arccos(np.clip(
                np.dot(r_eci / np.linalg.norm(r_eci), self.initialize_state()[0] / r_pad_initial),
                -1.0, 1.0
            ))
            
            telemetry['time'].append(t)
            telemetry['altitude'].append(alt)
            telemetry['velocity_inertial'].append(v_inertial_mag)
            telemetry['airspeed_rel'].append(v_rel_mag)
            telemetry['downrange_distance'].append(downrange_m)
            telemetry['dynamic_pressure'].append(q)
            telemetry['mach'].append(mach)
            telemetry['acceleration_g'].append(g_force)
            telemetry['drag_force'].append(f_drag_mag)
            telemetry['thrust_force'].append(thrust_mag)
            telemetry['mass'].append(m_total)
            telemetry['heat_flux'].append(heat_flux)
            telemetry['stage_name'].append(self.rocket.current_stage.name if self.rocket.current_stage else "Payload")
            telemetry['r_eci'].append(r_eci.copy())
            telemetry['v_eci'].append(v_eci.copy())
            
            # Stop condition if all stages exhausted and stable in orbit
            if self.rocket.current_stage is None and t > 400.0 and alt > 150000.0:
                break
                
            # Integrate state (Simple symplectic Verlet / RK integration for stability)
            v_eci = v_eci + a_total * dt
            r_eci = r_eci + v_eci * dt
            t += dt
            step_count += 1
            
        telemetry['summary'] = {
            'max_altitude_m': max(telemetry['altitude']),
            'final_velocity_m_s': telemetry['velocity_inertial'][-1],
            'max_q_Pa': max_q,
            'max_q_time_s': max_q_time,
            'max_q_alt_m': max_q_alt,
            'max_acceleration_g': max(telemetry['acceleration_g']),
            'karman_line_crossed': karman_reached,
            'orbital_insertion': orbit_inserted,
            'total_flight_time_s': t
        }
        return telemetry

"""
Publication-Quality Trajectory Plotting and Visualizer.
Generates comprehensive 6-panel flight trajectory analysis:
1. Altitude vs Downrange distance (Ascent flight profile)
2. Inertial & Relative Airspeed vs Time (with staging events)
3. Dynamic Pressure (Q) & Aerodynamic Drag vs Altitude (Max-Q highlight)
4. Acceleration G-Load vs Time
5. Vehicle Mass Depletion & Staging Profile
6. Aero-Thermal Stagnation Heat Flux vs Time
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_flight_profile(telemetry, output_png_path="trajectory_analysis.png"):
    t = np.array(telemetry['time'])
    alt_km = np.array(telemetry['altitude']) / 1000.0
    downrange_km = np.array(telemetry['downrange_distance']) / 1000.0
    v_inertial = np.array(telemetry['velocity_inertial'])
    v_rel = np.array(telemetry['airspeed_rel'])
    q_kpa = np.array(telemetry['dynamic_pressure']) / 1000.0
    g_force = np.array(telemetry['acceleration_g'])
    mass_tons = np.array(telemetry['mass']) / 1000.0
    heat_kw = np.array(telemetry['heat_flux']) / 1000.0
    
    fig, axs = plt.subplots(3, 2, figsize=(16, 14), dpi=150)
    fig.suptitle(f"Rocket Trajectory & Environmental Forces Simulation Analysis", fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Ascent Profile (Altitude vs Downrange)
    ax1 = axs[0, 0]
    ax1.plot(downrange_km, alt_km, color='#1f77b4', lw=2.5, label='Ascent Trajectory')
    ax1.axhline(100.0, color='#d62728', linestyle='--', lw=1.5, label='Kármán Line (100 km - Space)')
    ax1.set_xlabel('Downrange Distance (km)', fontweight='semibold')
    ax1.set_ylabel('Altitude (km)', fontweight='semibold')
    ax1.set_title('Altitude vs Downrange Profile', fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right')
    
    # Annotate Staging Events
    for t_ev, ev_name in telemetry['events']:
        # Find nearest index
        idx = (np.abs(t - t_ev)).argmin()
        if 'STAGE' in ev_name or 'BURNOUT' in ev_name or 'FAIRING' in ev_name:
            ax1.scatter([downrange_km[idx]], [alt_km[idx]], color='darkred', s=40, zorder=5)
            ax1.annotate(ev_name.replace('_', ' '), xy=(downrange_km[idx], alt_km[idx]),
                         xytext=(downrange_km[idx]+15, alt_km[idx]-5),
                         fontsize=8, arrowprops=dict(arrowstyle="->", color='black', lw=0.8))
            
    # 2. Velocity Profile
    ax2 = axs[0, 1]
    ax2.plot(t, v_inertial, color='#2ca02c', lw=2.0, label='Inertial Velocity (ECI)')
    ax2.plot(t, v_rel, color='#17becf', lw=1.8, linestyle='-.', label='Relative Airspeed (Atmosphere)')
    ax2.set_xlabel('Flight Time (s)', fontweight='semibold')
    ax2.set_ylabel('Velocity (m/s)', fontweight='semibold')
    ax2.set_title('Velocity Profiles vs Time', fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right')
    
    # 3. Dynamic Pressure and Max-Q
    ax3 = axs[1, 0]
    ax3.plot(t, q_kpa, color='#ff7f0e', lw=2.2, label='Dynamic Pressure Q (kPa)')
    sum_data = telemetry['summary']
    ax3.scatter([sum_data['max_q_time_s']], [sum_data['max_q_Pa'] / 1000.0], color='red', s=70, zorder=6)
    ax3.annotate(f"Max-Q: {sum_data['max_q_Pa']/1000.0:.1f} kPa\n(t={sum_data['max_q_time_s']:.1f}s, alt={sum_data['max_q_alt_m']/1000.0:.1f}km)",
                 xy=(sum_data['max_q_time_s'], sum_data['max_q_Pa']/1000.0),
                 xytext=(sum_data['max_q_time_s']+30, sum_data['max_q_Pa']/1000.0 - 5),
                 fontsize=9, fontweight='semibold', arrowprops=dict(arrowstyle="->", color='red', lw=1.2))
    ax3.set_xlabel('Flight Time (s)', fontweight='semibold')
    ax3.set_ylabel('Dynamic Pressure (kPa)', fontweight='semibold')
    ax3.set_title('Dynamic Pressure & Max-Q Aerodynamic Load', fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right')
    
    # 4. Acceleration (G-Load)
    ax4 = axs[1, 1]
    ax4.plot(t, g_force, color='#9467bd', lw=2.0, label='Axial + Aero G-Force')
    ax4.axhline(6.0, color='red', linestyle=':', lw=1.2, label='Typical Human/Payload Limit (6G)')
    ax4.set_xlabel('Flight Time (s)', fontweight='semibold')
    ax4.set_ylabel('Apparent Acceleration (g)', fontweight='semibold')
    ax4.set_title('Acceleration & Structural G-Load', fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper left')
    
    # 5. Mass Depletion and Staging
    ax5 = axs[2, 0]
    ax5.plot(t, mass_tons, color='#8c564b', lw=2.2, label='Vehicle Mass (t)')
    ax5.set_xlabel('Flight Time (s)', fontweight='semibold')
    ax5.set_ylabel('Total Mass (Metric Tons)', fontweight='semibold')
    ax5.set_title('Mass Consumption & Multi-Stage Jettison', fontweight='bold')
    ax5.grid(True, linestyle=':', alpha=0.6)
    ax5.legend(loc='upper right')
    
    # 6. Aero-Thermal Stagnation Heat Flux
    ax6 = axs[2, 1]
    ax6.plot(t, heat_kw, color='#e377c2', lw=2.0, label='Stagnation Heat Flux (kW/m²)')
    ax6.set_xlabel('Flight Time (s)', fontweight='semibold')
    ax6.set_ylabel('Convective Heat Flux (kW/m²)', fontweight='semibold')
    ax6.set_title('Aero-Thermodynamic Heating Rate', fontweight='bold')
    ax6.grid(True, linestyle=':', alpha=0.6)
    ax6.legend(loc='upper right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_png_path, dpi=180)
    plt.close()
    print(f"Trajectory analysis plot successfully saved to {output_png_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Complex Rocket Simulator")
    parser.add_argument("--preset", default="falcon9", choices=["falcon9", "saturn5", "custom"])
    parser.add_argument("--payload", type=float, default=5000.0)
    parser.add_argument("--target-alt", type=float, default=200000.0)
    parser.add_argument("--plot-output", default="trajectory_analysis.png")
    args = parser.parse_args()
    
    print("="*70)
    print("   COMPLEX MULTI-STAGE ROCKET FLIGHT SIMULATOR")
    print("="*70)
    if args.preset == "saturn5":
        rocket = create_saturn_v(payload_kg=args.payload)
    else:
        rocket = create_falcon_9_ft(payload_kg=args.payload)
        
    sim = FlightSimulation(rocket, target_orbit_alt_m=args.target_alt)
    telemetry = sim.run(max_time_s=600.0, dt_nominal=0.1)
    
    res = telemetry['summary']
    print(f"Max Altitude: {res['max_altitude_m']/1000.0:.2f} km")
    print(f"Final Velocity: {res['final_velocity_m_s']:.2f} m/s")
    print(f"Max-Q: {res['max_q_Pa']/1000.0:.2f} kPa at t={res['max_q_time_s']:.1f}s")
    print(f"Peak G-force: {res['max_acceleration_g']:.2f} G")
    print(f"Orbital Insertion: {'SUCCESS' if res['orbital_insertion'] else 'SUBORBITAL'}")
    
    plot_flight_profile(telemetry, output_png_path=args.plot_output)