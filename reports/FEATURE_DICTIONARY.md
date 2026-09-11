# Master Geotechnical Feature Dictionary
**Project:** AI-Enabled Mine Subsidence Monitoring & Early Warning (SIH26025)  
**Selected Feature Count:** 29 Features | **Collinearity Threshold:** |r| < 0.95

---

## Selected Feature Indicators

| No | Feature Identifier | Domain | Description | Physical Unit | Safe Range | Alert Threshold |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: |
| 1 | `spatial_disp_gradient_max` | Spatial | Max deformation gradient between adjacent gallery bolts | mm/m | < 0.05 | >= 0.25 |
| 2 | `tilt_x_current` | Tilt | Current inclinometer angle along transverse tunnel axis | deg | -0.5 to 0.5 | >= 2.0 |
| 3 | `tilt_mag_std` | Tilt | Standard deviation of angular magnitude | deg | < 0.05 | >= 0.35 |
| 4 | `vib_std` | Vibration | Standard deviation of acceleration magnitude | g | < 0.03 | >= 0.20 |
| 5 | `vib_variance` | Vibration | Variance of vibration acceleration | g² | < 0.001 | >= 0.05 |
| 6 | `spatial_avg_neighbor_disp` | Spatial | Mean roof displacement across connected gallery nodes | mm | < 1.5 | >= 3.5 |
| 7 | `spatial_avg_neighbor_vib` | Spatial | Mean vibration RMS across connected gallery nodes | g | < 0.08 | >= 0.30 |
| 8 | `vib_mean` | Vibration | Mean raw vibration acceleration | g | < 0.06 | >= 0.25 |
| 9 | `tilt_trend_r2` | Tilt | Linear regression goodness-of-fit for directional tilt | [0, 1] | < 0.20 | >= 0.75 |
| 10 | `vib_min` | Vibration | Minimum acceleration sample in window | g | > 0.0 | - |
| 11 | `tilt_stability_index` | Tilt | Normalized strata stability penalty metric | [0, 1] | > 0.80 | < 0.40 |
| 12 | `spatial_num_abnormal_neighbors` | Spatial | Count of connected gallery nodes exceeding physical thresholds | count | 0 | >= 2 |
| 13 | `tilt_accel_max` | Tilt | Peak angular acceleration (second numerical derivative) | deg/s² | < 0.10 | >= 0.60 |
| 14 | `spatial_pct_abnormal_neighbors` | Spatial | Fraction of neighboring nodes exhibiting physical disturbance | [0, 1] | 0.0 | >= 0.50 |
| 15 | `tilt_rate_max` | Tilt | Peak angular velocity in window | deg/min | < 0.20 | >= 1.50 |
| 16 | `disp_accel_max` | Displacement | Peak roof convergence acceleration | mm/s² | < 0.05 | >= 0.40 |
| 17 | `vib_skewness` | Vibration | Asymmetry of vibration distribution | dimensionless | -0.5 to 0.5 | > 1.5 |
| 18 | `disp_rate_max` | Displacement | Maximum roof-to-floor convergence velocity | mm/min | < 0.05 | >= 0.80 |
| 19 | `vib_cov` | Vibration | Coefficient of variation (std / mean) | dimensionless | < 0.5 | > 1.2 |
| 20 | `disp_stability_index` | Displacement | Comprehensive displacement stability index | [0, 1] | > 0.80 | < 0.35 |
| 21 | `vib_crest_factor` | Vibration | Peak to RMS ratio (fracturing shock detector) | ratio | < 3.5 | >= 6.0 |
| 22 | `vib_kurtosis` | Vibration | Tail heaviness of vibration distribution | dimensionless | < 3.0 | >= 6.0 |
| 23 | `vib_spectral_centroid` | Vibration | Frequency center of gravity in FFT spectrum | Hz | < 0.15 | >= 0.35 |
| 24 | `vib_spectral_energy_low` | Vibration | Spectral energy in 0.0 - 0.1 Hz ground resonance band | energy | - | elevated |
| 25 | `vib_dominant_frequency` | Vibration | Highest energy spectral peak frequency | Hz | < 0.10 | >= 0.25 |
| 26 | `vib_spectral_energy_mid` | Vibration | Spectral energy in 0.1 - 0.3 Hz strata fracturing band | energy | - | elevated |
| 27 | `vib_spectral_entropy` | Vibration | Disorder of vibration spectrum (broadband vs harmonic) | [0, 1] | < 0.4 | >= 0.80 |
| 28 | `vib_spectral_energy_high`| Vibration | Spectral energy in > 0.3 Hz blast/drill acoustic band | energy | - | noise check |
| 29 | `vib_second_dominant_frequency`| Vibration | Secondary spectral harmonic frequency | Hz | - | - |
