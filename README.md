# WDETS: ML-driven PLC controller with SCADA for wireless power transfer and power factor improvement

Simulation code and extracted data for the paper
"ML-Driven PLC Controller with SCADA Integration for Wireless Power Transfer and Power Factor Improvement:
A Datafied Electricity Transmission Framework"
by Md Zahin Hossain (Adamjee Cantonment College) and Mayeeshah Ferdousi (Viqarunnisa Noon School and College).

All results are **simulations**, not field measurements.

## Files

| File | What it is |
|---|---|
| `wdets_simulation.py` | Reproduces every table in Section VI of the paper (random seed 42) |
| `site_scada_5min.csv` | Extracted dataset: 8,640 SCADA-style records (30 days x 288 intervals): index, hour, weekday, P (kW), Q (kVAR), PF |
| `simulation_output_log.txt` | Raw printed output of the final run |

## Run

```
pip install numpy scikit-learn
python3 wdets_simulation.py
```

The script runs four parts:

1. **A** 85 kHz series-series WPT link: efficiency versus distance, detuning, ML capacitor retuning (loop L-B)
2. **B** Site power factor with 8 capacitor stages of 50 kVAR (loop L-D); writes `site_scada_5min.csv`
3. **C** PID output-voltage loop at 10 ms scan, ML gain scheduling, stability margins (loops L-A and L-C)
4. **D** Packet-style delivery test (Section VI-E of the paper)

## Main assumptions

- Coupling model: k(d) = 0.5 * (1 + (d/r)^2)^(-3/2), r = 0.15 m (approximation, geometry factor assumed)
- Output-voltage plant: second order, wn = 30 rad/s, zeta = 0.3, gain g = k_nom / k (assumed, not identified from hardware)
- Site load data is synthetic

## License

Add a license before publishing (for example MIT).
