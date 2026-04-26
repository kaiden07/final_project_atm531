# ATM 531 Final Project — Prompt Plan of Attack

## Context

Goal is a 2050 Long Island (NYISO Zone K) renewables capacity-planning study: project hourly demand, size a solar+wind baseline, run a deficit analysis, optimize the mix, layer in battery storage, then cost it. The work happens in `/network/rit/lab/CUERG/kaiden/531_final_project` (already a git repo; `solar/` has Bear Mountain HW3 PV code; `wind/` has a 5 MW Riverhead small-wind study). Two pieces of input data are not yet on disk: an hourly NYISO Zone K demand shape, and an NSRDB TMY pixel for Long Island. The lab's own load projections (peak/baseline/cooling/EV at the **annual** level, 2020–2050, per zone) live at `/network/rit/lab/CUERG/kaiden/Profiles/baselines_ev_climate_df_final/<scenario>/` — these provide the 2050 magnitude; we still need a separate hourly NYISO shape to disaggregate.

This file is a **prompt plan**, not an implementation plan: each entry is a chat session you'd open in Claude Code, what it does, what it consumes, and the rough token budget.

---

## Decisions locked in (from clarifying questions)

- **Load source**: Lab's own Zone K projections in `Profiles/baselines_ev_climate_df_final/` — pick a scenario (e.g. `normal_adoption_rcp45cooler`), use the 2050 row of `peak_summer_load.csv` (column `K`) for capacity sizing and the corresponding `baseline_summer_load.csv` + `cooling_summer_load.csv` + `ev_summer_load.csv` to build the annual GWh total. Disaggregate to 8760 hours using a NYISO Zone K historical hourly shape from `Profiles/all_NYIS_load_data` or `ALL_NYISO_LOADS_NEW`.
- **Wind**: Onshore utility-scale, scale the existing Riverhead 10 m / 60 m TMY (`wind/data/tmy_hourly_averages.csv`) by swapping in a utility-scale power curve (GE 2.8-127 or similar) and scaling installed MW. Reuse `wind/wind_analysis.py` as a starting point.
- **Solar**: Pull NSRDB TMY for a Long Island pixel (~40.8 N, –73.0 W), reuse the irradiance + PV physics in `solar/pv_system.py`, swap the residential SunPower panel for a utility tracker module, scale to GW.

---

## Token budget & Pro-plan mapping

Pro plan = Sonnet 4.x by default; useful ceiling per 5-hr session ≈ **~200 K tokens of work** before you hit the cap (less for Opus 4.x — ~50–80 K). Numbers below assume Sonnet for execution and Opus only when needed for design. Token estimates are *project-side* (what gets read/written in this repo plus tool transcripts) and exclude system prompt overhead.

| # | Session | Est. tokens | Pro sessions (Sonnet) |
|---|---------|-------------|-----------------------|
| 0 | Data fetch — NSRDB + NYISO hourly shape | 30–60 K | 0.5 |
| 1 | Build 2050 hourly Zone K demand profile | 60–90 K | 0.5 |
| 2 | Solar utility-scale model for LI | 60–90 K | 0.5 |
| 3 | Wind utility-scale model for LI | 50–80 K | 0.5 |
| 4 | Baseline mix + first deficit analysis | 80–120 K | 1 |
| 5 | Optimize wind/solar mix | 80–120 K | 1 |
| 6 | Battery dispatch + second deficit | 100–150 K | 1 |
| 7 | Economics + rebates + LCOE | 70–110 K | 0.5–1 |
| 8 | Write-up plots + final report assembly | 60–100 K | 0.5 |
|   | **Total**                                | **~600–900 K** | **~6–8 Pro sessions** |

If you run on Opus 4.7 throughout, expect ~2× session count (12–16). If you batch sessions 2 & 3, or 4 & 5, you can shave one. Add ~1 buffer session for debugging — call it **8–10 Pro sessions** end-to-end.

---

## The prompt sequence

Each block below is a self-contained prompt to open a new Claude Code session with. They're ordered so each one's outputs feed the next. Run from `/network/rit/lab/CUERG/kaiden/531_final_project`.

### Session 0 — Data acquisition (one-time)

> "I'm starting an ATM 531 final project on 2050 Long Island renewables. Before any modeling I need two datasets on disk under `531_final_project/data/`:
> 1. NSRDB hourly TMY CSV(s) for a Long Island pixel near 40.80 N, –73.00 W (Brookhaven). Use the same NREL NSRDB endpoint our HW3 code expects (`solar/pv_system.py` reads `read_nsrdb`). Save as `data/nsrdb_li/<lat>_<lon>_tmy.csv`.
> 2. A NYISO Zone K hourly load CSV for at least one recent year (2019 or 2022 are good). Check `/network/rit/lab/CUERG/kaiden/Profiles/all_NYIS_load_data` and `ALL_NYISO_LOADS_NEW` first — if a Zone K hourly file already exists there, copy/symlink it; only download if missing. Save as `data/nyiso_zonek_<year>_hourly.csv`.
>
> Print a short summary of each file: rows, date range, columns, mean / max values."

### Session 1 — 2050 hourly demand profile (Task 1)

> "Build the 2050 Long Island hourly demand profile and write it to `data/li_2050_hourly_demand.csv` (8760 rows, columns: `datetime`, `mw`).
>
> Inputs:
> - Lab annual projections at `/network/rit/lab/CUERG/kaiden/Profiles/baselines_ev_climate_df_final/normal_adoption_rcp45cooler/` — column `K` is Long Island. Use the 2050 row of `baseline_summer_load.csv`, `cooling_summer_load.csv`, `ev_summer_load.csv` (these look like component peak MW; confirm by reading the parent notebook `Profiles/load_calcs_trends.ipynb` if needed).
> - The historical NYISO Zone K hourly shape from session 0.
>
> Method: take the historical hourly Zone K shape, scale it so (a) the annual peak hour matches the 2050 `peak_summer_load.csv` Zone K value for that scenario and (b) annual energy matches the implied 2050 baseline+cooling+EV totals. If the projection components are MW-peak rather than GWh-energy, document the assumption you made to convert.
>
> Output deliverables:
> - `data/li_2050_hourly_demand.csv`
> - `outputs/demand/annual_total_GWh.txt`
> - `outputs/demand/monthly_GWh.png`, `diurnal_average_MW.png`, `load_duration_curve.png`"

### Session 2 — Solar utility-scale model (Task 2 prep)

> "Adapt `solar/pv_system.py` into `solar/li_utility_solar.py` for a utility-scale single-axis-tracker farm on Long Island.
> - Read the NSRDB TMY from `data/nsrdb_li/`.
> - Replace the SunPower E19/318 panel with a current utility module (e.g. First Solar Series 6 or LONGi Hi-MO). Single-axis tracker (azimuth tracking N–S axis), DC/AC ratio 1.30, system losses ~14 %.
> - Output: a normalized hourly capacity factor series (per-MW-AC). Save to `data/solar_li_cf_hourly.csv` (8760 rows, columns: `datetime`, `cf`).
> - Print: annual CF, peak summer noon CF, and a `outputs/solar/diurnal_cf.png`.
> Reuse `solar/pv_system.py` functions wherever possible (`compute_hourly_pv`, isotropic model, NOCT) — don't rewrite the physics."

### Session 3 — Wind utility-scale model (Task 2 prep)

> "Adapt `wind/wind_analysis.py` into `wind/li_utility_wind.py` for utility-scale onshore wind on Long Island.
> - Reuse the Riverhead TMY at `wind/data/tmy_hourly_averages.csv` and the same shear/extrapolation pipeline.
> - Hub height 100 m. Replace the Energie PGE 20/35 power curve with a GE 2.8-127 (2.8 MW, 127 m rotor) curve — digitize from a public spec sheet or ask me to provide one.
> - Output: per-MW-AC normalized hourly CF series at `data/wind_li_cf_hourly.csv` (8760, columns: `datetime`, `cf`).
> - Print: annual CF, monthly CFs, `outputs/wind/diurnal_cf.png`."

### Session 4 — Baseline mix + first deficit analysis (Tasks 2 & 3)

> "Now combine. Open `analysis/baseline_deficit.py`.
> 1. Load the 2050 demand from `data/li_2050_hourly_demand.csv`. Print peak MW and annual GWh.
> 2. Pick a baseline mix: solar capacity ≈ peak / 2, wind capacity ≈ peak / 2 (we'll refine later — keep the choice in a `BASELINE = {...}` dict at top).
> 3. Hourly generation = solar_MW × solar_cf + wind_MW × wind_cf, no curtailment.
> 4. Hourly deficit = max(demand − generation, 0). Hourly surplus = max(generation − demand, 0).
> 5. Deliverables in `outputs/baseline/`:
>    - `summary.txt` — peak demand, total GWh, total deficit GWh, total surplus GWh, deficit hours, max instantaneous deficit MW.
>    - `diurnal_deficit.png` — year-average hour-of-day deficit (MW).
>    - `monthly_deficit.png`, `load_vs_gen_week.png` (a representative week)."

### Session 5 — Optimize the mix (Task 4 part 1)

> "Sweep the wind/solar capacity mix to minimize total annual deficit GWh. Open `analysis/optimize_mix.py`.
> - Hold total nameplate (solar + wind) at a few totals: 1.0×, 1.25×, 1.5×, 2.0× peak demand.
> - For each total, sweep solar share from 0 → 100 % in 5 % steps; record total deficit GWh, max deficit MW, and total surplus GWh.
> - Write `outputs/optimize/sweep.csv` and a heatmap `outputs/optimize/deficit_heatmap.png`.
> - Identify the (solar_MW, wind_MW) combination that minimizes deficit GWh subject to surplus ≤ 1.5× deficit (so we don't oversize wildly). Call this `OPTIMIZED_MIX`. Write it to `outputs/optimize/chosen_mix.txt`.
> - Conclude: is residual deficit small enough to ignore, or is battery storage required? Print a one-paragraph judgment."

### Session 6 — Battery dispatch + second deficit (Tasks 4 part 2 & 5)

> "Add a battery to the optimized mix. Open `analysis/battery_dispatch.py`.
> - Inputs: hourly demand, hourly generation from `OPTIMIZED_MIX`.
> - Battery: 4-hour Li-ion, round-trip efficiency 0.86, 90 % depth-of-discharge cap, no degradation. Power rating P_MW and energy rating 4×P_MW are the design variables.
> - Greedy dispatch: charge from surplus, discharge into deficit, respect SOC and power limits.
> - Sweep P_MW from 0 → max-deficit-MW in ~10 steps. Record residual deficit GWh, deficit hours, % of demand served.
> - Pick the smallest battery that gets residual deficit < 1 % of annual demand (or, if infeasible, the knee of the curve). Call it `CHOSEN_BATTERY`.
> - Deliverables in `outputs/battery/`:
>    - `sweep.csv`, `sweep_curve.png`
>    - `diurnal_deficit_with_battery.png` overlaid against the no-battery diurnal from session 4
>    - `soc_week.png` for a representative week."

### Session 7 — Economics (Task 6)

> "Cost out the `OPTIMIZED_MIX` + `CHOSEN_BATTERY` system. Open `analysis/economics.py`.
> - Use NREL ATB 2024 (or latest you can cite) midcase $/kW for utility solar tracker, onshore wind, and 4-hr Li-ion. Confirm CAPEX, fixed O&M, and a 25-year life with discount rate 6 %.
> - Apply: federal ITC 30 % (IRA), federal PTC for wind ($27/MWh) — pick whichever is more favorable per technology. NY-State NYSERDA NY-Sun and Bulk Storage incentives where they apply.
> - Compute LCOE per technology and blended LCOE for the system, plus annual avoided emissions vs the 2024 NYCW grid factor (~0.28 kg CO₂/kWh — already used in `wind/wind_analysis.py`).
> - Deliverables: `outputs/economics/summary.txt`, a stacked CAPEX bar `capex_breakdown.png`, an LCOE comparison bar `lcoe.png`."

### Session 8 — Final report assembly

> "Assemble the final deliverable. Read every `outputs/*/summary.txt` and key plot, then write `report.md` with sections:
> 1. 2050 LI demand projection (numbers + monthly_GWh.png)
> 2. Baseline mix and first deficit (diurnal_deficit.png)
> 3. Optimized mix (deficit_heatmap.png + chosen_mix.txt)
> 4. Battery sizing (sweep_curve.png + diurnal_deficit_with_battery.png)
> 5. Economics (capex_breakdown.png + lcoe.png + summary)
> 6. Conclusions and caveats.
> Render any LaTeX-friendly tables inline. Don't invent numbers — every figure must come from a file in `outputs/`."

---

## Critical files to read or modify

- `solar/pv_system.py` — reuse irradiance + Duffie-Beckman PV model in session 2 (do **not** rewrite).
- `wind/wind_analysis.py` — reuse shear extrapolation, WPD, power-curve interp, economics scaffold in sessions 3 & 7.
- `wind/data/tmy_hourly_averages.csv` — Riverhead 10 m/60 m hourly, source for wind CF.
- `/network/rit/lab/CUERG/kaiden/Profiles/baselines_ev_climate_df_final/<scenario>/` — 2050 zonal annual values; column `K`.
- `/network/rit/lab/CUERG/kaiden/Profiles/load_calcs_trends.ipynb` — read once in session 1 to confirm what each component CSV represents (peak vs energy).
- `/network/rit/lab/CUERG/kaiden/Profiles/all_NYIS_load_data` and `ALL_NYISO_LOADS_NEW` — check before downloading anything.

---

## Verification (end-to-end)

After session 8, sanity-check by:
1. `python solar/li_utility_solar.py` — annual CF should land in 0.18–0.22 for LI single-axis tracker.
2. `python wind/li_utility_wind.py` — annual CF should land in 0.30–0.40 for a 2.8 MW class onshore turbine at 100 m on LI; if the existing Riverhead study reports 0.47, that's small-turbine-specific, expect lower for utility class.
3. `python analysis/baseline_deficit.py` — annual demand GWh should be in the rough order of 35–50 TWh for 2050 LI (today's ~20 TWh × electrification growth).
4. Eyeball `outputs/baseline/diurnal_deficit.png` — deficit should peak in evening (post-sunset summer A/C + evening EV charging) and be near-zero midday.
5. `outputs/battery/diurnal_deficit_with_battery.png` should compress the evening peak materially vs the no-battery line.
6. LCOE should land around $30–60 /MWh for solar, $30–50 /MWh for onshore wind, $80–150 /MWh for storage — flag anything outside those bands.

---

## Risks / loose ends

- **Hourly NYISO Zone K shape may not be on disk** in either Profiles folder. Session 0 will discover this; if missing, the user will need to download from NYISO `OASIS` or `Markets & Operations → Energy Market & Operational Data → Real-Time Dispatch`. Add ~0.5 session if a manual download is needed.
- **The 2050 component CSVs** may be peak MW rather than annual GWh. Session 1 will confirm by reading `load_calcs_trends.ipynb`. If they're peak only, use NYISO load factor (~0.55) to back into annual GWh.
- **Power curve digitization** for the GE 2.8-127 in session 3 may need a manual paste from a manufacturer PDF. Plan to provide it inline.
