# Glioma White Matter Invasion × Subcortical Motor Mapping
## Conversation export for local Cursor upload

**Exported:** 2026-09-14  
**Purpose:** Continuity document for resuming this research thread on a laptop Cursor instance.

---

## How to use this file on your laptop

1. Download this file from the Cloud Agent artifacts / chat attachments.
2. Open Cursor Desktop on your laptop.
3. Put the file in your project (or open it alone).
4. Start a new Agent chat and attach / `@`-reference this file, e.g.:
   - “Continue from this conversation export. Focus on Microgap F and paranodal-first conduction failure.”

---

# PART 1 — Literature: glioma infiltration of white matter (last ~10 years)

Curated list ordered by approximate 2025 Journal Impact Factor (JCR 2026 release). Representative, not exhaustive.

## Conceptual domains
| Domain | Focus |
|--------|--------|
| Molecular / cellular | Why tumors migrate along axons/myelin |
| Histopathology | Scherer structures; peri-axonal vs peri-vascular |
| Neuroimaging | DTI/DKI/connectomics of occult infiltration |
| Clinical | Surgery, RT planning, prognosis |

## High-yield papers (by IF tier)

### Tier 1 (IF > ~15)
- Venkataramani et al., *Nature* (2025) — axonal injury as driver of GBM progression; early WM colonization
- Wang et al., *Nature Neuroscience* (2019) — NOTCH1–SOX2 loop; GSC invasion along WM tracts
- Nature Communications papers (2021, 2025, 2026) — WM as pro-differentiative niche; invasion phenotypes by cell state; local vs distal invasion programs (peri-axonal vs peri-vascular)

### Tier 2 (IF ~10–15)
- Liu et al., *Neuro-Oncology* — lesion–tract density / network framing of GBM
- Wei et al., *Brain* (2023) — structural connectome quantifies invasion & predicts survival
- Mauffrey et al., *Acta Neuropathologica* (2019) — RhoA–SPARC–Nogo-A axis in WM-invading GBM

### Tier 3–4 (imaging / clinical)
- Jang et al., *NeuroImage* (2022) — tractography in tumors; infiltration vs displacement vs disruption
- Cancers systematic reviews (2023) — WM tract alteration patterns in glioma
- FWE tractometry (*J Neuro-Oncology*, 2025) — local/remote WM alterations in diffuse gliomas
- Li et al., *J Neurosurg* (2016) — resecting peritumoral DTI abnormality & survival
- Yan et al. (2019) — multimodal MRI of infiltration beyond enhancement
- Shiban et al., *J Neurosurg* (2015) — subcortical MEP threshold vs CST distance (nonlinear)

### Pre-2016 classics still essential
- Price et al., *AJNR* (2006) — biopsy-validated DTI infiltration maps
- Provenzale et al., *Radiology* (2004) — peritumoral NAWM FA

---

# PART 2 — Literature: electrophysiology & physics of subcortical motor mapping / monopolar stimulation

## Clinical consensus / large series
- PIONEER / RANO Resect, *Lancet Oncology* (2025) — triple motor mapping recommendations
- Seidel/Raabe continuous dynamic mapping papers (*J Neurosurg*)
- Chang et al., *J Neurosurg* (2019) — 702 perirolandic cases
- Triple motor mapping (*J Neurosurg*, 2021)
- Shiban 2015 — 1 mA ≈ 1 mm is heuristic; nonlinear; cathodal often better subcortically

## Physics / modeling
- Gomez-Tames et al., *Phys Med Biol* (2018) — monopolar vs bipolar subcortical stimulation model
- Dali et al. (2019 preprint) — modeling subcortical WM stimulation; anisotropy critical
- McIntyre myelinated axon models (*J Neurophysiol*, etc.)
- Ranck / Rattay — activating function classics
- Szelényi et al., *Clin Neurophysiol* (2011) — monopolar cathodal > bipolar for subcortical CST

## Parameter conventions (typical)
| Parameter | Typical value | Notes |
|-----------|---------------|-------|
| Configuration | Monopolar | Higher sensitivity |
| Polarity (subcortical) | Often cathodal | Orientation-dependent |
| Frequency | 250–500 Hz short train | Taniguchi method |
| Pulse width | 0.3–0.7 ms | Shorter may be more linear with distance |
| Train | 4–5 pulses (sometimes To2) | Summation under anesthesia |
| Heuristic | ~1 mA ≈ 1 mm | Useful but pathology-fragile |

---

# PART 3 — D-wave vs I-wave (core physiology)

## Distinction
| | **D-wave (Direct)** | **I-waves (Indirect)** |
|---|---------------------|------------------------|
| Mechanism | Direct depolarization of CST **axons** | Trans-synaptic cortical activation of pyramidal neurons |
| Synapses needed? | No | Yes |
| Under anesthesia | Preserved | Reduced/abolished |
| Subcortical WM stimulation | **Dominant** | Essentially absent |
| Latency | Earliest volley | Later (I1, I2, I3…) |

## Propagation summary
- **Cortical stimulation:** can produce D and/or I waves depending on intensity, polarity, awake vs asleep, Taniguchi vs Penfield.
- **Subcortical stimulation:** mainly **direct axonal activation → D-waves → spinal motoneurons → muscle MEP**.
- Muscle MEP ≠ D-wave. MEP is downstream muscle output; D-wave is the CST volley itself (epidural).

## Why To5 works under anesthesia
High-frequency short trains create temporal summation of descending volleys (EPSPs) at spinal motoneurons so a muscle MEP appears even when synaptic cortical I-wave machinery is depressed.

---

# PART 4 — Conceptual cascade (UPDATED with paranodal-first staging)

## Core update
Do **not** model as:

```text
invasion → demyelination → conduction failure
```

Prefer:

```text
invasion / microenvironment stress
        │
        ▼
paranodal junction failure   ← often first electrophysiologically meaningful step
        │
        ▼
nodal channel dispersion / safety-factor drop
        │
        ▼
conduction slowing / desynchronization / intermittent block
        │
        ▼
myelin thinning / demyelination / axonal injury
        │
        ▼
permanent conduction failure / fiber loss
```

**Key claim:** paranodal uncoupling can cause clinically meaningful conduction change **before** frank myelin loss.

---

## Electrophysiological stage ladder (E0–E5)

| Stage | Structural state | Conduction phenotype | Likely mapping signature |
|------:|------------------|----------------------|--------------------------|
| **E0** | Intact paranode/myelin | Normal saltatory conduction | Stable thresholds; crisp MEPs |
| **E1** | Paranodal uncoupling, myelin mostly intact | Safety-factor drop; mild slowing; early dispersion | Mild threshold drift; MEP more fragile to anesthesia/ischemia |
| **E2** | Nodal/juxtaparanodal channel dispersion (Nav dilution, Kv invasion) | Unstable propagation; intermittent block | Fluctuating MEPs; threshold less distance-faithful |
| **E3** | Myelin damage / demyelination | Frank slowing + desynchrony | Higher thresholds / stronger trains needed; weaker MEPs |
| **E4** | Axonal injury / partial fiber loss | Reduced conducting population | Persistently small MEPs; high intensity needed |
| **E5** | Severe fiber loss / failure | No reliable orthodromic volley | MEP loss; “CST silent” despite residual streamlines |

Parallel track: edema / ECM remodeling / anisotropy loss alter **volume-conductor physics** (current shunting, field spread) independently of axon health.

---

## Full schematic

```text
[0] Invasion (peri-axonal / peri-vascular / NAWM)
              │
              ▼
[1] Paranodal uncoupling  ← PRIMARY early conduction lesion
    (Caspr–Contactin–NF155 failure; seal leak)
              │
              ▼
[2] Nodal channel dispersion
    (Nav dilution, Kv invasion, safety-factor collapse)
              │
              ├──────────────────────────────┐
              ▼                              ▼
[3] Myelin damage                    [5] Matrix/conductivity change
    (thinning → demyelination)           (edema, anisotropy loss)
              │                              │
              ▼                              │
[4] Axonal injury / fiber loss               │
              │                              │
              └──────────────┬───────────────┘
                             ▼
                    [6] Altered activation physics
                             │
                             ▼
                    [7] Degraded / dispersed D-wave
                             │
                             ▼
                    [8] MEP threshold / stability change
                             │
                             ▼
                    [9] Surgical interpretation (distance/safety)
```

---

## Microgaps (updated)

| Gap | Transition | Missing link |
|-----|------------|--------------|
| **A** | Invasion → paranodal uncoupling | Do glioma cells/proteases/hypoxia dismantle Caspr–NF155 before demyelination? |
| **B** | Paranodal uncoupling → channel reorganization | Kinetics of Nav/Kv domain remodeling in human CST |
| **C** | Channel failure → myelin damage | When is myelin loss consequence vs co-driver? |
| **D** | Structural injury → remaining conducting fraction | What fraction of CST remains orthodromically competent? |
| **E** | Injury ↔ volume-conductor change | Edema/infiltration alter current delivery independently of axon health |
| **F** *(refined former “Gap D”)* | Axonal stage → D-wave/MEP observables | Calibrated link from E1–E5 to intraoperative readouts |
| **G** | Mapping signal → surgical meaning | Why 1 mA≈1 mm fails across pathology + edema |

### Formal definition of Microgap F
The missing quantitative link between **stage of axon–glia junctional injury (especially paranodal uncoupling and nodal reorganization)** and **intraoperative CST readouts (D-wave morphology; MEP threshold/stability)**.

One sentence: *Mapping can change because paranodes fail before myelin is destroyed — and we currently cannot read that stage from threshold/MEP alone.*

---

# PART 5 — Train-of-five: what is recorded?

## Stimulation vs recording
- **Stimulation:** 5 discrete pulses exist (e.g. 250–500 Hz, ISI ~2–4 ms).
- **Standard muscle MEP recording:** usually **one summated MEP sweep per train**, not five scored MEPs.
- **Why:** individual pulses often do not fire spinal motoneurons under anesthesia; EPSPs summate; then one CMAP appears.

## Where per-pulse data can exist
1. **Epidural D-wave recording:** each pulse can yield its own descending volley (D1–D5).
2. **Research EMG analysis:** possible to inspect buildup across the train, but commercial IONM usually does not store 5 MEP amplitudes.
3. **Logged metadata:** intensity, polarity, pulse width, pulse count, ISI/frequency + resulting MEP waveform/amplitude/latency.

## Implication for E1–E2
To5 summation can **mask** early dispersion/low safety factor. Per-pulse D-waves would be more sensitive but are rarely recorded in glioma surgery.

---

# PART 6 — Granular information beyond threshold + twitch

Yes — much more can be derived if logged/analyzed deliberately.

## Stimulation parameters
| Parameter | Beyond binary use | Richer meaning |
|-----------|-------------------|----------------|
| **Intensity** | Distance heuristic | Recruitment curve slope ≈ recruitable fiber population; threshold drift ≈ excitability/edema/ischemia |
| **Polarity** | Convention | Cathode/anode asymmetry ≈ orientation + nodal initiation geometry |
| **Pulse width** | Fixed setting | Strength–duration / chronaxie ≈ membrane time constant / excitability (pathology-sensitive) |
| **Pulse number (To2 vs To5)** | Reliability under anesthesia | Summation demand ≈ safety factor; To2 fail / To5 succeed ⇒ impaired but present conduction |
| **ISI / frequency** | Fixed | Optimal ISI may shift with volley desynchrony (mostly research) |

## MEP features
| Feature | Richer meaning |
|---------|----------------|
| **Amplitude** | Relative amp, recruitment, trial-to-trial variability (instability phenotype) |
| **Latency** | Slowing (paranodal/demyelinating phenotypes) |
| **Waveform morphology** | Broadening/polyphasia ⇒ dispersion; amp↓ with stable shape ⇒ fewer axons |

## High-yield combinations
- Intensity + imaging distance → distance vs pathology
- Pulse width + intensity → excitability vs distance
- Polarity pair → orientation/node geometry
- To2 vs To5 → safety factor
- Amplitude variability → intermittent block
- Latency + waveform width → slowing/dispersion
- Recruitment curve → remaining CST population
- MEP vs D-wave → muscle/spinal filter vs CST volley truth

## Predicted signatures by stage
| Stage | Expected richer signature |
|------:|---------------------------|
| E0 | Low threshold, steep recruitment, To2 often enough, stable amp, narrow waveform |
| E1 | Mild threshold↑, ↑ variability, more pulse dependence, early broadening |
| E2 | Unstable thresholds, polarity asymmetry changes, To5 dependence↑, polyphasic MEPs |
| E3 | Clearer latency↑, broader waveforms, shallower recruitment |
| E4 | Right-shifted recruitment, low max amplitude |
| E5 | No MEP despite high intensity/full train |

## Practical richer OR package (minimal)
1. Threshold at standard PW/polarity/To5  
2. Recruitment: amp at thr, +1 mA, +2 mA  
3. To2 vs To5 at same intensity  
4. Latency + MEP duration/polyphasia  
5. Optional opposite-polarity threshold  
6. Optional second pulse-width threshold (crude chronaxie)

Highest-yield unused signals for paranodal-first model:
1. To2 vs To5 dependence  
2. Waveform broadening / variability  
3. Strength–duration or recruitment curves  
4. D-wave dispersion when obtainable  

---

# PART 7 — Open research priorities (from this thread)

1. Demonstrate **paranodal-first injury** in glioma WM (Caspr, NF155, nodal Nav/Kv).
2. Correlate E1–E2 markers with conduction (latency, dispersion, safety factor).
3. Co-register those stages with intraoperative D-wave/MEP.
4. Reinterpret distance heuristics conditioned on electrophysiological stage + edema/conductivity.
5. Close Microgap F with same-site: tissue class + microstructure + threshold/polarity/PW curve + D-wave + MEP + outcome.

---

# PART 8 — Resume prompts (paste into laptop Cursor)

**Prompt A — continue theory**
> Continue from this export. Refine Microgap F with a proposed measurement model linking paranodal uncoupling stage to To2/To5 dependence, MEP polyphasia, and D-wave dispersion.

**Prompt B — experiment design**
> Design a human intraoperative study to distinguish E1 (paranodal) from E3 (demyelination) using stimulation parameter sweeps and MEP/D-wave features. Include feasible OR constraints.

**Prompt C — figure**
> Turn the paranodal-first cascade into a figure-ready panel plan (A–F) with legends for a review or thesis.

**Prompt D — lit deep dive**
> Find and summarize evidence that paranodal/junctional disruption precedes demyelination in glioma or related WM injury models, and map it onto intraoperative mapping parameters.

---

# PART 9 — Session notes / continuity

- User goal: become SME on **glioma infiltration impact on white matter**, then connect to **subcortical mapping electrophysiology/physics**.
- Important conceptual turn: staged electrophysiology with **paranodal uncoupling preceding myelin damage** as likely early cause of conduction alteration.
- Cloud Agent cannot sync laptop mouse cursor; same chat can be resumed via Cursor Desktop **Agents Window** (same account) or by uploading this file into a new local chat.
- Related local workspace files observed at export time: `RF prediction of recurrence.R`, `RF spatial prediction of recurrence.R` (recurrence modeling; not yet integrated into this theory thread).

---

*End of export.*
