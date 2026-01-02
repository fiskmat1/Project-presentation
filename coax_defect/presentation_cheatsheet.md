# 📋 POSTER PRESENTATION CHEAT SHEET
## "How High is the Field in That Cable Defect?"

---

## 🎯 ONE-SENTENCE PITCH
> "I simulated how tiny defects in high-voltage cable insulation change the electric field and increase breakdown risk."

---

## 📊 KEY NUMBERS

| Parameter | Value |
|-----------|-------|
| **Voltage** | 15 kV |
| **Inner conductor radius** | 2 mm |
| **Outer conductor radius** | 10 mm |
| **Insulation (XLPE)** | ε = 2.3 |
| **Defect radius** | 0.5 mm |
| **Grid resolution** | 720 × 720 |

| Defect Type | ε | Field Inside | Enhancement |
|-------------|---|--------------|-------------|
| **Air bubble** | 1 | ~1.45 kV/mm | **+5-10%** locally |
| **Water inclusion** | 80 | ~1.3 kV/mm | **+20-25%** in surrounding XLPE |
| **No defect (baseline)** | — | ~1.3 kV/mm | — |

**Global peak field:** ~6 kV/mm (at inner conductor) — changes <1% with defects

---

## 🔑 KEY MESSAGES

### Why it matters
- Defects = weak spots that concentrate electric stress
- Air bubbles → risk of **partial discharge** (sparks inside)
- Water inclusions → stress on **surrounding plastic**

### Main findings
1. **Global field barely changes** (<1%) because defect is small & far from electrodes
2. **Local field increases significantly** at defect location
3. Field enhancements stay **below typical PD thresholds** for mm-sized gaps
4. But repeated stress could still cause **long-term degradation**

---

## 🛠️ METHOD IN 30 SECONDS

1. **Model:** 2D cross-section of coaxial cable
2. **Physics:** Solve ∇·(ε∇V) = 0 (Laplace equation with varying permittivity)
3. **Solver:** Finite volume method with harmonic averaging at material interfaces
4. **Boundary conditions:** Inner conductor = 15 kV, Outer = 0 V, Far edges = no-flux
5. **Output:** E = −∇V (electric field from potential gradient)

---

## ❓ ANTICIPATED QUESTIONS

| Question | Answer |
|----------|--------|
| **Why finite volume?** | Conserves flux naturally; handles ε-jumps correctly without special treatment |
| **Why not finite element?** | FEM needs complex meshing; FVM is simpler for rectangular grids |
| **Why 2D?** | Captures essential physics; 3D would be computationally expensive |
| **What is permittivity ε?** | How much a material resists electric field. Air≈1, XLPE≈2.3, Water≈80 |
| **What is partial discharge?** | Small internal sparks that don't bridge electrodes but degrade insulation over time |
| **Why harmonic averaging?** | Preserves normal flux continuity: ε₁E₁ᵢ = ε₂E₂ᵢ at interfaces |
| **What's the analytical solution?** | E₀(r) = V₀ / [r·ln(Rₒᵤₜ/Rᵢₙ)] — only valid for homogeneous cable |
| **How did you validate?** | Compared with analytical solution for defect-free case; checked mesh convergence |

---

## 🔬 PHYSICS INTUITION

**Air bubble (ε=1 < ε_XLPE=2.3):**
- Field lines "prefer" lower-ε material → field concentrates **inside** bubble
- Risk: air ionizes → partial discharge

**Water inclusion (ε=80 >> ε_XLPE=2.3):**
- Field lines "avoid" high-ε material → field squeezed into **surrounding XLPE**
- Risk: stress concentration in solid insulation

---

## 📈 FUTURE WORK IDEAS
- Defects **closer to electrodes** (stronger fields)
- **Non-spherical** defects (needles, cracks)
- **Multiple defects** interacting
- **3D simulations** for realistic geometries

---

## 📚 KEY REFERENCES (if asked)
- [2] Jackson — Classical Electrodynamics (analytical coax formula)
- [3] Kuffel — High Voltage Engineering (PD thresholds, breakdown)
- [5] Eymard — Finite Volume Methods (harmonic averaging technique)

---

*Good luck with your presentation! 🍀*




