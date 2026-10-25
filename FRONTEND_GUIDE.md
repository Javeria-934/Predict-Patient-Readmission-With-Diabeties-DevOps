# Clinical Decision Support System: Deployment & User Guide

This guide details the deployable architecture, how to run the system, and how the Angular frontend interacts with the Flask machine learning backend.

---

## 1. Quick Start (Single-Command Launch)

The entire application (Flask API backend + compiled production Angular frontend) runs as a single unified service:

```powershell
python api/app.py
```

Once started:
- Open your browser at: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**
- The Flask service immediately serves the compiled Angular application and all prediction endpoints.

---

## 2. Optional Development Mode (Live Reloading)

If you wish to make live modifications to the Angular frontend source code:

1. **Terminal 1 (Backend API):**
   ```powershell
   python api/app.py
   ```
2. **Terminal 2 (Angular Development Server):**
   ```powershell
   cd frontend
   npm start
   ```
   Open **[http://localhost:4200](http://localhost:4200)**. The Angular development server proxies requests to `http://127.0.0.1:5000/api`.

3. **Recompiling the Production Bundle:**
   ```powershell
   cd frontend
   npm run build
   ```

---

## 3. System Architecture & Interactivity

```
PatientReadmissionDataset/
├── api/
│   └── app.py                  # Flask service (serves API & Angular static bundle)
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── app.ts          # Angular Signals state management & API client
│   │   │   ├── app.html        # Institutional clinical template (zero emojis)
│   │   │   └── app.scss        # Medical neutral theme & fill-slide button animation
│   │   └── styles.scss         # Global clinical design tokens
│   └── dist/frontend/browser/  # Pre-compiled deployable Angular production bundle
```

### Key Interactive Features:
1. **Dynamic Risk Stratification:** Modifying any clinical parameter (days in hospital, inpatient admissions, HbA1c result, insulin adjustment) triggers instant recalculation of 30-day readmission probability and risk tier.
2. **Archetype Presets:** Buttons with **fill slide animation** load clinically verified archetypes (*High Risk Frequent Inpatient*, *Moderate Risk Cardiac*, *Low Risk Routine Care*).
3. **Threshold Simulator:** Slider allows testing the clinical trade-off between the paper's 89% accuracy collapse (threshold 0.50) and 60% clinical readmission capture (lower threshold).
4. **Feature Impact Waterfall:** Real-time breakdown displaying which Electronic Health Record factors drive risk upward or act protectively.
5. **Audited Benchmark View:** Complete tabulated comparison against the published Cureus 2025 paper metrics across LightGBM, CatBoost, XGBoost, Random Forest, and Logistic Regression.
6. **Cohort Batch Screening:** Screens multi-patient encounters concurrently and visualizes an emergency triage breakdown.

---

## 4. Design Compliance Summary

- **Zero Emojis:** Completely eliminated emoji and non-clinical decorative characters.
- **No Hover Animations:** Removed all bouncing, floating, scaling, and pulsing animations.
- **No Vibrant Gradients:** Applied a restrained, high-contrast institutional medical aesthetic using slate and muted clinical indicators.
- **Fill Slide Button Animation:** All buttons feature a slide-in background curtain effect on hover/focus.
- **Streamlined Workflow:** All clinical parameters and model outputs are accessible within a unified view without extraneous screens.
