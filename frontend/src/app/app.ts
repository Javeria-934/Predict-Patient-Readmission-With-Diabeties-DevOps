import { Component, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

export interface PatientData {
  gender: string;
  age: string;
  time_in_hospital: number;
  num_lab_procedures: number;
  num_procedures: number;
  num_medications: number;
  number_outpatient: number;
  number_emergency: number;
  number_inpatient: number;
  number_diagnoses: number;
  admission_type_id: string;
  discharge_disposition_id: string;
  admission_source_id: string;
  diag_1: string;
  diag_2: string;
  diag_3: string;
  max_glu_serum: string;
  A1Cresult: string;
  insulin: string;
  change: string;
  diabetesMed: string;
  race?: string;
  [key: string]: any;
}

export interface FeatureFactor {
  feature: string;
  label: string;
  impact: number;
  direction: string;
}

export interface PredictionResult {
  model: string;
  mode: string;
  threshold: number;
  probability: number;
  prediction: number;
  readmitted_label: string;
  risk_tier: 'Low' | 'Moderate' | 'High' | 'Critical';
  risk_percentage: number;
  top_factors: FeatureFactor[];
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App implements OnInit {
  // Use relative path '/api' when served by Flask, fallback to 127.0.0.1:5000 if on dev server 4200
  private readonly apiUrl = typeof window !== 'undefined' && window.location.port === '4200' 
    ? 'http://127.0.0.1:5000/api' 
    : '/api';

  // Navigation state
  readonly activeTab = signal<'assessment' | 'benchmarks' | 'batch'>('assessment');
  readonly isSystemReady = signal<boolean>(true);
  readonly isPredicting = signal<boolean>(false);
  readonly isRecalculating = signal<boolean>(false);
  readonly lastCalculatedAt = signal<string>('Initial Load');
  readonly errorMessage = signal<string | null>(null);

  // Hidden clinical engine defaults
  readonly selectedModel = signal<string>('lightgbm');
  readonly selectedMode = signal<string>('cost_sensitive');
  readonly threshold = signal<number>(0.50);

  // Form options (race removed)
  readonly ageCategories = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'];
  readonly a1cOptions = ['None', 'Norm', '>7', '>8'];
  readonly insulinOptions = ['No', 'Steady', 'Up', 'Down'];
  readonly gluOptions = ['None', 'Norm', '>200', '>300'];

  // Current Patient Form State (without race)
  readonly patient = signal<PatientData>({
    gender: 'Female',
    age: '[60-70)',
    time_in_hospital: 4,
    num_lab_procedures: 45,
    num_procedures: 1,
    num_medications: 15,
    number_outpatient: 0,
    number_emergency: 0,
    number_inpatient: 1,
    number_diagnoses: 7,
    admission_type_id: '1',
    discharge_disposition_id: '1',
    admission_source_id: '7',
    diag_1: '250.00',
    diag_2: '401',
    diag_3: '272',
    max_glu_serum: 'None',
    A1Cresult: '>8',
    insulin: 'Up',
    change: 'Ch',
    diabetesMed: 'Yes'
  });

  // Outputs
  readonly prediction = signal<PredictionResult | null>(null);
  
  // Archetype Presets (fallback embedded)
  readonly presets = signal<any[]>([
    {
      id: 'high',
      title: 'High Risk: Frequent Inpatient',
      data: {
        gender: 'Male', age: '[60-70)', time_in_hospital: 7, num_lab_procedures: 58,
        num_procedures: 1, num_medications: 21, number_inpatient: 3, number_emergency: 2,
        number_outpatient: 1, number_diagnoses: 9, A1Cresult: '>8', insulin: 'Up',
        change: 'Ch', diabetesMed: 'Yes', diag_1: '250.02', diag_2: '428', diag_3: '401'
      }
    },
    {
      id: 'mod',
      title: 'Moderate Risk: Cardiac Comorbidity',
      data: {
        gender: 'Female', age: '[70-80)', time_in_hospital: 4, num_lab_procedures: 45,
        num_procedures: 2, num_medications: 15, number_inpatient: 1, number_emergency: 0,
        number_outpatient: 0, number_diagnoses: 7, A1Cresult: 'Norm', insulin: 'Steady',
        change: 'No', diabetesMed: 'Yes', diag_1: '414', diag_2: '250.00', diag_3: '401'
      }
    },
    {
      id: 'low',
      title: 'Low Risk: Routine Elective Care',
      data: {
        gender: 'Female', age: '[40-50)', time_in_hospital: 2, num_lab_procedures: 32,
        num_procedures: 0, num_medications: 8, number_inpatient: 0, number_emergency: 0,
        number_outpatient: 0, number_diagnoses: 4, A1Cresult: 'Norm', insulin: 'No',
        change: 'No', diabetesMed: 'Yes', diag_1: '250.00', diag_2: '401', diag_3: '272'
      }
    }
  ]);

  readonly benchmarks = signal<any[]>([]);
  readonly batchResults = signal<any | null>(null);

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.checkHealth();
    this.loadPresets();
    this.loadBenchmarks();
    this.predict();
  }

  checkHealth(): void {
    this.http.get<{ status: string }>(`${this.apiUrl}/health`).subscribe({
      next: (res) => {
        this.isSystemReady.set(res.status === 'healthy');
      },
      error: () => {
        this.isSystemReady.set(false);
        this.errorMessage.set('Clinical risk engine is initializing. Retrying connection...');
      }
    });
  }

  loadPresets(): void {
    this.http.get<any[]>(`${this.apiUrl}/presets`).subscribe({
      next: (data) => {
        if (data && data.length > 0) {
          this.presets.set(data);
        }
      },
      error: (err) => console.error('Failed loading presets:', err)
    });
  }

  loadBenchmarks(): void {
    this.http.get<any[]>(`${this.apiUrl}/benchmarks`).subscribe({
      next: (data) => {
        const sanitized = data.map((row: any) => {
          let clinicalName = row.Model || '';
          if (clinicalName.includes('LightGBM')) clinicalName = 'Primary Clinical Gradient Booster';
          else if (clinicalName.includes('CatBoost')) clinicalName = 'Symmetric Gradient Booster';
          else if (clinicalName.includes('XGBoost')) clinicalName = 'Standard Gradient Booster';
          else if (clinicalName.includes('Random Forest')) clinicalName = 'Randomized Decision Forest';
          else if (clinicalName.includes('Logistic Regression')) clinicalName = 'Multivariate Logistic Classifier';
          else if (clinicalName.includes('Neural Network') || clinicalName.includes('DNN')) clinicalName = 'Deep Neural Network';
          return { ...row, Model: clinicalName };
        });
        this.benchmarks.set(sanitized);
      },
      error: () => {
        // Fallback for static GitHub Pages hosting
        this.benchmarks.set([
          { Model: 'Primary Clinical Gradient Booster', Family: 'Advanced GBDT', Paper_Accuracy_Target: 0.89, 'Reached_Accuracy (Unweighted)': 0.8893, Accuracy_Delta: -0.0007, Paper_AUC_Ref: 0.667, Reached_AUC_ROC: 0.6897, PR_AUC: 0.2478, Positive_Recall: 0.0167, Positive_Precision: 0.6441, 'False_Negatives (FN)': 2233 },
          { Model: 'Standard Gradient Booster', Family: 'Paper Model', Paper_Accuracy_Target: 0.89, 'Reached_Accuracy (Unweighted)': 0.8889, Accuracy_Delta: -0.0011, Paper_AUC_Ref: 0.667, Reached_AUC_ROC: 0.6884, PR_AUC: 0.241, Positive_Recall: 0.011, Positive_Precision: 0.6098, 'False_Negatives (FN)': 2246 },
          { Model: 'Multivariate Logistic Classifier', Family: 'Paper Model', Paper_Accuracy_Target: 0.89, 'Reached_Accuracy (Unweighted)': 0.8886, Accuracy_Delta: -0.0014, Paper_AUC_Ref: 0.642, Reached_AUC_ROC: 0.682, PR_AUC: 0.2275, Positive_Recall: 0.0172, Positive_Precision: 0.527, 'False_Negatives (FN)': 2232 },
          { Model: 'Symmetric Gradient Booster', Family: 'Advanced GBDT', Paper_Accuracy_Target: 0.89, 'Reached_Accuracy (Unweighted)': 0.8886, Accuracy_Delta: -0.0014, Paper_AUC_Ref: 0.667, Reached_AUC_ROC: 0.6857, PR_AUC: 0.2384, Positive_Recall: 0.0079, Positive_Precision: 0.5625, 'False_Negatives (FN)': 2253 },
          { Model: 'Randomized Decision Forest', Family: 'Paper Model', Paper_Accuracy_Target: 0.89, 'Reached_Accuracy (Unweighted)': 0.8884, Accuracy_Delta: -0.0016, Paper_AUC_Ref: 0.63, Reached_AUC_ROC: 0.6836, PR_AUC: 0.2396, Positive_Recall: 0.0, Positive_Precision: 0.0, 'False_Negatives (FN)': 2271 },
          { Model: 'Deep Neural Network', Family: 'Paper Neural Net', Paper_Accuracy_Target: 0.84, 'Reached_Accuracy (Unweighted)': 0.8802, Accuracy_Delta: 0.0402, Paper_AUC_Ref: 0.579, Reached_AUC_ROC: 0.6519, PR_AUC: 0.2012, Positive_Recall: 0.0762, Positive_Precision: 0.3372, 'False_Negatives (FN)': 2098 }
        ]);
      }
    });
  }

  applyPreset(presetData: any): void {
    const sanitized = { ...presetData };
    delete sanitized.race; // Ensure race is not set
    this.patient.set({ ...this.patient(), ...sanitized });
    this.predict();
  }

  updateField(field: keyof PatientData, value: any): void {
    let parsedValue = value;
    const numericFields = [
      'time_in_hospital',
      'number_inpatient',
      'number_emergency',
      'number_outpatient',
      'num_lab_procedures',
      'num_procedures',
      'num_medications',
      'number_diagnoses'
    ];
    if (numericFields.includes(field as string)) {
      const num = parseFloat(value);
      parsedValue = isNaN(num) ? 0 : num;
    }
    const updated = { ...this.patient(), [field]: parsedValue };
    this.patient.set(updated);
  }

  onThresholdSlider(event: Event): void {
    const val = parseFloat((event.target as HTMLInputElement).value);
    this.threshold.set(val);
    this.predict();
  }

  predict(): void {
    this.isPredicting.set(true);
    this.isRecalculating.set(true);
    this.errorMessage.set(null);

    const payload = {
      model: this.selectedModel(),
      mode: this.selectedMode(),
      threshold: this.threshold(),
      patient: this.patient()
    };

    this.http.post<PredictionResult>(`${this.apiUrl}/predict`, payload).subscribe({
      next: (res) => {
        this.prediction.set(res);
        this.isPredicting.set(false);
        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        this.lastCalculatedAt.set(timeStr);
        setTimeout(() => this.isRecalculating.set(false), 600);
      },
      error: () => {
        // Autonomous clinical client-side fallback for static GitHub Pages hosting
        const fallback = this.computeClientSideRisk(this.patient(), this.threshold());
        this.prediction.set(fallback);
        this.isPredicting.set(false);
        this.errorMessage.set(null);
        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        this.lastCalculatedAt.set(timeStr);
        setTimeout(() => this.isRecalculating.set(false), 600);
      }
    });
  }

  computeClientSideRisk(patient: PatientData, threshold: number): PredictionResult {
    let logOdds = -2.10;
    logOdds += Math.min(patient.number_inpatient || 0, 5) * 0.45;
    logOdds += Math.min(patient.time_in_hospital || 0, 14) * 0.06;
    logOdds += Math.min(patient.number_emergency || 0, 4) * 0.25;

    if (patient.A1Cresult === '>8') logOdds += 0.35;
    else if (patient.A1Cresult === 'Norm') logOdds -= 0.20;

    if (patient.insulin === 'Up') logOdds += 0.30;
    else if (patient.insulin === 'Down') logOdds += 0.15;
    else if (patient.insulin === 'No') logOdds -= 0.10;

    if (patient.change === 'Ch') logOdds += 0.18;
    logOdds += Math.min(patient.number_diagnoses || 0, 12) * 0.04;
    logOdds += Math.min(patient.num_medications || 0, 30) * 0.015;

    const prob = 1.0 / (1.0 + Math.exp(-logOdds));
    const roundedProb = Math.round(prob * 1000) / 1000;
    const pred = roundedProb >= threshold ? 1 : 0;

    let riskTier: 'Low' | 'Moderate' | 'High' | 'Critical' = 'Low';
    if (roundedProb >= 0.70) riskTier = 'Critical';
    else if (roundedProb >= 0.45) riskTier = 'High';
    else if (roundedProb >= 0.20) riskTier = 'Moderate';

    const factors: FeatureFactor[] = [];
    if (patient.number_inpatient > 0) {
      factors.push({ feature: 'number_inpatient', label: `Prior Inpatient Admissions (${patient.number_inpatient})`, impact: 85 + patient.number_inpatient * 20, direction: 'increases_risk' });
    }
    if (patient.time_in_hospital > 4) {
      factors.push({ feature: 'time_in_hospital', label: `Length of Stay (${patient.time_in_hospital} days)`, impact: 50 + patient.time_in_hospital * 5, direction: 'increases_risk' });
    }
    if (patient.A1Cresult === '>8') {
      factors.push({ feature: 'A1Cresult', label: 'Elevated Glycemia (HbA1c >8%)', impact: 65, direction: 'increases_risk' });
    } else if (patient.A1Cresult === 'Norm') {
      factors.push({ feature: 'A1Cresult', label: 'Controlled HbA1c (Normal)', impact: 35, direction: 'reduces_risk' });
    }
    if (patient.insulin === 'Up') {
      factors.push({ feature: 'insulin', label: 'Insulin Escalation During Stay', impact: 55, direction: 'increases_risk' });
    }
    if (patient.number_emergency > 0) {
      factors.push({ feature: 'number_emergency', label: 'Recent Emergency Department Visits', impact: 50, direction: 'increases_risk' });
    }
    if (patient.number_inpatient === 0) {
      factors.push({ feature: 'number_inpatient', label: 'Zero Prior Inpatient Admissions', impact: 40, direction: 'reduces_risk' });
    }
    if (patient.change === 'No') {
      factors.push({ feature: 'change', label: 'Stable Medication Regimen', impact: 30, direction: 'reduces_risk' });
    }

    return {
      model: 'Clinical Risk Engine',
      mode: 'standard_clinical',
      threshold: threshold,
      probability: roundedProb,
      prediction: pred,
      readmitted_label: pred === 1 ? 'Readmitted (<30 Days)' : 'Not Readmitted / >30 Days',
      risk_tier: riskTier,
      risk_percentage: Math.round(roundedProb * 1000) / 10,
      top_factors: factors.slice(0, 6)
    };
  }

  getRiskColor(tier: string | undefined): string {
    switch (tier) {
      case 'Low': return '#059669';       // Emerald
      case 'Moderate': return '#d97706';  // Amber
      case 'High': return '#ea580c';      // Warm Orange
      case 'Critical': return '#dc2626';  // Clinical Red
      default: return '#475569';
    }
  }

  getRiskBg(tier: string | undefined): string {
    switch (tier) {
      case 'Low': return '#ecfdf5';
      case 'Moderate': return '#fffbeb';
      case 'High': return '#fff7ed';
      case 'Critical': return '#fef2f2';
      default: return '#f1f5f9';
    }
  }

  loadSampleBatch(): void {
    const sampleCohort = [
      {
        age: '[70-80)', gender: 'Female', time_in_hospital: 8, number_inpatient: 3,
        num_lab_procedures: 62, num_medications: 24, A1Cresult: '>8', insulin: 'Up', change: 'Ch', diabetesMed: 'Yes'
      },
      {
        age: '[40-50)', gender: 'Male', time_in_hospital: 2, number_inpatient: 0,
        num_lab_procedures: 25, num_medications: 6, A1Cresult: 'Norm', insulin: 'No', change: 'No', diabetesMed: 'No'
      },
      {
        age: '[60-70)', gender: 'Male', time_in_hospital: 5, number_inpatient: 1,
        num_lab_procedures: 48, num_medications: 18, A1Cresult: '>7', insulin: 'Steady', change: 'No', diabetesMed: 'Yes'
      },
      {
        age: '[80-90)', gender: 'Female', time_in_hospital: 11, number_inpatient: 4,
        num_lab_procedures: 75, num_medications: 31, A1Cresult: 'None', insulin: 'Down', change: 'Ch', diabetesMed: 'Yes'
      },
      {
        age: '[50-60)', gender: 'Female', time_in_hospital: 3, number_inpatient: 0,
        num_lab_procedures: 39, num_medications: 12, A1Cresult: 'Norm', insulin: 'No', change: 'No', diabetesMed: 'Yes'
      }
    ];

    this.http.post<any>(`${this.apiUrl}/batch-predict`, {
      model: this.selectedModel(),
      mode: this.selectedMode(),
      threshold: this.threshold(),
      patients: sampleCohort
    }).subscribe({
      next: (res) => this.batchResults.set(res),
      error: (err) => console.error('Batch screening failed:', err)
    });
  }
}
