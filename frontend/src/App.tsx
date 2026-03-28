import { useState } from "react";
import { 
  Gauge, Upload, FileText, Users, 
  User, Phone, Shield, CreditCard, BarChart3 
} from "lucide-react";

type Prediction = {
  churn_probability: number;
  prediction: string;
  risk_level: string;
};

export default function App() {
  const [tab, setTab] = useState<"single" | "batch">("single");
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [batchPreview, setBatchPreview] = useState<string[][]>([]);

  const [form, setForm] = useState({
    gender: "Male",
    SeniorCitizen: 0,
    Partner: "No",
    Dependents: "No",
    tenure: 12,
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "No",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "No",
    StreamingMovies: "No",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",
    MonthlyCharges: 70.35,
    TotalCharges: 843.0,
  });

  const loadSampleData = () => {
    setForm({
      gender: "Female",
      SeniorCitizen: 0,
      Partner: "Yes",
      Dependents: "No",
      tenure: 24,
      PhoneService: "Yes",
      MultipleLines: "Yes",
      InternetService: "Fiber optic",
      OnlineSecurity: "No",
      OnlineBackup: "Yes",
      DeviceProtection: "No",
      TechSupport: "Yes",
      StreamingTV: "Yes",
      StreamingMovies: "No",
      Contract: "One year",
      PaperlessBilling: "Yes",
      PaymentMethod: "Bank transfer (automatic)",
      MonthlyCharges: 89.5,
      TotalCharges: 2148.0,
    });
  };

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>) => {
    const { name, value, type } = e.target;
    setForm(prev => ({
      ...prev,
      [name]: type === "number" ? Number(value) : value
    }));
  };

  const handleSingle = async () => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data: Prediction = await res.json();
      setResult(data);
    } catch (err) {
      alert("Prediction failed. Is the backend running on port 8000?");
    }
    setLoading(false);
  };

  const handleBatch = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    const fd = new FormData();
    fd.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/predict_batch", {
        method: "POST",
        body: fd,
      });
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "churn_predictions.csv";
      a.click();

      const text = await blob.text();
      const rows = text.split("\n").slice(0, 6).map(row => row.split(","));
      setBatchPreview(rows);
    } catch (err) {
      alert("Batch failed. Make sure your CSV has all required columns.");
    }
    setLoading(false);
  };

  const inputClass = "w-full bg-zinc-800 border border-zinc-700 focus:border-emerald-400 focus:ring-2 focus:ring-emerald-400/30 rounded-2xl p-4 text-white transition outline-none";
  const labelClass = "block text-zinc-400 text-sm mb-1 font-medium";

  return (
    <div className="min-h-screen bg-zinc-950 text-white font-sans selection:bg-emerald-500/30">
      <div className="max-w-7xl mx-auto p-6 md:p-8">
        
        {/* Header */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-10">
          <div className="flex items-center gap-4">
            <div className="bg-emerald-500/10 p-3 rounded-2xl border border-emerald-500/20">
              <Gauge className="w-8 h-8 md:w-10 md:h-10 text-emerald-400" />
            </div>
            <div>
              <h1 className="text-4xl md:text-5xl font-bold tracking-tighter bg-gradient-to-br from-white to-emerald-200 bg-clip-text text-transparent">
                ChurnGuard
              </h1>
              <p className="text-zinc-400 text-sm mt-1">Telco Customer Churn Predictor</p>
            </div>
          </div>
          <span className="px-4 py-2 bg-emerald-900/30 border border-emerald-800/50 text-emerald-400 text-xs md:text-sm font-medium rounded-full flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            Stacked Ensemble • LGBM + XGB + CatBoost
          </span>
        </div>

        {/* Tabs */}
        <div className="flex bg-zinc-900/80 p-1 mb-8 w-fit rounded-full border border-zinc-800">
          <button
            onClick={() => setTab("single")}
            className={`px-6 py-2.5 text-sm md:text-base font-medium rounded-full transition-all duration-300 ${
              tab === "single"
                ? "bg-emerald-500 text-white shadow-lg shadow-emerald-500/20"
                : "text-zinc-400 hover:text-white"
            }`}
          >
            Single Customer
          </button>
          <button
            onClick={() => setTab("batch")}
            className={`px-6 py-2.5 text-sm md:text-base font-medium rounded-full transition-all duration-300 ${
              tab === "batch"
                ? "bg-emerald-500 text-white shadow-lg shadow-emerald-500/20"
                : "text-zinc-400 hover:text-white"
            }`}
          >
            Batch Upload
          </button>
        </div>

        {tab === "single" ? (
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
            
            {/* Form Card */}
            <div className="xl:col-span-8 bg-zinc-900/50 backdrop-blur-sm rounded-3xl p-6 md:p-8 border border-zinc-800 shadow-2xl">
              <div className="flex items-center justify-between mb-8">
                <h2 className="text-2xl md:text-3xl font-semibold">Customer Details</h2>
                <button
                  onClick={loadSampleData}
                  className="text-emerald-400 hover:text-emerald-300 text-sm font-medium flex items-center gap-2 transition bg-emerald-400/10 px-4 py-2 rounded-full"
                >
                  Load Sample Data
                </button>
              </div>

              <div className="space-y-10">
                {/* Demographics */}
                <div>
                  <div className="flex items-center gap-3 mb-5 border-b border-zinc-800 pb-3">
                    <User className="w-5 h-5 text-emerald-400" />
                    <p className="uppercase text-zinc-300 text-sm font-semibold tracking-widest">Demographics</p>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <label className={labelClass}>Gender</label>
                      <select name="gender" value={form.gender} onChange={handleChange} className={inputClass}>
                        <option>Male</option><option>Female</option>
                      </select>
                    </div>
                    <div>
                      <label className={labelClass}>Senior Citizen</label>
                      <select name="SeniorCitizen" value={form.SeniorCitizen} onChange={handleChange} className={inputClass}>
                        <option value={0}>No (0)</option><option value={1}>Yes (1)</option>
                      </select>
                    </div>
                    <div>
                      <label className={labelClass}>Partner</label>
                      <select name="Partner" value={form.Partner} onChange={handleChange} className={inputClass}>
                        <option>Yes</option><option>No</option>
                      </select>
                    </div>
                    <div>
                      <label className={labelClass}>Dependents</label>
                      <select name="Dependents" value={form.Dependents} onChange={handleChange} className={inputClass}>
                        <option>Yes</option><option>No</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Services */}
                <div>
                  <div className="flex items-center gap-3 mb-5 border-b border-zinc-800 pb-3">
                    <Phone className="w-5 h-5 text-emerald-400" />
                    <p className="uppercase text-zinc-300 text-sm font-semibold tracking-widest">Services</p>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                    <div>
                      <label className={labelClass}>Phone Service</label>
                      <select name="PhoneService" value={form.PhoneService} onChange={handleChange} className={inputClass}>
                        <option>Yes</option><option>No</option>
                      </select>
                    </div>
                    <div>
                      <label className={labelClass}>Multiple Lines</label>
                      <select name="MultipleLines" value={form.MultipleLines} onChange={handleChange} className={inputClass}>
                        <option>Yes</option><option>No</option><option>No phone service</option>
                      </select>
                    </div>
                    <div>
                      <label className={labelClass}>Internet Service</label>
                      <select name="InternetService" value={form.InternetService} onChange={handleChange} className={inputClass}>
                        <option>DSL</option><option>Fiber optic</option><option>No</option>
                      </select>
                    </div>
                    <div>
                      <label className={labelClass}>Online Security</label>
                      <select name="OnlineSecurity" value={form.OnlineSecurity} onChange={handleChange} className={inputClass}>
                        <option>Yes</option><option>No</option><option>No internet service</option>
                      </select>
                    </div>
                    <div>
                      <label className={labelClass}>Online Backup</label>
                      <select name="OnlineBackup" value={form.OnlineBackup} onChange={handleChange} className={inputClass}>
                        <option>Yes</option><option>No</option><option>No internet service</option>
                      </select>
                    </div>
                    <div>
                      <label className={labelClass}>Device Protection</label>
                      <select name="DeviceProtection" value={form.DeviceProtection} onChange={handleChange} className={inputClass}>
                        <option>Yes</option><option>No</option><option>No internet service</option>
                      </select>
                    </div>
                    <div>
                      <label className={labelClass}>Tech Support</label>
                      <select name="TechSupport" value={form.TechSupport} onChange={handleChange} className={inputClass}>
                        <option>Yes</option><option>No</option><option>No internet service</option>
                      </select>
                    </div>
                    <div>
                      <label className={labelClass}>Streaming TV</label>
                      <select name="StreamingTV" value={form.StreamingTV} onChange={handleChange} className={inputClass}>
                        <option>Yes</option><option>No</option><option>No internet service</option>
                      </select>
                    </div>
                    <div>
                      <label className={labelClass}>Streaming Movies</label>
                      <select name="StreamingMovies" value={form.StreamingMovies} onChange={handleChange} className={inputClass}>
                        <option>Yes</option><option>No</option><option>No internet service</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Billing */}
                <div>
                  <div className="flex items-center gap-3 mb-5 border-b border-zinc-800 pb-3">
                    <CreditCard className="w-5 h-5 text-emerald-400" />
                    <p className="uppercase text-zinc-300 text-sm font-semibold tracking-widest">Billing & Account</p>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                    <div>
                      <label className={labelClass}>Contract</label>
                      <select name="Contract" value={form.Contract} onChange={handleChange} className={inputClass}>
                        <option>Month-to-month</option><option>One year</option><option>Two year</option>
                      </select>
                    </div>
                    <div>
                      <label className={labelClass}>Paperless Billing</label>
                      <select name="PaperlessBilling" value={form.PaperlessBilling} onChange={handleChange} className={inputClass}>
                        <option>Yes</option><option>No</option>
                      </select>
                    </div>
                    <div>
                      <label className={labelClass}>Payment Method</label>
                      <select name="PaymentMethod" value={form.PaymentMethod} onChange={handleChange} className={inputClass}>
                        <option>Electronic check</option><option>Mailed check</option>
                        <option>Bank transfer (automatic)</option><option>Credit card (automatic)</option>
                      </select>
                    </div>
                    <div>
                      <label className={labelClass}>Tenure (Months)</label>
                      <input type="number" name="tenure" value={form.tenure} onChange={handleChange} className={inputClass} />
                    </div>
                    <div>
                      <label className={labelClass}>Monthly Charges ($)</label>
                      <input type="number" step="0.01" name="MonthlyCharges" value={form.MonthlyCharges} onChange={handleChange} className={inputClass} />
                    </div>
                    <div>
                      <label className={labelClass}>Total Charges ($)</label>
                      <input type="number" step="0.01" name="TotalCharges" value={form.TotalCharges} onChange={handleChange} className={inputClass} />
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Sticky Result Card Sidebar */}
            <div className="xl:col-span-4 flex flex-col gap-6">
              <div className="bg-zinc-900/50 backdrop-blur-sm rounded-3xl p-8 border border-zinc-800 shadow-2xl sticky top-8">
                <button
                  onClick={handleSingle}
                  disabled={loading}
                  className="w-full bg-emerald-500 hover:bg-emerald-400 disabled:bg-zinc-800 disabled:text-zinc-500 py-5 rounded-2xl text-lg font-bold transition-all active:scale-[0.98] shadow-xl shadow-emerald-500/20 mb-8"
                >
                  {loading ? "Analyzing Data..." : "Predict Churn Risk"}
                </button>

                <div className="flex flex-col items-center justify-center min-h-[300px] bg-zinc-950/50 rounded-2xl border border-zinc-800/50 p-6">
                  {result ? (
                    <>
                      <div className="relative w-48 h-48 mb-6">
                        <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
                          <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#27272a" strokeWidth="3" />
                          <path
                            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                            fill="none"
                            stroke={result.prediction === "Churn" ? "#ef4444" : "#10b981"}
                            strokeWidth="3"
                            strokeDasharray={`${result.churn_probability * 100}, 100`}
                            strokeLinecap="round"
                            className="transition-all duration-1000 ease-out"
                          />
                        </svg>
                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                          <div className="text-5xl font-bold text-white tracking-tighter">
                            {Math.round(result.churn_probability * 100)}%
                          </div>
                        </div>
                      </div>

                      <h3 className={`text-3xl font-bold tracking-tight mb-2 ${result.prediction === "Churn" ? "text-red-400" : "text-emerald-400"}`}>
                        {result.prediction}
                      </h3>
                      <div className={`px-4 py-1.5 rounded-full text-sm font-semibold flex items-center gap-2
                        ${result.risk_level === "High" ? "bg-red-500/20 text-red-400" : 
                          result.risk_level === "Medium" ? "bg-amber-500/20 text-amber-400" : 
                          "bg-emerald-500/20 text-emerald-400"}`}>
                        <Shield className="w-4 h-4" />
                        {result.risk_level} Risk
                      </div>
                    </>
                  ) : (
                    <div className="text-center text-zinc-500">
                      <Gauge className="w-16 h-16 mx-auto mb-4 opacity-20" />
                      <p className="text-lg font-medium">No Data Yet</p>
                      <p className="text-xs mt-1 max-w-[200px] mx-auto">Fill out the customer details and run the prediction.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

          </div>
        ) : (
          /* Batch Section */
          <div className="bg-zinc-900/50 backdrop-blur-sm rounded-3xl p-10 md:p-20 text-center border border-zinc-800 shadow-2xl">
            <div className="max-w-xl mx-auto">
              <div className="bg-emerald-500/10 w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-8">
                <Upload className="w-10 h-10 text-emerald-400" />
              </div>
              <h2 className="text-3xl md:text-4xl font-bold mb-4">Upload Customer List</h2>
              <p className="text-zinc-400 mb-10 text-lg">Upload your dataset to generate batch predictions. Ensure your CSV contains all 19 training columns.</p>

              <label className="cursor-pointer flex flex-col items-center justify-center border-2 border-dashed border-zinc-700 hover:border-emerald-500 rounded-3xl p-16 transition-all bg-zinc-950/50 hover:bg-zinc-900 group">
                <FileText className="w-12 h-12 mb-4 text-zinc-500 group-hover:text-emerald-400 transition-colors" />
                <p className="text-xl font-semibold text-white mb-2">Select CSV File</p>
                <p className="text-zinc-500 text-sm">or drag and drop here</p>
                <input type="file" accept=".csv" onChange={handleBatch} className="hidden" />
              </label>
            </div>

            {batchPreview.length > 0 && (
              <div className="mt-20 animate-in fade-in slide-in-from-bottom-4">
                <h3 className="flex items-center gap-2 text-xl font-semibold mb-6 text-left text-zinc-200">
                  <Users className="text-emerald-400" /> Data Preview
                </h3>
                <div className="overflow-x-auto rounded-2xl border border-zinc-800 bg-zinc-950 scrollbar-thin scrollbar-thumb-zinc-800 scrollbar-track-transparent">
                  <table className="w-full text-sm text-left">
                    <tbody className="divide-y divide-zinc-800/50">
                      {batchPreview.map((row, i) => (
                        <tr key={i} className="hover:bg-zinc-900/50 transition-colors">
                          {row.map((cell, j) => (
                            <td key={j} className="px-4 py-3 text-zinc-300 whitespace-nowrap">
                              {cell}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}