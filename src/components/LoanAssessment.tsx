import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { CheckCircle2, XCircle, TrendingUp, AlertTriangle, ArrowRight } from "lucide-react";

interface AssessmentResult {
  approved: boolean;
  score: number;
  reasons: string[];
  improvements: string[];
  advice?: string;
  topFactors?: string[];
}

const LoanAssessment = () => {
  const [step, setStep] = useState(1);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AssessmentResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const [formData, setFormData] = useState({
    income: "",
    employmentType: "",
    creditScore: [650],
    debtToIncome: [30],
    creditHistory: "",
    loanAmount: "",
  });

  // Get API base URL (supports both dev and production)
  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5000';

  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    setError(null);
    
    try {
      // Map credit history to years for backend
      const creditHistoryYears = formData.creditHistory === "short" ? 1 : formData.creditHistory === "medium" ? 3 : 7;
      const monthlyIncome = parseFloat(formData.income) || 3000;
      const loanAmount = parseFloat(formData.loanAmount) || 5000;
      
      const payload = {
        monthly_income: monthlyIncome,
        credit_score: formData.creditScore[0],
        loan_amount: loanAmount,
        loan_duration: 36, // Default 36 months
        total_debt_payments: (monthlyIncome * formData.debtToIncome[0]) / 100,
        bankruptcy_history: 0,
        previous_defaults: 0,
        payment_history: creditHistoryYears * 12,
        length_credit_history: creditHistoryYears,
        checking_balance: 1000,
        savings_balance: 5000,
        total_assets: 50000,
        total_liabilities: 10000,
        credit_card_utilization: 0.3,
      };

      const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();

      // Parse backend response and create UI-friendly result
      const riskScore = data.risk_score || 50;
      const approved = data.approval === 1;
      const advice = data.advice || "";
      const topFactors = data.top_factors || [];

      // Calculate approval score (0-100)
      const approvalScore = Math.min(100, Math.max(0, 100 - (riskScore * 2)));

      const reasons: string[] = [];
      const improvements: string[] = [];

      if (formData.creditScore[0] < 700) {
        reasons.push("Credit score below preferred threshold (700+)");
        improvements.push("Pay all bills on time for the next 6 months to boost your score by ~50 points");
      }
      if (formData.debtToIncome[0] > 35) {
        reasons.push("Debt-to-income ratio exceeds 35%");
        improvements.push("Pay down existing debts to reduce DTI below 30%");
      }
      if (formData.creditHistory === "short") {
        reasons.push("Limited credit history length");
        improvements.push("Keep oldest credit accounts open and active");
      }
      if (formData.employmentType === "self-employed") {
        reasons.push("Self-employment income requires additional verification");
        improvements.push("Prepare 2+ years of tax returns and bank statements");
      }

      if (reasons.length === 0) {
        reasons.push("Strong financial profile across all metrics");
      }
      if (improvements.length === 0) {
        improvements.push("Maintain current financial habits", "Consider applying for a rewards credit card to further build history");
      }

      // Parse advice if it's a string with steps
      let improvementSteps = improvements;
      if (advice.includes("Suggested steps:")) {
        const stepsMatch = advice.match(/Suggested steps: (.+)/);
        if (stepsMatch) {
          improvementSteps = stepsMatch[1].split(" ").filter((s) => s.length > 0);
        }
      }

      setResult({
        approved,
        score: approvalScore,
        reasons,
        improvements: improvementSteps.slice(0, 3),
        advice,
        topFactors,
      });
      setStep(3);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze profile. Please try again.");
      console.error("Assessment error:", err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetAssessment = () => {
    setStep(1);
    setResult(null);
    setFormData({
      income: "",
      employmentType: "",
      creditScore: [650],
      debtToIncome: [30],
      creditHistory: "",
      loanAmount: "",
    });
  };

  return (
    <section className="py-24 relative" id="assessment">
      <div className="container px-4 md:px-6">
        <div className="text-center max-w-2xl mx-auto mb-12">
          <h2 className="text-3xl md:text-4xl font-bold font-display mb-4">
            Try Your Free Assessment
          </h2>
          <p className="text-muted-foreground text-lg">
            Get instant feedback on your loan eligibility and personalized improvement tips.
          </p>
        </div>

        <Card className="max-w-2xl mx-auto glass border-border/50">
          <CardHeader>
            <div className="flex items-center justify-between mb-4">
              <div className="flex gap-2">
                {[1, 2, 3].map((s) => (
                  <div
                    key={s}
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-colors ${
                      step >= s ? "gradient-bg text-primary-foreground" : "bg-secondary text-muted-foreground"
                    }`}
                  >
                    {s}
                  </div>
                ))}
              </div>
              <span className="text-sm text-muted-foreground">
                {step === 1 && "Financial Info"}
                {step === 2 && "Credit Details"}
                {step === 3 && "Results"}
              </span>
            </div>
            <CardTitle className="font-display">
              {step === 1 && "Tell us about your finances"}
              {step === 2 && "Credit & loan details"}
              {step === 3 && (result?.approved ? "Congratulations!" : "Let's improve together")}
            </CardTitle>
            <CardDescription>
              {step === 1 && "This information helps us analyze your eligibility."}
              {step === 2 && "Almost there! A few more details."}
              {step === 3 && "Here's your personalized assessment."}
            </CardDescription>
          </CardHeader>
          
          <CardContent className="space-y-6">
            {error && (
              <div className="p-4 bg-destructive/10 border border-destructive/30 rounded-lg text-destructive text-sm">
                <p className="font-medium">Error: {error}</p>
                <p className="text-xs mt-1">Make sure the backend is running on port 5000.</p>
              </div>
            )}

            {step === 1 && (
              <>
                <div className="space-y-2">
                  <Label htmlFor="income">Annual Income ($)</Label>
                  <Input
                    id="income"
                    type="number"
                    placeholder="75000"
                    value={formData.income}
                    onChange={(e) => setFormData({ ...formData, income: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="employment">Employment Type</Label>
                  <Select
                    value={formData.employmentType}
                    onValueChange={(value) => setFormData({ ...formData, employmentType: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select employment type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="full-time">Full-time Employee</SelectItem>
                      <SelectItem value="part-time">Part-time Employee</SelectItem>
                      <SelectItem value="self-employed">Self-Employed</SelectItem>
                      <SelectItem value="contractor">Contractor</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="loanAmount">Desired Loan Amount ($)</Label>
                  <Input
                    id="loanAmount"
                    type="number"
                    placeholder="25000"
                    value={formData.loanAmount}
                    onChange={(e) => setFormData({ ...formData, loanAmount: e.target.value })}
                  />
                </div>
                <Button 
                  className="w-full gradient-bg text-primary-foreground"
                  onClick={() => setStep(2)}
                  disabled={!formData.income || !formData.employmentType || !formData.loanAmount}
                >
                  Continue <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              </>
            )}

            {step === 2 && (
              <>
                <div className="space-y-4">
                  <Label>Credit Score: {formData.creditScore[0]}</Label>
                  <Slider
                    value={formData.creditScore}
                    onValueChange={(value) => setFormData({ ...formData, creditScore: value })}
                    min={300}
                    max={850}
                    step={10}
                    className="py-4"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Poor (300)</span>
                    <span>Fair (580)</span>
                    <span>Good (670)</span>
                    <span>Excellent (850)</span>
                  </div>
                </div>

                <div className="space-y-4">
                  <Label>Debt-to-Income Ratio: {formData.debtToIncome[0]}%</Label>
                  <Slider
                    value={formData.debtToIncome}
                    onValueChange={(value) => setFormData({ ...formData, debtToIncome: value })}
                    min={0}
                    max={60}
                    step={1}
                    className="py-4"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Excellent (0-20%)</span>
                    <span>Good (21-35%)</span>
                    <span>High (36%+)</span>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Credit History Length</Label>
                  <Select
                    value={formData.creditHistory}
                    onValueChange={(value) => setFormData({ ...formData, creditHistory: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select credit history length" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="short">Less than 2 years</SelectItem>
                      <SelectItem value="medium">2-5 years</SelectItem>
                      <SelectItem value="long">5+ years</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex gap-3">
                  <Button variant="outline" onClick={() => setStep(1)} className="flex-1">
                    Back
                  </Button>
                  <Button 
                    className="flex-1 gradient-bg text-primary-foreground"
                    onClick={handleAnalyze}
                    disabled={!formData.creditHistory}
                  >
                    Analyze My Profile
                  </Button>
                </div>
              </>
            )}

            {step === 3 && isAnalyzing && (
              <div className="py-12 text-center space-y-4">
                <div className="w-16 h-16 mx-auto rounded-full gradient-bg animate-pulse-glow flex items-center justify-center">
                  <TrendingUp className="w-8 h-8 text-primary-foreground animate-bounce" />
                </div>
                <p className="text-muted-foreground">Analyzing your financial profile...</p>
                <Progress value={66} className="max-w-xs mx-auto" />
              </div>
            )}

            {step === 3 && result && !isAnalyzing && (
              <div className="space-y-6">
                {/* Result header */}
                <div className={`p-6 rounded-xl text-center ${result.approved ? "bg-success/10 border border-success/30" : "bg-warning/10 border border-warning/30"}`}>
                  <div className={`w-16 h-16 mx-auto rounded-full flex items-center justify-center mb-4 ${result.approved ? "bg-success" : "bg-warning"}`}>
                    {result.approved ? (
                      <CheckCircle2 className="w-8 h-8 text-success-foreground" />
                    ) : (
                      <AlertTriangle className="w-8 h-8 text-warning-foreground" />
                    )}
                  </div>
                  <h3 className="text-xl font-bold font-display mb-2">
                    {result.approved ? "High Approval Likelihood!" : "Improvement Needed"}
                  </h3>
                  <p className="text-muted-foreground">
                    Approval Score: <span className="font-bold text-foreground">{Math.round(result.score)}%</span>
                  </p>
                </div>

                {/* Key factors */}
                <div className="space-y-3">
                  <h4 className="font-semibold flex items-center gap-2">
                    {result.approved ? <CheckCircle2 className="w-4 h-4 text-success" /> : <XCircle className="w-4 h-4 text-destructive" />}
                    Key Factors
                  </h4>
                  <ul className="space-y-2">
                    {result.reasons.map((reason, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                        <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground mt-2 shrink-0" />
                        {reason}
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Improvement plan */}
                <div className="space-y-3">
                  <h4 className="font-semibold flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-primary" />
                    Your Improvement Plan
                  </h4>
                  <ul className="space-y-2">
                    {result.improvements.map((improvement, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm bg-primary/5 p-3 rounded-lg border border-primary/20">
                        <span className="w-5 h-5 rounded-full gradient-bg text-primary-foreground text-xs flex items-center justify-center shrink-0">
                          {i + 1}
                        </span>
                        {improvement}
                      </li>
                    ))}
                  </ul>
                </div>

                <Button onClick={resetAssessment} variant="outline" className="w-full">
                  Start New Assessment
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </section>
  );
};

export default LoanAssessment;
