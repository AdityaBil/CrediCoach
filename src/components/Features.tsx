import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, FileSearch, MessageSquareText, Compass, BarChart3, Shield } from "lucide-react";

const features = [
  {
    icon: Brain,
    title: "Smart Risk Assessment",
    description: "ML/DL models predict loan approval probability with 95%+ accuracy using ensemble learning.",
  },
  {
    icon: FileSearch,
    title: "Document Analysis",
    description: "Analyze income, employment history, credit utilization, debts, and payment behavior.",
  },
  {
    icon: MessageSquareText,
    title: "Explainable Decisions",
    description: "No more black-box rejections. Understand exactly why with SHAP-powered explanations.",
  },
  {
    icon: Compass,
    title: "Personalized Roadmap",
    description: "Get step-by-step guidance: reduce utilization, avoid inquiries, improve EMI consistency.",
  },
  {
    icon: BarChart3,
    title: "Credit Score Tracking",
    description: "Monitor your progress with visual dashboards and milestone celebrations.",
  },
  {
    icon: Shield,
    title: "Secure & Private",
    description: "Your financial data is encrypted and never shared. Bank-level security standards.",
  },
];

const Features = () => {
  return (
    <section className="py-24 relative" id="features">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-secondary/30 to-transparent" />
      
      <div className="container relative z-10 px-4 md:px-6">
        {/* Section header */}
        <div className="text-center max-w-2xl mx-auto mb-16">
          <h2 className="text-3xl md:text-4xl font-bold font-display mb-4">
            More Than a Loan Decision
          </h2>
          <p className="text-muted-foreground text-lg">
            CrediCoach AI combines cutting-edge machine learning with financial coaching to help you understand, improve, and succeed.
          </p>
        </div>

        {/* Features grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <Card 
              key={feature.title} 
              className="glass border-border/50 hover:glow-sm transition-all duration-300 group"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <CardHeader>
                <div className="w-12 h-12 rounded-lg gradient-bg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <feature.icon className="w-6 h-6 text-primary-foreground" />
                </div>
                <CardTitle className="font-display text-xl">{feature.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base">{feature.description}</CardDescription>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
