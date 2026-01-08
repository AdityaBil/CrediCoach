import { FileText, Brain, BarChart, Compass } from "lucide-react";

const steps = [
  {
    icon: FileText,
    title: "Submit Your Profile",
    description: "Enter your financial information including income, debts, and credit history.",
  },
  {
    icon: Brain,
    title: "AI Analysis",
    description: "Our ML models analyze 50+ factors using ensemble learning and XAI techniques.",
  },
  {
    icon: BarChart,
    title: "Get Clear Results",
    description: "Receive your approval probability with detailed, explainable reasons.",
  },
  {
    icon: Compass,
    title: "Follow Your Roadmap",
    description: "Get a personalized action plan to improve your credit and financial health.",
  },
];

const HowItWorks = () => {
  return (
    <section className="py-24 relative overflow-hidden" id="how-it-works">
      {/* Background decoration */}
      <div className="absolute inset-0">
        <div className="absolute top-1/2 left-0 right-0 h-px bg-gradient-to-r from-transparent via-border to-transparent" />
      </div>

      <div className="container relative z-10 px-4 md:px-6">
        <div className="text-center max-w-2xl mx-auto mb-16">
          <h2 className="text-3xl md:text-4xl font-bold font-display mb-4">
            How It Works
          </h2>
          <p className="text-muted-foreground text-lg">
            Four simple steps to understand your credit potential and improve it.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {steps.map((step, index) => (
            <div key={step.title} className="relative group">
              {/* Connector line */}
              {index < steps.length - 1 && (
                <div className="hidden lg:block absolute top-8 left-[60%] w-full h-px bg-gradient-to-r from-primary/50 to-transparent" />
              )}
              
              <div className="relative text-center">
                {/* Step number */}
                <div className="absolute -top-3 -right-3 w-8 h-8 rounded-full bg-accent text-accent-foreground text-sm font-bold flex items-center justify-center z-10">
                  {index + 1}
                </div>
                
                {/* Icon container */}
                <div className="w-16 h-16 mx-auto rounded-2xl gradient-bg flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300 glow-sm">
                  <step.icon className="w-8 h-8 text-primary-foreground" />
                </div>
                
                <h3 className="text-xl font-semibold font-display mb-3">{step.title}</h3>
                <p className="text-muted-foreground">{step.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
