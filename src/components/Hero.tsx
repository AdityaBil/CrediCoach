import { Button } from "@/components/ui/button";
import { ArrowRight, Brain, Shield, TrendingUp } from "lucide-react";

const Hero = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background gradient orbs */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-96 h-96 rounded-full bg-primary/20 blur-3xl animate-float" />
        <div className="absolute -bottom-40 -left-40 w-96 h-96 rounded-full bg-accent/20 blur-3xl animate-float" style={{ animationDelay: "2s" }} />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-gradient-to-br from-primary/5 to-accent/5 blur-3xl" />
      </div>

      {/* Grid pattern overlay */}
      <div 
        className="absolute inset-0 opacity-[0.02]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23000' fill-opacity='1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`
        }}
      />

      <div className="container relative z-10 px-4 md:px-6">
        <div className="flex flex-col items-center text-center space-y-8 max-w-4xl mx-auto">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass animate-fade-in">
            <Brain className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium text-muted-foreground">AI-Powered Credit Intelligence</span>
          </div>

          {/* Main heading */}
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold font-display tracking-tight animate-fade-in" style={{ animationDelay: "0.1s" }}>
            From Credit Scoring to{" "}
            <span className="gradient-text">Credit Coaching</span>
          </h1>

          {/* Subtitle */}
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl animate-fade-in" style={{ animationDelay: "0.2s" }}>
            CrediCoach AI transforms loan rejections into opportunities. Get personalized guidance to improve your credit score and unlock your financial potential.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 animate-fade-in" style={{ animationDelay: "0.3s" }}>
            <Button size="lg" className="gradient-bg text-primary-foreground hover:opacity-90 transition-opacity glow-sm group">
              Start Your Assessment
              <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
            </Button>
            <Button size="lg" variant="outline" className="glass hover:bg-card/90">
              Learn How It Works
            </Button>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-8 md:gap-16 pt-12 animate-fade-in" style={{ animationDelay: "0.4s" }}>
            <div className="text-center">
              <div className="text-3xl md:text-4xl font-bold font-display gradient-text">95%</div>
              <p className="text-sm text-muted-foreground mt-1">Prediction Accuracy</p>
            </div>
            <div className="text-center">
              <div className="text-3xl md:text-4xl font-bold font-display gradient-text">50K+</div>
              <p className="text-sm text-muted-foreground mt-1">Users Helped</p>
            </div>
            <div className="text-center">
              <div className="text-3xl md:text-4xl font-bold font-display gradient-text">+85</div>
              <p className="text-sm text-muted-foreground mt-1">Avg. Score Increase</p>
            </div>
          </div>
        </div>

        {/* Floating feature cards */}
        <div className="hidden lg:block absolute -right-4 top-1/4 animate-slide-in-right" style={{ animationDelay: "0.5s" }}>
          <div className="glass rounded-xl p-4 glow-sm">
            <Shield className="w-8 h-8 text-primary mb-2" />
            <p className="text-sm font-medium">Secure Analysis</p>
          </div>
        </div>
        <div className="hidden lg:block absolute -left-4 top-1/3 animate-slide-in-right" style={{ animationDelay: "0.6s" }}>
          <div className="glass rounded-xl p-4 glow-sm">
            <TrendingUp className="w-8 h-8 text-accent mb-2" />
            <p className="text-sm font-medium">Score Boost</p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
