import React from "react";
import { useNavigate } from "react-router-dom";
import "../styles/HomePage.css";

// Demo layout image can be a placeholder or actual mockup
import demoImage from "../assets/demo-layout.png";

const HomePage = () => {
  const navigate = useNavigate();

  return (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <h1>Hotel Design AI</h1>
          <p className="tagline">
            Revolutionize hotel layout design using artificial intelligence and
            architectural optimization algorithms
          </p>

          <div className="hero-buttons">
            <button
              className="btn-hero-primary"
              onClick={() => navigate("/configure")}
            >
              Start Designing
            </button>
            <button
              className="btn-hero-secondary"
              onClick={() => navigate("/layouts")}
            >
              Browse Layouts
            </button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="features-container">
          <div className="section-title">
            <h2>Key Features</h2>
            <p>
              Our AI-powered system helps you optimize hotel layouts based on
              architectural principles and efficient space utilization
            </p>
          </div>

          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">
                <i className="feature-icon-ai">AI</i>
              </div>
              <h3 className="feature-title">AI-Driven Design</h3>
              <p className="feature-description">
                Leverage advanced AI algorithms to generate optimized hotel
                layouts based on your requirements and industry best practices.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">
                <i className="feature-icon-edit">⚙️</i>
              </div>
              <h3 className="feature-title">Interactive Editing</h3>
              <p className="feature-description">
                Fine-tune generated layouts with our intuitive drag-and-drop
                interface to match your specific design vision.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">
                <i className="feature-icon-analytics">📊</i>
              </div>
              <h3 className="feature-title">Space Optimization</h3>
              <p className="feature-description">
                Maximize space efficiency and flow with architectural
                constraints like adjacency requirements and circulation
                patterns.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">
                <i className="feature-icon-3d">3D</i>
              </div>
              <h3 className="feature-title">3D Visualization</h3>
              <p className="feature-description">
                View your designs in interactive 3D to better understand spatial
                relationships and overall architecture.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">
                <i className="feature-icon-export">⬇️</i>
              </div>
              <h3 className="feature-title">Export Options</h3>
              <p className="feature-description">
                Export layouts in multiple formats including JSON, CSV, and
                visualization-ready formats for further processing.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">
                <i className="feature-icon-learn">🧠</i>
              </div>
              <h3 className="feature-title">Learning Algorithm</h3>
              <p className="feature-description">
                Our system learns from your feedback to continuously improve
                generation results and adapt to your preferences.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section className="demo-section">
        <div className="demo-container">
          <div className="section-title">
            <h2>See It In Action</h2>
            <p>
              Experience the power of AI-driven hotel design with our
              interactive demo
            </p>
          </div>

          <div className="demo-content">
            <img
              src={demoImage}
              alt="Hotel Design AI Demo"
              className="demo-image"
              onError={(e) => {
                e.target.onerror = null;
                e.target.src =
                  "https://via.placeholder.com/800x500?text=Hotel+Layout+Demo";
              }}
            />

            <button
              className="btn-demo"
              onClick={() => navigate("/interactive/sample/sample")}
            >
              Try Interactive Demo
            </button>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="cta-container">
          <h2 className="cta-title">
            Ready to Transform Your Hotel Design Process?
          </h2>
          <p className="cta-description">
            Start creating optimized hotel layouts with our AI-powered platform
            today
          </p>
          <button
            className="btn-hero-primary"
            onClick={() => navigate("/configure")}
          >
            Get Started Now
          </button>
        </div>
      </section>
    </div>
  );
};

export default HomePage;
