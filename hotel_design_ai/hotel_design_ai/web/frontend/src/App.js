import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Link,
  useNavigate,
} from "react-router-dom";
import "./App.css";

// Import pages
import ConfigGenerator from "./pages/ConfigGenerator";
import InteractiveLayoutPage from "./pages/InteractiveLayoutPage";
import LayoutGallery from "./pages/LayoutGallery";
import ViewLayoutPage from "./pages/ViewLayoutPage";
import NotFoundPage from "./pages/NotFoundPage";

// Main navigation component
function Navigation() {
  return (
    <nav className="main-nav">
      <ul>
        <li>
          <Link to="/">Home</Link>
        </li>
        <li>
          <Link to="/configure">Create New Design</Link>
        </li>
        <li>
          <Link to="/layouts">Layout Gallery</Link>
        </li>
      </ul>
    </nav>
  );
}

// Home page component
function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="home-page">
      <header className="App-header">
        <h1>Hotel Design AI</h1>
        <p>Generate optimized hotel layouts using artificial intelligence</p>
      </header>

      <main className="home-content">
        <div className="feature-section">
          <h2>Create Your Hotel Design</h2>
          <p>
            Use our AI-powered system to generate customized hotel layouts based
            on your requirements. Specify the size, number of rooms, amenities,
            and other parameters to get started.
          </p>
          <button
            className="btn-primary"
            onClick={() => navigate("/configure")}
          >
            Create New Design
          </button>
        </div>

        <div className="feature-section">
          <h2>Browse Previous Designs</h2>
          <p>
            View, modify, and learn from previously generated hotel layouts. Our
            gallery provides examples and inspiration for your next design.
          </p>
          <button
            className="btn-secondary"
            onClick={() => navigate("/layouts")}
          >
            Browse Layout Gallery
          </button>
        </div>

        <div className="demo-section">
          <h2>Try Our Demo</h2>
          <p>
            Jump right in and experience our interactive design system with a
            pre-configured sample hotel layout.
          </p>
          <button
            className="btn-demo"
            onClick={() => navigate("/interactive/sample/sample")}
          >
            Try Interactive Demo
          </button>
        </div>
      </main>
    </div>
  );
}

// Main App component with routing
function App() {
  return (
    <Router>
      <div className="App">
        <Navigation />

        <div className="content-container">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/configure" element={<ConfigGenerator />} />
            <Route path="/layouts" element={<LayoutGallery />} />
            <Route path="/view-layout/:layoutId" element={<ViewLayoutPage />} />
            <Route
              path="/interactive/:buildingId/:programId"
              element={<InteractiveLayoutPage />}
            />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </div>

        <footer className="App-footer">
          <p>&copy; {new Date().getFullYear()} Hotel Design AI</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;
