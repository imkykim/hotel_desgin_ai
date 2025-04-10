import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import "./App.css";

// Import pages
import HomePage from "./pages/HomePage";
import ConfigGenerator from "./pages/ConfigGenerator";
import ConfigBrowserPage from "./pages/ConfigBrowserPage";
import ConfigurationDetailPage from "./pages/ConfigurationDetailPage";
import InteractiveLayoutPage from "./pages/InteractiveLayoutPage";
import LayoutGallery from "./pages/LayoutGallery";
import ViewLayoutPage from "./pages/ViewLayoutPage";
import NotFoundPage from "./pages/NotFoundPage";

// Main navigation component
function Navigation() {
  return (
    <nav className="main-nav">
      <div className="nav-logo">
        <Link to="/">Hotel Design AI</Link>
      </div>
      <ul className="nav-links">
        <li>
          <Link to="/">Home</Link>
        </li>
        <li>
          <Link to="/configure">Create New Design</Link>
        </li>
        <li>
          <Link to="/configurations">Configurations</Link>
        </li>
        <li>
          <Link to="/layouts">Layout Gallery</Link>
        </li>
      </ul>
    </nav>
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
            <Route path="/configurations" element={<ConfigBrowserPage />} />
            <Route
              path="/configuration/:configType/:configId"
              element={<ConfigurationDetailPage />}
            />
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
