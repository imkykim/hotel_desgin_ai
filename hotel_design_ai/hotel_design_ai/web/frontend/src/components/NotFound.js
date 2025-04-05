import React from "react";
import { Link } from "react-router-dom";
import "../styles/NotFound.css";

const NotFoundPage = () => {
  return (
    <div className="not-found-page">
      <div className="not-found-container">
        <h1>404</h1>
        <h2>Page Not Found</h2>
        <p>The page you're looking for doesn't exist or has been moved.</p>
        <div className="not-found-actions">
          <Link to="/" className="btn-home">
            Go Home
          </Link>
          <Link to="/layouts" className="btn-layouts">
            View Layouts
          </Link>
        </div>
      </div>
    </div>
  );
};

export default NotFoundPage;
