import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { listLayouts } from "../services/api";
import "../styles/LayoutGallery.css";

const LayoutGallery = () => {
  const [layouts, setLayouts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sortBy, setSortBy] = useState("date");
  const [imageErrors, setImageErrors] = useState({});
  const navigate = useNavigate();

  // Use API base URL
  const apiBaseUrl = process.env.REACT_APP_API_URL || "http://localhost:8000";

  useEffect(() => {
    fetchLayouts();
  }, []);

  const fetchLayouts = async () => {
    try {
      setLoading(true);
      console.log("Fetching layouts...");

      const response = await listLayouts();
      console.log("Layouts response:", response);

      if (response.success && response.layouts) {
        setLayouts(response.layouts);
      } else {
        console.error(
          "Failed to fetch layouts:",
          response?.error || "Unknown error"
        );
        setError(
          "Failed to fetch layouts: " + (response?.error || "Unknown error")
        );
      }
    } catch (err) {
      console.error("Error loading layouts:", err);
      setError("Error loading layouts: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleViewLayout = (layoutId) => {
    console.log(`Navigating to view layout: ${layoutId}`);
    navigate(`/view-layout/${layoutId}`);
  };

  const formatDate = (dateString) => {
    if (!dateString) return "Unknown date";

    // Try to format date nicely if it's in YYYY-MM-DD format
    try {
      const [year, month, day] = dateString.split("-");
      return `${month}/${day}/${year}`;
    } catch (e) {
      return dateString;
    }
  };

  // Sort layouts based on selected criteria
  const getSortedLayouts = () => {
    return [...layouts].sort((a, b) => {
      switch (sortBy) {
        case "date":
          // Sort by date (newest first)
          return (b.created_at || "").localeCompare(a.created_at || "");
        case "rooms":
          // Sort by room count (highest first)
          return (b.room_count || 0) - (a.room_count || 0);
        case "score":
          // Sort by overall score (highest first)
          const scoreA = a.metrics?.overall_score || 0;
          const scoreB = b.metrics?.overall_score || 0;
          return scoreB - scoreA;
        default:
          return 0;
      }
    });
  };

  // Handle image error by storing the error state
  const handleImageError = (layoutId) => {
    console.error(`Image load error for layout ${layoutId}`);
    setImageErrors((prev) => ({
      ...prev,
      [layoutId]: true,
    }));
  };

  return (
    <div className="config-browser">
      <div className="config-browser-header">
        <h2>Layout Gallery</h2>
        <p>Browse previously generated hotel layouts</p>

        <div className="filters">
          <div className="sort-by">
            <label htmlFor="sort-select">Sort by:</label>
            <select
              id="sort-select"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="sort-select"
            >
              <option value="date">Date (newest first)</option>
              <option value="rooms">Room Count (highest first)</option>
              <option value="score">Score (highest first)</option>
            </select>
          </div>
        </div>
      </div>

      {loading && <div className="loading">Loading layouts...</div>}

      {error && <div className="error-message">{error}</div>}

      {!loading && layouts.length === 0 && !error && (
        <div className="no-configs">
          <p>No layouts found. Generate your first layout!</p>
          <button
            className="btn-primary"
            onClick={() => navigate("/configure")}
          >
            Create New Layout
          </button>
        </div>
      )}

      <div className="configs-grid">
        {getSortedLayouts().map((layout) => {
          // Generate image URL - using the preview_image from backend if available
          const imageUrl =
            imageErrors[layout.id] || !layout.has_preview
              ? null
              : `${apiBaseUrl}${layout.preview_image}`;

          return (
            <div key={layout.id} className="config-card">
              <div className="config-preview">
                {!imageUrl ? (
                  <div className="placeholder-image">
                    <span className="config-icon">🏨</span>
                  </div>
                ) : (
                  <img
                    src={imageUrl}
                    alt={`Layout ${layout.id}`}
                    onError={() => handleImageError(layout.id)}
                  />
                )}
              </div>
              <div className="config-info">
                <h3>{layout.id.split("_")[1] || layout.id}</h3>
                <p>Created: {formatDate(layout.created_at)}</p>
                <p>Rooms: {layout.room_count || 0}</p>
                <p>
                  Dimensions: {layout.width || 0}m × {layout.length || 0}m ×{" "}
                  {layout.height || 0}m
                </p>
                {layout.metrics && layout.metrics.overall_score && (
                  <p>
                    Score: {(layout.metrics.overall_score * 100).toFixed(1)}%
                  </p>
                )}
                <div className="config-actions">
                  <button
                    className="btn-view"
                    onClick={() => handleViewLayout(layout.id)}
                  >
                    View Details
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="browser-actions">
        <button className="btn-primary" onClick={() => navigate("/configure")}>
          Create New Layout
        </button>
        <button
          className="btn-refresh"
          onClick={fetchLayouts}
          disabled={loading}
        >
          Refresh Gallery
        </button>
      </div>
    </div>
  );
};

export default LayoutGallery;
