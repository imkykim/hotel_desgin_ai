import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { listLayouts } from "../services/api";
import "../styles/LayoutGallery.css";

const LayoutGallery = () => {
  const [layouts, setLayouts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    fetchLayouts();
  }, []);

  const fetchLayouts = async () => {
    try {
      setLoading(true);
      const response = await listLayouts();

      if (response.success && response.layouts) {
        setLayouts(response.layouts);
      } else {
        throw new Error("Failed to fetch layouts");
      }
    } catch (err) {
      setError("Error loading layouts: " + err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleViewLayout = (layoutId) => {
    navigate(`/view-layout/${layoutId}`);
  };

  const formatDate = (dateString) => {
    if (!dateString) return "Unknown date";

    // Parse YYYYMMDD_HHMMSS format
    const year = dateString.substring(0, 4);
    const month = dateString.substring(4, 6);
    const day = dateString.substring(6, 8);

    return `${year}-${month}-${day}`;
  };

  return (
    <div className="layout-gallery">
      <h2>Layout Gallery</h2>
      <p>Browse previously generated hotel layouts</p>

      {loading && <div className="loading">Loading layouts...</div>}

      {error && <div className="error-message">{error}</div>}

      {!loading && layouts.length === 0 && (
        <div className="no-layouts">
          <p>No layouts found. Generate your first layout!</p>
          <button
            className="btn-primary"
            onClick={() => navigate("/configure")}
          >
            Create New Layout
          </button>
        </div>
      )}

      <div className="layouts-grid">
        {layouts.map((layout) => (
          <div key={layout.id} className="layout-card">
            <div className="layout-thumbnail">
              <img src={layout.thumbnail} alt={`Layout ${layout.id}`} />
            </div>
            <div className="layout-info">
              <h3>Layout {layout.id.split("_")[1] || layout.id}</h3>
              <p>Created: {formatDate(layout.created_at)}</p>
              <p>Rooms: {layout.room_count}</p>
              {layout.metrics && layout.metrics.overall_score && (
                <p>Score: {(layout.metrics.overall_score * 100).toFixed(1)}%</p>
              )}
              <button
                className="btn-view"
                onClick={() => handleViewLayout(layout.id)}
              >
                View Details
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="gallery-actions">
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
