import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getLayout } from "../services/api";
import "../styles/ViewLayout.css";

const ViewLayout = () => {
  const [layout, setLayout] = useState(null);
  const [imageUrls, setImageUrls] = useState({ floor_plans: {} });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeFloor, setActiveFloor] = useState(0);
  const [imageErrors, setImageErrors] = useState({});
  const { layoutId } = useParams();
  const navigate = useNavigate();

  // Get API base URL for images
  const apiBaseUrl = process.env.REACT_APP_API_URL || "http://localhost:8000";

  useEffect(() => {
    fetchLayoutDetails();
  }, [layoutId]);

  // Update this in the ViewLayout component's fetchLayoutDetails function

  const fetchLayoutDetails = async () => {
    try {
      setLoading(true);
      console.log("Fetching layout details for:", layoutId);

      // Explicitly make sure we're calling the API endpoint, not navigating
      const response = await getLayout(layoutId);
      console.log("Layout details response:", response);

      if (response.success && response.layout_data) {
        setLayout(response.layout_data);
        setImageUrls(response.image_urls || { floor_plans: {} });
      } else {
        console.error(
          "Failed to fetch layout details:",
          response?.error || "Unknown error"
        );
        setError(
          "Failed to fetch layout details: " +
            (response?.error || "Unknown error")
        );
      }
    } catch (err) {
      console.error("Error loading layout:", err);
      setError("Error loading layout: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleEditLayout = () => {
    // Navigate to interactive editor with this layout
    navigate(`/interactive/default/default?layoutId=${layoutId}`);
  };

  const formatMetric = (value) => {
    if (typeof value === "number") {
      return (value * 100).toFixed(1) + "%";
    }
    return value;
  };

  const handleImageError = (key) => {
    console.error(`Image load error for ${key}`);
    setImageErrors((prev) => ({
      ...prev,
      [key]: true,
    }));
  };

  if (loading) {
    return <div className="loading-container">Loading layout details...</div>;
  }

  if (error) {
    return (
      <div className="error-container">
        <h2>Error</h2>
        <p>{error}</p>
        <button className="btn-primary" onClick={() => navigate("/layouts")}>
          Back to Gallery
        </button>
      </div>
    );
  }

  if (!layout) {
    return (
      <div className="not-found-container">
        <h2>Layout Not Found</h2>
        <p>The requested layout could not be found.</p>
        <button className="btn-primary" onClick={() => navigate("/layouts")}>
          Back to Gallery
        </button>
      </div>
    );
  }

  // Extract room count by type
  const roomsByType = {};
  Object.values(layout.rooms || {}).forEach((room) => {
    const type = room.type;
    if (!roomsByType[type]) {
      roomsByType[type] = 0;
    }
    roomsByType[type]++;
  });

  // Get total area
  const totalArea = Object.values(layout.rooms || {}).reduce((total, room) => {
    const width = room.dimensions[0];
    const length = room.dimensions[1];
    return total + width * length;
  }, 0);

  // Get the image URL for the current view
  const getImageUrl = (floor) => {
    if (floor === -1) {
      // 3D view
      return imageUrls.has_3d_preview
        ? `${apiBaseUrl}${imageUrls["3d"]}`
        : null;
    } else {
      // Floor plan
      return floor in imageUrls.floor_plans
        ? `${apiBaseUrl}${imageUrls.floor_plans[floor]}`
        : null;
    }
  };

  // Create an error key for image error tracking
  const getImageErrorKey = (floor) => {
    return floor === -1 ? "3d" : `floor${floor}`;
  };

  // Check if we have an image for this floor
  const hasFloorImage = (floor) => {
    if (floor === -1) {
      return imageUrls.has_3d_preview && !imageErrors["3d"];
    }
    return floor in imageUrls.floor_plans && !imageErrors[`floor${floor}`];
  };

  // Display a placeholder for failed images
  const getPlaceholderText = (floor) => {
    return floor === -1
      ? "No 3D View Available"
      : `No Floor ${floor} Plan Available`;
  };

  return (
    <div className="view-layout-page">
      <div className="layout-header">
        <h1>Layout Details</h1>
        <p>ID: {layoutId}</p>
      </div>

      <div className="layout-content">
        <div className="layout-visualizations">
          <h2>Visualizations</h2>

          <div className="visualization-tabs">
            <button
              className={activeFloor === -1 ? "active" : ""}
              onClick={() => setActiveFloor(-1)}
            >
              3D View
            </button>

            {/* Floor buttons - only show floors that have images */}
            {[-2, -1, 0, 1, 2, 3, 4, 5].map((floor) => (
              <button
                key={`floor-${floor}`}
                className={activeFloor === floor ? "active" : ""}
                onClick={() => setActiveFloor(floor)}
              >
                {floor < 0 ? `B${Math.abs(floor)}` : `F${floor}`}
              </button>
            ))}
          </div>

          <div className="visualization-display">
            {!hasFloorImage(activeFloor) ? (
              <div className="placeholder-image">
                {getPlaceholderText(activeFloor)}
              </div>
            ) : (
              <img
                src={getImageUrl(activeFloor)}
                alt={
                  activeFloor === -1
                    ? "3D Layout View"
                    : `Floor ${activeFloor} Plan`
                }
                className="layout-image"
                onError={() => handleImageError(getImageErrorKey(activeFloor))}
              />
            )}
          </div>
        </div>

        <div className="layout-details">
          <div className="layout-stats">
            <h2>Layout Statistics</h2>

            <div className="stats-grid">
              <div className="stat-item">
                <h3>Total Rooms</h3>
                <p>{Object.keys(layout.rooms || {}).length}</p>
              </div>

              <div className="stat-item">
                <h3>Total Area</h3>
                <p>{totalArea.toFixed(1)} m²</p>
              </div>

              <div className="stat-item">
                <h3>Building Dimensions</h3>
                <p>
                  {layout.width}m × {layout.length}m × {layout.height}m
                </p>
              </div>

              <div className="stat-item">
                <h3>Score</h3>
                <p>
                  {layout.metrics && layout.metrics.overall_score
                    ? formatMetric(layout.metrics.overall_score)
                    : "N/A"}
                </p>
              </div>
            </div>

            <h3>Room Distribution</h3>
            <div className="room-distribution">
              {Object.entries(roomsByType).map(([type, count]) => (
                <div key={type} className="room-type-count">
                  <span className="room-type">{type}</span>
                  <span className="room-count">{count}</span>
                </div>
              ))}
            </div>

            {layout.metrics && (
              <>
                <h3>Performance Metrics</h3>
                <div className="metrics-list">
                  {Object.entries(layout.metrics).map(([key, value]) => {
                    // Skip complex metrics and overall score (already shown)
                    if (typeof value === "object" || key === "overall_score")
                      return null;

                    return (
                      <div key={key} className="metric-item">
                        <span className="metric-name">
                          {key.replace(/_/g, " ")}:
                        </span>
                        <span className="metric-value">
                          {formatMetric(value)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>

          <div className="layout-actions">
            <button className="btn-edit" onClick={handleEditLayout}>
              Edit Layout
            </button>

            <button className="btn-back" onClick={() => navigate("/layouts")}>
              Back to Gallery
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ViewLayout;
