import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getLayout } from "../services/api";
import "../styles/ViewLayout.css";

const ViewLayout = () => {
  const [layout, setLayout] = useState(null);
  const [imageUrls, setImageUrls] = useState({ floor_plans: {} });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeView, setActiveView] = useState("3d"); // Changed from activeFloor to activeView
  const [imageErrors, setImageErrors] = useState({});
  const { layoutId } = useParams();
  const navigate = useNavigate();

  // Get API base URL for images
  const apiBaseUrl = process.env.REACT_APP_API_URL || "http://localhost:8000";

  useEffect(() => {
    fetchLayoutDetails();
  }, [layoutId]);

  // Set default active view once images are loaded
  useEffect(() => {
    if (!loading && imageUrls) {
      // Try to set the default view to the first available image
      if (imageUrls.has_3d_preview && !imageErrors["3d"]) {
        setActiveView("3d");
      } else {
        // Check for available floor plans
        const availableFloors = Object.keys(imageUrls.floor_plans).filter(
          (floor) => !imageErrors[`floor${floor}`]
        );

        if (availableFloors.length > 0) {
          // Sort floors to prioritize ground floor (0), then positive floors, then basements
          availableFloors.sort((a, b) => {
            if (a === "0") return -1;
            if (b === "0") return 1;
            if (a === "std") return 1;
            if (b === "std") return -1;
            return parseInt(a) - parseInt(b);
          });

          setActiveView(availableFloors[0]);
        }
      }
    }
  }, [loading, imageUrls, imageErrors]);

  const fetchLayoutDetails = async () => {
    try {
      setLoading(true);
      console.log("Fetching layout details for:", layoutId);

      const response = await getLayout(layoutId);
      console.log("Layout details response:", response);

      if (response.success && response.layout_data) {
        setLayout(response.layout_data);

        // Process image URLs - check for both standard and basement naming conventions
        const processedImageUrls = {
          ...response.image_urls,
          floor_plans: { ...response.image_urls.floor_plans },
        };

        // Check for basement images with alternate naming
        checkForBasementImages(processedImageUrls, layoutId);

        // Check for standard floor image
        checkForStandardFloorImage(processedImageUrls, layoutId);

        setImageUrls(processedImageUrls);
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

  // Check for basement images with alternate naming
  const checkForBasementImages = (imageUrls, layoutId) => {
    // Check for basement1.png instead of floor-1.png
    if (!imageUrls.floor_plans[-1]) {
      const basementUrl = `/layouts/${layoutId}/hotel_layout_basement1.png`;
      // We'll add it and let the image error handler remove it if it doesn't exist
      imageUrls.floor_plans[-1] = basementUrl;
    }

    // Check for basement2.png instead of floor-2.png
    if (!imageUrls.floor_plans[-2]) {
      const basement2Url = `/layouts/${layoutId}/hotel_layout_basement2.png`;
      imageUrls.floor_plans[-2] = basement2Url;
    }
  };

  // Check for standard floor image
  const checkForStandardFloorImage = (imageUrls, layoutId) => {
    const standardFloorUrl = `/layouts/${layoutId}/hotel_layout_standard_floor.png`;
    // Add special key for standard floor
    imageUrls.floor_plans["std"] = standardFloorUrl;
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

  // Extract building dimensions - handle both direct and nested structures
  const buildingWidth =
    layout.width || (layout.dimensions && layout.dimensions.width) || 0;
  const buildingLength =
    layout.length || (layout.dimensions && layout.dimensions.length) || 0;
  const buildingHeight =
    layout.height || (layout.dimensions && layout.dimensions.height) || 0;

  // Extract metrics - both from root and from metrics property
  const metrics = layout.metrics || {};
  const overallScore = metrics.overall_score || 0;

  // Get the image URL for the current view
  const getImageUrl = () => {
    if (activeView === "3d") {
      // 3D view
      return imageUrls.has_3d_preview
        ? `${apiBaseUrl}${imageUrls["3d"]}`
        : null;
    } else {
      // Floor plan or standard floor
      const floorKey = activeView === "std" ? "std" : parseInt(activeView);
      return floorKey in imageUrls.floor_plans
        ? `${apiBaseUrl}${imageUrls.floor_plans[floorKey]}`
        : null;
    }
  };

  // Create an error key for image error tracking
  const getImageErrorKey = () => {
    return activeView === "3d"
      ? "3d"
      : activeView === "std"
      ? "std"
      : `floor${activeView}`;
  };

  // Check if we have an image for the active view
  const hasImage = () => {
    if (activeView === "3d") {
      return imageUrls.has_3d_preview && !imageErrors["3d"];
    }
    const floorKey = activeView === "std" ? "std" : parseInt(activeView);
    return (
      floorKey in imageUrls.floor_plans && !imageErrors[getImageErrorKey()]
    );
  };

  // Display a placeholder for failed images
  const getPlaceholderText = () => {
    if (activeView === "3d") return "No 3D View Available";
    if (activeView === "std") return "No Standard Floor Plan Available";

    const floorNum = parseInt(activeView);
    return floorNum < 0
      ? `No Basement ${Math.abs(floorNum)} Plan Available`
      : `No Floor ${floorNum} Plan Available`;
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
            {/* 3D View button - only show if it exists */}
            {imageUrls.has_3d_preview && !imageErrors["3d"] && (
              <button
                className={activeView === "3d" ? "active" : ""}
                onClick={() => setActiveView("3d")}
              >
                3D View
              </button>
            )}

            {/* Basement floors - only show if images exist */}
            {["-2", "-1"].map((floor) => {
              const floorKey = parseInt(floor);
              const hasImage =
                floorKey in imageUrls.floor_plans &&
                !imageErrors[`floor${floor}`];

              return hasImage ? (
                <button
                  key={`floor-${floor}`}
                  className={activeView === floor ? "active" : ""}
                  onClick={() => setActiveView(floor)}
                >
                  {`B${Math.abs(parseInt(floor))}`}
                </button>
              ) : null;
            })}

            {/* Main floors - only show if images exist */}
            {["0", "1", "2", "3", "4", "5"].map((floor) => {
              const floorKey = parseInt(floor);
              const hasImage =
                floorKey in imageUrls.floor_plans &&
                !imageErrors[`floor${floor}`];

              return hasImage ? (
                <button
                  key={`floor-${floor}`}
                  className={activeView === floor ? "active" : ""}
                  onClick={() => setActiveView(floor)}
                >
                  {`F${floor}`}
                </button>
              ) : null;
            })}

            {/* Standard floor button - only show if image exists */}
            {imageUrls.floor_plans["std"] && !imageErrors["std"] && (
              <button
                className={activeView === "std" ? "active" : ""}
                onClick={() => setActiveView("std")}
              >
                Standard
              </button>
            )}
          </div>

          <div className="visualization-display">
            {!hasImage() ? (
              <div className="placeholder-image">{getPlaceholderText()}</div>
            ) : (
              <img
                src={getImageUrl()}
                alt={
                  activeView === "3d"
                    ? "3D Layout View"
                    : activeView === "std"
                    ? "Standard Floor Plan"
                    : `Floor ${activeView} Plan`
                }
                className="layout-image"
                onError={() => handleImageError(getImageErrorKey())}
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
                  {buildingWidth}m × {buildingLength}m × {buildingHeight}m
                </p>
              </div>

              <div className="stat-item">
                <h3>Score</h3>
                <p>{overallScore ? formatMetric(overallScore) : "N/A"}</p>
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

            {Object.keys(metrics).length > 0 && (
              <>
                <h3>Performance Metrics</h3>
                <div className="metrics-list">
                  {Object.entries(metrics).map(([key, value]) => {
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
