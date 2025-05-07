import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  getLayout,
  generateLayoutVisualizations,
  exportLayoutToRhinoScript,
} from "../services/api";
import "../styles/ViewLayout.css";

const ViewLayout = () => {
  const [layout, setLayout] = useState(null);
  const [detailedMetrics, setDetailedMetrics] = useState(null);
  const [imageUrls, setImageUrls] = useState({ floor_plans: {} });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [activeView, setActiveView] = useState("3d");
  const [imageErrors, setImageErrors] = useState({});
  const { layoutId } = useParams();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("visualizations");
  const [layoutVisualizations, setLayoutVisualizations] = useState(null);
  const [visualizationLoading, setVisualizationLoading] = useState(false);

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
  const [lightbox, setLightbox] = useState({
    open: false,
    image: null,
    title: "",
  });

  // Add a function to open the lightbox
  const openLightbox = (imageUrl, title) => {
    setLightbox({
      open: true,
      image: imageUrl,
      title: title,
    });
  };

  // Add a function to close the lightbox
  const closeLightbox = () => {
    setLightbox({
      open: false,
      image: null,
      title: "",
    });
  };
  const fetchLayoutDetails = async () => {
    try {
      setLoading(true);
      console.log("Fetching layout details for:", layoutId);

      const response = await getLayout(layoutId);
      console.log("Layout details response:", response);

      if (response.success && response.layout_data) {
        setLayout(response.layout_data);

        if (response.detailed_metrics) {
          setDetailedMetrics(response.detailed_metrics);
        }

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

  const renderDetailedMetrics = () => {
    if (!detailedMetrics) return null;

    return (
      <div className="metrics-section">
        <h3>Detailed Performance Metrics</h3>
        <div className="detailed-metrics">
          {Object.entries(detailedMetrics).map(([key, value]) => {
            // Skip complex nested objects and arrays
            if (typeof value === "object" || Array.isArray(value)) {
              return null;
            }

            // Format the metric name and value
            const formattedName = key.replace(/_/g, " ");
            const formattedValue =
              typeof value === "number"
                ? value > 0 && value < 1
                  ? `${(value * 100).toFixed(1)}%`
                  : value.toFixed(2)
                : value.toString();

            return (
              <div key={key} className="metric-item">
                <span className="metric-name">{formattedName}</span>
                <span className="metric-value">{formattedValue}</span>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

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
    // Extract buildingId and programId from layout if not present in layout
    let buildingId = "default";
    let programId = "default";
    if (layout?.building_id && layout?.program_id) {
      buildingId = layout.building_id;
      programId = layout.program_id;
    } else if (layoutId) {
      // Example: layoutId = "building01_hotelreqs3_20240601_xxxxxx"
      const parts = layoutId.split("_");
      if (parts.length >= 2) {
        buildingId = parts[0];
        programId = parts[1];
      }
    }
    navigate(`/interactive/${buildingId}/${programId}?layoutId=${layoutId}`);
  };

  const formatMetric = (value) => {
    if (typeof value === "number") {
      return (value * 100).toFixed(1) + "%";
    }
    return value;
  };

  // Simplified error message
  const handleImageError = (key) => {
    console.error(`Image load error for ${key}`);
    setImageErrors((prev) => ({
      ...prev,
      [key]: true,
    }));
  };

  // Generate layout visualizations
  // In the handleGenerateVisualizations function:
  const handleGenerateVisualizations = async () => {
    try {
      setVisualizationLoading(true);
      setError(null);

      const response = await generateLayoutVisualizations(layoutId);
      console.log("Layout visualization response:", response);

      if (response.success) {
        setLayoutVisualizations(response.visualizations);
        // Switch to visualization tab
        setActiveTab("metrics");
      } else {
        setError(
          "Failed to generate visualizations: " +
            (response?.error || "Unknown error")
        );
      }
    } catch (err) {
      console.error("Error generating visualizations:", err);
      setError("Error generating visualizations: " + err.message);
    } finally {
      setVisualizationLoading(false);
    }
  };

  // In the metrics tab section:
  {
    activeTab === "metrics" && layoutVisualizations && (
      <div className="metrics-visualizations">
        <h3>Layout Metrics Visualizations</h3>

        {layoutVisualizations.filter((viz) => !imageErrors[viz.url]).length >
        0 ? (
          <div className="visualization-grid">
            {layoutVisualizations
              .filter((viz) => !imageErrors[viz.url])
              .map((viz, index) => (
                <div key={index} className="visualization-card">
                  <h4>{viz.title}</h4>
                  {viz.type === "text/html" ? (
                    <iframe
                      src={`${apiBaseUrl}${viz.url}`}
                      title={viz.title}
                      className="visualization-iframe"
                      onError={() => handleImageError(viz.url)}
                    />
                  ) : (
                    <img
                      src={`${apiBaseUrl}${viz.url}`}
                      alt={viz.title}
                      className="visualization-image"
                      onClick={() =>
                        openLightbox(`${apiBaseUrl}${viz.url}`, viz.title)
                      }
                      style={{ cursor: "pointer" }}
                      onError={() => handleImageError(viz.url)}
                    />
                  )}
                  // Then, add the lightbox component at the end of your
                  component:
                  {lightbox.open && (
                    <div className="lightbox-overlay" onClick={closeLightbox}>
                      <div
                        className="lightbox-content"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <div className="lightbox-header">
                          <h3>{lightbox.title}</h3>
                          <button
                            className="lightbox-close"
                            onClick={closeLightbox}
                          >
                            ×
                          </button>
                        </div>
                        <div className="lightbox-body">
                          <img
                            src={lightbox.image}
                            alt={lightbox.title}
                            className="lightbox-image"
                          />
                        </div>
                      </div>
                    </div>
                  )}
                  <p>{viz.description}</p>
                </div>
              ))}
          </div>
        ) : (
          <div className="no-visualizations">
            <p>
              No visualizations could be loaded. Please try regenerating them.
            </p>
          </div>
        )}
      </div>
    );
  }

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
  const getOverallScore = () => {
    if (
      detailedMetrics &&
      typeof detailedMetrics.overall_score !== "undefined"
    ) {
      return detailedMetrics.overall_score;
    }
    return metrics.overall_score || 0;
  };

  const overallScore = getOverallScore();

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
  // In your ViewLayout component
  const handleExportToRhino = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);

      const result = await exportLayoutToRhinoScript(layoutId);

      if (result.success) {
        setSuccess(
          "Rhino script downloaded successfully. Open the script in Rhino using the RunPythonScript command."
        );
      } else {
        setError(result.error || "Failed to export Rhino script");
      }
    } catch (err) {
      console.error("Error in handleExportToRhino:", err);
      setError("Error exporting to Rhino: " + (err.message || String(err)));
    } finally {
      setLoading(false);
    }
  };
  // Render room distribution for the blue box area
  const renderRoomDistribution = () => {
    return (
      <div className="room-distribution-container">
        <h3>Room Distribution</h3>
        <div className="room-distribution">
          {Object.entries(roomsByType).map(([type, count]) => (
            <div key={type} className="room-type-count">
              <span className="room-type">{type}</span>
              <span className="room-count">{count}</span>
            </div>
          ))}
        </div>
      </div>
    );
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

          <div className="view-tabs">
            <button
              className={`tab-button ${
                activeTab === "visualizations" ? "active" : ""
              }`}
              onClick={() => setActiveTab("visualizations")}
            >
              Floor Plans
            </button>
            <button
              className={`tab-button ${
                activeTab === "metrics" ? "active" : ""
              }`}
              onClick={() => setActiveTab("metrics")}
              disabled={!layoutVisualizations}
            >
              Metrics Visualizations
            </button>
            <button
              className="tab-button generate-vis"
              onClick={handleGenerateVisualizations}
              disabled={visualizationLoading}
            >
              {visualizationLoading
                ? "Generating..."
                : layoutVisualizations
                ? "Regenerate Visualizations"
                : "Generate Metrics Visualizations"}
            </button>
          </div>

          {activeTab === "visualizations" && (
            <>
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
                  <div className="placeholder-image">
                    {getPlaceholderText()}
                  </div>
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
            </>
          )}

          {activeTab === "metrics" && layoutVisualizations && (
            <div className="metrics-visualizations">
              <h3>Layout Metrics Visualizations</h3>

              {layoutVisualizations.filter((viz) => !imageErrors[viz.url])
                .length > 0 ? (
                <div className="visualization-grid">
                  {layoutVisualizations
                    .filter((viz) => !imageErrors[viz.url])
                    .map((viz, index) => (
                      <div key={index} className="visualization-card">
                        <h4>{viz.title}</h4>
                        {viz.type === "image/png" ? (
                          <img
                            src={`${apiBaseUrl}${viz.url}`}
                            alt={viz.title}
                            onError={() => handleImageError(viz.url)}
                          />
                        ) : (
                          <iframe
                            src={`${apiBaseUrl}${viz.url}`}
                            title={viz.title}
                            className="visualization-iframe"
                            onError={() => handleImageError(viz.url)}
                          />
                        )}
                        <p>{viz.description}</p>
                      </div>
                    ))}
                </div>
              ) : (
                <div className="no-visualizations">
                  <p>
                    No visualizations could be loaded. Please try regenerating
                    them.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Room distribution moved to blue box location */}
          {renderRoomDistribution()}
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

              {/* Score moved to red box location */}
              <div className="stat-item">
                <h3>Score</h3>
                <p>
                  {" "}
                  {overallScore > 0
                    ? formatMetric(overallScore)
                    : metrics.overall_score === 0
                    ? "0%"
                    : "N/A"}
                </p>
              </div>
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

            {detailedMetrics && renderDetailedMetrics()}
            {/* Rhino Export Button - Add this part below the detailed metrics */}
            <div className="rhino-export-section">
              <button
                className="btn-rhino-export"
                onClick={handleExportToRhino}
                disabled={loading}
              >
                <span className="export-icon">🦏</span> Open in Rhino
              </button>
              <p className="export-instructions">
                This will download a Python script that you can run in
                Rhinoceros 3D using the RunPythonScript command.
              </p>
            </div>
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
