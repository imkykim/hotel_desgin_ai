import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getConfiguration, generateVisualizations } from "../services/api";
import "../styles/ConfigurationDetail.css";

const ConfigurationDetail = () => {
  const { configType, configId } = useParams();
  const navigate = useNavigate();
  const [config, setConfig] = useState(null);
  const [visualizations, setVisualizations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [visualizationLoading, setVisualizationLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("json");
  const [imageErrors, setImageErrors] = useState({});

  const apiBaseUrl = process.env.REACT_APP_API_URL || "http://localhost:8000";

  useEffect(() => {
    fetchConfigurationDetails();
  }, [configType, configId]);

  const fetchConfigurationDetails = async () => {
    try {
      setLoading(true);
      setError(null);
      console.log("Fetching configuration details for:", configType, configId);

      const response = await getConfiguration(configType, configId);
      console.log("Configuration details response:", response);

      if (response.success && response.config_data) {
        setConfig({
          id: configId,
          type: configType,
          data: response.config_data,
        });
      } else {
        setError(
          "Failed to fetch configuration: " +
            (response?.error || "Unknown error")
        );
      }
    } catch (err) {
      console.error("Error loading configuration:", err);
      setError("Error loading configuration: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateVisualizations = async () => {
    try {
      setVisualizationLoading(true);
      setError(null);

      const response = await generateVisualizations(configType, configId);
      console.log("Visualization response:", response);

      if (response.success) {
        setVisualizations(response.visualizations);
        // Switch to visualizations tab
        setActiveTab("visualizations");
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

  const handleImageError = (key) => {
    console.error(`Image load error for ${key}`);
    setImageErrors((prev) => ({
      ...prev,
      [key]: true,
    }));
  };

  // Format JSON for display
  const formatJson = (jsonObj) => {
    return JSON.stringify(jsonObj, null, 2);
  };

  // Get a descriptive name for the configuration
  const getConfigTitle = () => {
    if (!config) return "";

    if (configType === "building") {
      return `Building Configuration: ${configId}`;
    } else if (configType === "program") {
      return `Program Configuration: ${configId}`;
    } else {
      return `Configuration: ${configId}`;
    }
  };

  if (loading) {
    return (
      <div className="loading-container">Loading configuration details...</div>
    );
  }

  if (error) {
    return (
      <div className="error-container">
        <h2>Error</h2>
        <p>{error}</p>
        <button
          className="btn-primary"
          onClick={() => navigate("/configurations")}
        >
          Back to Configurations
        </button>
      </div>
    );
  }

  if (!config) {
    return (
      <div className="not-found-container">
        <h2>Configuration Not Found</h2>
        <p>The requested configuration could not be found.</p>
        <button
          className="btn-primary"
          onClick={() => navigate("/configurations")}
        >
          Back to Configurations
        </button>
      </div>
    );
  }

  return (
    <div className="config-detail-page">
      <div className="config-header">
        <h1>{getConfigTitle()}</h1>
      </div>

      <div className="config-actions">
        <button
          className="btn-back"
          onClick={() => navigate("/configurations")}
        >
          Back to Configurations
        </button>
        <button
          className="btn-generate"
          onClick={handleGenerateVisualizations}
          disabled={visualizationLoading}
        >
          {visualizationLoading
            ? "Generating..."
            : visualizations
            ? "Regenerate Visualizations"
            : "Generate Visualizations"}
        </button>
      </div>

      <div className="config-content">
        <div className="config-tabs">
          <button
            className={`tab-button ${activeTab === "json" ? "active" : ""}`}
            onClick={() => setActiveTab("json")}
          >
            JSON Content
          </button>
          <button
            className={`tab-button ${
              activeTab === "visualizations" ? "active" : ""
            }`}
            onClick={() => setActiveTab("visualizations")}
            disabled={!visualizations}
          >
            Visualizations
          </button>
        </div>

        <div className="tab-content">
          {activeTab === "json" && (
            <div className="json-content">
              <pre>{formatJson(config.data)}</pre>
            </div>
          )}

          {activeTab === "visualizations" && visualizations && (
            <div className="visualizations-content">
              {configType === "building" && (
                <div className="visualization-section">
                  <h2>Building Configuration Visualizations</h2>
                  {visualizations.filter((viz) => !imageErrors[viz.url])
                    .length > 0 ? (
                    <div className="visualization-grid">
                      {visualizations
                        .filter((viz) => !imageErrors[viz.url])
                        .map((viz, index) => (
                          <div key={index} className="visualization-card">
                            <h3>{viz.title}</h3>
                            <img
                              src={`${apiBaseUrl}${viz.url}`}
                              alt={viz.title}
                              onError={() => handleImageError(viz.url)}
                            />
                            <p>{viz.description}</p>
                          </div>
                        ))}
                    </div>
                  ) : (
                    <div className="no-visualizations">
                      <p>
                        No visualizations could be loaded. Please try
                        regenerating them.
                      </p>
                    </div>
                  )}
                </div>
              )}

              {configType === "program" && (
                <div className="visualization-section">
                  <h2>Program Configuration Visualizations</h2>
                  {visualizations.filter((viz) => !imageErrors[viz.url])
                    .length > 0 ? (
                    <div className="visualization-grid">
                      {visualizations
                        .filter((viz) => !imageErrors[viz.url])
                        .map((viz, index) => (
                          <div key={index} className="visualization-card">
                            <h3>{viz.title}</h3>
                            <img
                              src={`${apiBaseUrl}${viz.url}`}
                              alt={viz.title}
                              onError={() => handleImageError(viz.url)}
                            />
                            <p>{viz.description}</p>
                          </div>
                        ))}
                    </div>
                  ) : (
                    <div className="no-visualizations">
                      <p>
                        No visualizations could be loaded. Please try
                        regenerating them.
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ConfigurationDetail;
