import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { listConfigurations } from "../services/api";
import "../styles/ConfigBrowser.css";

const ConfigBrowser = () => {
  const [buildingConfigs, setBuildingConfigs] = useState([]);
  const [programConfigs, setProgramConfigs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("building");
  const navigate = useNavigate();

  useEffect(() => {
    // Safe fetch configurations function
    const fetchConfigs = async () => {
      try {
        setLoading(true);
        console.log("Fetching configurations...");

        // Fetch configurations from API
        const response = await listConfigurations();
        console.log("Configuration response:", response);

        if (response && response.success) {
          setBuildingConfigs(response.building_configs || []);
          setProgramConfigs(response.program_configs || []);
        } else {
          console.error(
            "Failed to fetch configurations:",
            response?.error || "Unknown error"
          );
          setError("Failed to fetch configurations. Please try again later.");
        }
      } catch (err) {
        console.error("Error in fetchConfigs:", err);
        setError(
          "An error occurred while loading configurations: " + err.message
        );
      } finally {
        setLoading(false);
      }
    };

    fetchConfigs();
  }, []);

  return (
    <div className="config-browser">
      <h2>Configuration Browser</h2>
      <p>Browse and reuse previously created hotel configurations</p>

      {loading && <div className="loading">Loading configurations...</div>}

      {error && <div className="error-message">{error}</div>}

      {!loading && !error && (
        <div className="tab-navigation">
          <button
            className={`tab-button ${activeTab === "building" ? "active" : ""}`}
            onClick={() => setActiveTab("building")}
          >
            Building Configurations ({buildingConfigs.length})
          </button>
          <button
            className={`tab-button ${activeTab === "program" ? "active" : ""}`}
            onClick={() => setActiveTab("program")}
          >
            Program Configurations ({programConfigs.length})
          </button>
        </div>
      )}

      {!loading &&
        !error &&
        activeTab === "building" &&
        (buildingConfigs.length > 0 ? (
          <div className="configs-grid">
            {buildingConfigs.map((config) => (
              <div key={config.id} className="config-card">
                <div className="config-info">
                  <h3>{config.id}</h3>
                  <p>Type: Building</p>
                  <p>
                    Dimensions: {config.dimensions?.width || 0}m ×{" "}
                    {config.dimensions?.length || 0}m ×{" "}
                    {config.dimensions?.height || 0}m
                  </p>
                  <button
                    className="btn-view"
                    onClick={() => {
                      if (programConfigs.length > 0) {
                        navigate(
                          `/interactive/${config.id}/${programConfigs[0].id}`
                        );
                      } else {
                        alert("No program configurations available.");
                      }
                    }}
                  >
                    Use Configuration
                  </button>
                  {/* Add a div spacer */}
                  <div style={{ height: "5px" }}></div>
                  <button
                    className="btn-view"
                    onClick={() =>
                      navigate(`/configuration/program/${config.id}`)
                    }
                  >
                    View Details
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="no-configs">
            <p>No building configurations found.</p>
            <button
              className="btn-primary"
              onClick={() => navigate("/configure")}
            >
              Create New Configuration
            </button>
          </div>
        ))}

      {!loading &&
        !error &&
        activeTab === "program" &&
        (programConfigs.length > 0 ? (
          <div className="configs-grid">
            {programConfigs.map((config) => (
              <div key={config.id} className="config-card">
                <div className="config-info">
                  <h3>{config.id}</h3>
                  <p>Type: Program</p>
                  <p>Hotel Type: {config.hotel_type || "Standard"}</p>
                  <button
                    className="btn-view"
                    onClick={() => {
                      if (buildingConfigs.length > 0) {
                        navigate(
                          `/interactive/${buildingConfigs[0].id}/${config.id}`
                        );
                      } else {
                        alert("No building configurations available.");
                      }
                    }}
                  >
                    Use Configuration
                  </button>
                  {/* Add a div spacer */}
                  <div style={{ height: "5px" }}></div>
                  <button
                    className="btn-view"
                    onClick={() =>
                      navigate(`/configuration/building/${config.id}`)
                    }
                  >
                    View Details
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="no-configs">
            <p>No program configurations found.</p>
            <button
              className="btn-primary"
              onClick={() => navigate("/configure")}
            >
              Create New Configuration
            </button>
          </div>
        ))}

      <div className="browser-actions">
        <button className="btn-primary" onClick={() => navigate("/configure")}>
          Create New Configuration
        </button>
      </div>
    </div>
  );
};

export default ConfigBrowser;
