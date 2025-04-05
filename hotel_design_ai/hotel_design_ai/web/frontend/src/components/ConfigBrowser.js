import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { listConfigurations } from "../services/api";
import "../styles/ConfigBrowser.css";

const ConfigBrowser = () => {
  const [configurations, setConfigurations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    fetchConfigurations();
  }, []);

  const fetchConfigurations = async () => {
    try {
      setLoading(true);
      const response = await listConfigurations();

      if (response.success && response.configurations) {
        setConfigurations(response.configurations);
      } else {
        throw new Error("Failed to fetch configurations");
      }
    } catch (err) {
      setError("Error loading configurations: " + err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleUseConfig = (configId) => {
    navigate(`/configure/${configId}`);
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
    <div className="config-browser">
      <h2>Configuration Browser</h2>
      <p>Browse and reuse previously created hotel configurations</p>

      {loading && <div className="loading">Loading configurations...</div>}

      {error && <div className="error-message">{error}</div>}

      {!loading && configurations.length === 0 && (
        <div className="no-configs">
          <p>No configurations found. Create your first configuration!</p>
          <button
            className="btn-primary"
            onClick={() => navigate("/configure")}
          >
            Create New Configuration
          </button>
        </div>
      )}

      <div className="configs-grid">
        {configurations.map((config) => (
          <div key={config.id} className="config-card">
            <div className="config-preview">
              <div className="config-icon">
                <i className="fas fa-building"></i>
              </div>
            </div>
            <div className="config-info">
              <h3>{config.name || `Config ${config.id}`}</h3>
              <p>Created: {formatDate(config.created_at)}</p>
              <p>Type: {config.hotel_type}</p>
              <p>Rooms: {config.num_rooms}</p>
              <button
                className="btn-view"
                onClick={() => handleUseConfig(config.id)}
              >
                Use Configuration
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="browser-actions">
        <button className="btn-primary" onClick={() => navigate("/configure")}>
          Create New Configuration
        </button>
        <button
          className="btn-refresh"
          onClick={fetchConfigurations}
          disabled={loading}
        >
          Refresh Configurations
        </button>
      </div>
    </div>
  );
};

export default ConfigBrowser;
