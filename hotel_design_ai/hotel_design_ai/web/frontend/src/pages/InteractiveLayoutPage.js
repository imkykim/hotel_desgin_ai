import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import GridSelector from "../components/GridSelector";
import LayoutEditor from "../components/LayoutEditor";
import "../styles/InteractiveLayout.css";

const sampleBuildingConfig = {
  width: 60.0,
  length: 40.0,
  height: 30.0,
  min_floor: -2,
  max_floor: 5,
  floor_height: 4.5,
  structural_grid_x: 8.0,
  structural_grid_y: 8.0,
  grid_size: 1.0,
  main_entry: "front",
  description: "Sample hotel building for testing",
  units: "meters",
};

const sampleInitialLayout = {
  rooms: {
    1: {
      id: 1,
      type: "lobby",
      position: [10, 10, 0],
      dimensions: [20, 30, 4.5],
      metadata: { name: "Main Lobby" },
    },
    2: {
      id: 2,
      type: "entrance",
      position: [30, 10, 0],
      dimensions: [10, 8, 4.5],
      metadata: { name: "Main Entrance" },
    },
    3: {
      id: 3,
      type: "vertical_circulation",
      position: [40, 20, 0],
      dimensions: [8, 8, 25],
      metadata: { name: "Main Core" },
    },
    4: {
      id: 4,
      type: "restaurant",
      position: [10, 40, 0],
      dimensions: [20, 15, 4.5],
      metadata: { name: "Restaurant" },
    },
  },
};

const InteractiveLayoutPage = () => {
  const [activeTab, setActiveTab] = useState("grid");
  const [buildingConfig, setBuildingConfig] = useState(null);
  const [initialLayout, setInitialLayout] = useState(null);
  const [selectedAreas, setSelectedAreas] = useState([]);
  const [modifiedLayout, setModifiedLayout] = useState(null);
  const [layoutId, setLayoutId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const { buildingId, programId } = useParams();
  const navigate = useNavigate();

  // Fetch building configuration
  useEffect(() => {
    const fetchBuildingConfig = async () => {
      try {
        // If using sample data, don't fetch
        if (buildingId === "sample" && programId === "sample") {
          setBuildingConfig(sampleBuildingConfig);
          setInitialLayout(sampleInitialLayout);
          return;
        }

        // Original fetch logic for real configurations
        const response = await fetch(`/api/data/building/${buildingId}`);
        if (!response.ok) {
          throw new Error("Failed to fetch building configuration");
        }
        const data = await response.json();
        setBuildingConfig(data);
      } catch (err) {
        setError(err.message);
      }
    };

    if (buildingId) {
      fetchBuildingConfig();
    }
  }, [buildingId, programId]);

  // Handle grid selection changes
  const handleSelectionChange = (areas) => {
    setSelectedAreas(areas);
  };

  // Save standard floor zones
  const saveStandardFloorZones = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);

      const response = await fetch("/api/engine/standard-floor-zones", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          building_id: buildingId,
          floor_zones: selectedAreas,
          start_floor: 1,
          end_floor: buildingConfig?.max_floor || 3,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || "Failed to save standard floor zones"
        );
      }

      const data = await response.json();
      setSuccess("Standard floor zones saved successfully");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Generate layout with defined zones
  const generateWithZones = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);

      const response = await fetch("/api/engine/generate-with-zones", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          building_id: buildingId,
          program_id: programId,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to generate layout");
      }

      const data = await response.json();
      setLayoutId(data.layout_id);
      setInitialLayout(data.layout_data);
      setModifiedLayout(data.layout_data);
      setSuccess("Layout generated successfully");

      // Switch to layout editor tab
      setActiveTab("layout");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle layout changes
  const handleLayoutChange = (updatedLayout) => {
    setModifiedLayout(updatedLayout);
  };

  // Save modified layout
  const saveModifiedLayout = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);

      const response = await fetch("/api/engine/save-modified-layout", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          layout_id: layoutId,
          layout_data: modifiedLayout,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to save modified layout");
      }

      const data = await response.json();
      setSuccess("Layout modifications saved successfully");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Train RL with feedback
  const handleTrainRL = async (feedbackData) => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);

      // First save the modified layout
      await saveModifiedLayout();

      // Then submit for RL training
      const response = await fetch("/api/engine/train-rl", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          layout_id: layoutId,
          modified_layout: modifiedLayout,
          user_rating: feedbackData.userRating,
          comments: "User modified layout",
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to train RL model");
      }

      const data = await response.json();
      setSuccess("RL model trained successfully with your feedback");

      // Ask user if they want to generate an improved layout
      if (
        window.confirm(
          "Would you like to generate an improved layout based on your feedback?"
        )
      ) {
        generateImprovedLayout();
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Generate improved layout with RL
  const generateImprovedLayout = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);

      const response = await fetch("/api/engine/generate-improved", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          building_id: buildingId,
          program_id: programId,
          reference_layout_id: layoutId,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || "Failed to generate improved layout"
        );
      }

      const data = await response.json();

      // Update with the new layout
      setLayoutId(data.layout_id);
      setInitialLayout(data.layout_data);
      setModifiedLayout(data.layout_data);
      setSuccess("Improved layout generated successfully using RL");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="interactive-layout-page">
      <header>
        <h1>Interactive Hotel Design</h1>
        <p>
          Building: {buildingId} | Program: {programId}
        </p>
      </header>

      <div className="tab-navigation">
        <button
          className={`tab-button ${activeTab === "grid" ? "active" : ""}`}
          onClick={() => setActiveTab("grid")}
        >
          1. Define Standard Floors
        </button>
        <button
          className={`tab-button ${activeTab === "layout" ? "active" : ""}`}
          onClick={() => setActiveTab("layout")}
          disabled={!initialLayout}
        >
          2. Modify Layout
        </button>
      </div>

      {error && (
        <div className="error-message">
          <p>{error}</p>
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      {success && (
        <div className="success-message">
          <p>{success}</p>
          <button onClick={() => setSuccess(null)}>Dismiss</button>
        </div>
      )}

      <div className="tab-content">
        {activeTab === "grid" && (
          <div className="grid-tab">
            <p className="description">
              Define which areas of the building will contain standard floors
              (typically guest rooms and circulation cores). These will be
              repeated from the start floor to the end floor.
            </p>

            {buildingConfig && (
              <GridSelector
                buildingWidth={buildingConfig.width}
                buildingLength={buildingConfig.length}
                gridSize={buildingConfig.structural_grid_x}
                onSelectionChange={handleSelectionChange}
              />
            )}

            <div className="action-buttons">
              <button
                className="btn-secondary"
                onClick={() => navigate("/configure")}
              >
                Back to Configuration
              </button>

              <button
                className="btn-primary"
                onClick={saveStandardFloorZones}
                disabled={selectedAreas.length === 0 || loading}
              >
                Save Standard Floor Zones
              </button>

              <button
                className="btn-success"
                onClick={generateWithZones}
                disabled={loading}
              >
                Generate Layout
              </button>
            </div>
          </div>
        )}

        {activeTab === "layout" && (
          <div className="layout-tab">
            <p className="description">
              Modify the layout by dragging rooms to new positions. When
              satisfied, provide feedback to train the AI.
            </p>

            {initialLayout && buildingConfig && (
              <LayoutEditor
                initialLayout={initialLayout}
                buildingConfig={buildingConfig}
                onLayoutChange={handleLayoutChange}
                onTrainRL={handleTrainRL}
              />
            )}

            <div className="action-buttons">
              <button
                className="btn-secondary"
                onClick={() => setActiveTab("grid")}
              >
                Back to Grid Selector
              </button>

              <button
                className="btn-primary"
                onClick={saveModifiedLayout}
                disabled={!modifiedLayout || loading}
              >
                Save Modified Layout
              </button>

              <button
                className="btn-success"
                onClick={generateImprovedLayout}
                disabled={!layoutId || loading}
              >
                Generate Improved Layout with RL
              </button>
            </div>
          </div>
        )}
      </div>

      {loading && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <p>Processing, please wait...</p>
        </div>
      )}
    </div>
  );
};

export default InteractiveLayoutPage;
