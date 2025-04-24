import React, { useState, useEffect } from "react";
import { useParams, useNavigate, useLocation } from "react-router-dom";
import GridSelector from "../components/GridSelector";
import LayoutEditor from "../components/LayoutEditor";
import {
  generateLayout,
  getLayout,
  modifyLayout,
  getConfiguration,
} from "../services/api";
import "../styles/InteractiveLayout.css";

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
  const location = useLocation();

  const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

  const [standardFloorParams, setStandardFloorParams] = useState({
    start_floor: 2,
    end_floor: 20,
    width: 56,
    length: 20,
    position_x: 0,
    position_y: 32,
    corridor_width: 4,
    room_depth: 8,
  });

  // Get layoutId from query params if it exists
  useEffect(() => {
    const queryParams = new URLSearchParams(location.search);
    const layoutIdParam = queryParams.get("layoutId");
    if (layoutIdParam) {
      setLayoutId(layoutIdParam);
    }
  }, [location]);

  // Add useEffect to load initial values from building config
  useEffect(() => {
    if (buildingConfig && buildingConfig.standard_floor) {
      setStandardFloorParams({
        start_floor: buildingConfig.standard_floor.start_floor || 2,
        end_floor: buildingConfig.standard_floor.end_floor || 20,
        width: buildingConfig.standard_floor.width || 56,
        length: buildingConfig.standard_floor.length || 20,
        position_x: buildingConfig.standard_floor.position_x || 0,
        position_y: buildingConfig.standard_floor.position_y || 32,
        corridor_width: buildingConfig.standard_floor.corridor_width || 4,
        room_depth: buildingConfig.standard_floor.room_depth || 8,
      });
    }
  }, [buildingConfig]);

  // Fetch building configuration
  useEffect(() => {
    const fetchBuildingConfig = async () => {
      try {
        // If using sample data, load default config
        if (buildingId === "sample" && programId === "sample") {
          const response = await getConfiguration("building", buildingId);
          if (response.success) {
            setBuildingConfig({
              width: response.config_data.width,
              length: response.config_data.length,
              height: response.config_data.height,
              min_floor: response.config_data.min_floor,
              max_floor: response.config_data.max_floor,
              floor_height: response.config_data.floor_height,
              structural_grid_x: response.config_data.structural_grid_x,
              structural_grid_y: response.config_data.structural_grid_y,
              grid_size: response.config_data.grid_size,
            });
          }
          setInitialLayout({
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
          });
          return;
        }

        // For real data, fetch from backend
        setLoading(true);

        // If we have a layoutId, fetch that specific layout
        if (layoutId) {
          const layoutData = await getLayout(layoutId);
          if (layoutData.success) {
            setBuildingConfig({
              width: layoutData.layout_data.width || 60,
              length: layoutData.layout_data.length || 80,
              height: layoutData.layout_data.height || 30,
              min_floor: -2,
              max_floor: 5,
              floor_height: 4.5,
              structural_grid_x: 8.0,
              structural_grid_y: 8.0,
              grid_size: 1.0,
            });
            setInitialLayout(layoutData.layout_data);

            // Switch to layout editor tab since we're loading an existing layout
            setActiveTab("layout");
          }
        } else {
          // Otherwise use configuration info from params
          // In a real app, we'd fetch these from the backend
          setBuildingConfig({
            width: 60.0,
            length: 80.0,
            height: 30.0,
            min_floor: -2,
            max_floor: 5,
            floor_height: 4.5,
            structural_grid_x: 8.0,
            structural_grid_y: 8.0,
            grid_size: 1.0,
            main_entry: "front",
            description: "Hotel building generated from parameters",
            units: "meters",
          });
        }

        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchBuildingConfig();
  }, [buildingId, programId, layoutId]);

  // Handle grid selection changes
  const handleSelectionChange = (areas) => {
    setSelectedAreas(areas);

    // If areas selected, update standard floor position based on first selection
    if (areas.length > 0) {
      // Find the top-left corner of the selection
      let minX = Infinity;
      let minY = Infinity;

      areas.forEach((area) => {
        minX = Math.min(minX, area.x);
        minY = Math.min(minY, area.y);
      });

      // Update position_x and position_y in standardFloorParams
      setStandardFloorParams((prev) => ({
        ...prev,
        position_x: minX,
        position_y: minY,
      }));
    }
  };

  const updateBuildingConfig = async () => {
    try {
      setLoading(true);
      setError(null);

      // Create updated building config
      const updatedConfig = {
        ...buildingConfig,
        standard_floor: {
          ...standardFloorParams,
        },
      };

      // Call API to update building config
      const response = await fetch(`${API_BASE_URL}/update-building-config`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          building_id: buildingId,
          building_config: updatedConfig,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to update building configuration");
      }

      const result = await response.json();

      // Update state with new config
      setBuildingConfig(updatedConfig);
      setSuccess("Building configuration updated successfully");
    } catch (err) {
      setError(err.message || "Error updating building configuration");
    } finally {
      setLoading(false);
    }
  };

  // Change the saveStandardFloorZones function to incorporate your new parameters
  const saveStandardFloorZones = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);

      // Create the data to send to the backend
      const standardFloorData = {
        session_id: buildingId,
        floor_zones: selectedAreas,
        start_floor: standardFloorParams.start_floor,
        end_floor: standardFloorParams.end_floor,
        standard_floor: {
          ...standardFloorParams,
        },
      };

      // Call updateBuildingConfig instead of using the old implementation
      await updateBuildingConfig(buildingId, {
        ...buildingConfig,
        standard_floor: standardFloorParams,
      });

      setSuccess("Standard floor zones saved successfully");
    } catch (err) {
      setError(err.message || "Error saving standard floor zones");
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

      // Call the backend to generate a layout
      const result = await generateLayout({
        building_config: buildingId,
        program_config: programId,
        mode: "rule",
        include_standard_floors: true,
        fixed_positions: {}, // We can add fixed positions here if needed
      });

      if (result.success) {
        setLayoutId(result.layout_id);
        setInitialLayout({
          rooms: result.rooms,
        });
        setModifiedLayout({
          rooms: result.rooms,
        });
        setSuccess("Layout generated successfully");

        // Switch to layout editor tab
        setActiveTab("layout");
      } else {
        throw new Error("Failed to generate layout");
      }

      setLoading(false);
    } catch (err) {
      setError(err.message);
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

      // Find which rooms were moved
      const movedRooms = [];

      if (initialLayout && modifiedLayout) {
        for (const roomId in modifiedLayout.rooms) {
          if (initialLayout.rooms[roomId]) {
            const originalPos = initialLayout.rooms[roomId].position;
            const newPos = modifiedLayout.rooms[roomId].position;

            // Check if position changed
            if (
              originalPos[0] !== newPos[0] ||
              originalPos[1] !== newPos[1] ||
              originalPos[2] !== newPos[2]
            ) {
              movedRooms.push({
                room_id: parseInt(roomId),
                new_position: newPos,
              });
            }
          }
        }
      }

      // For each moved room, call the modify API
      for (const movedRoom of movedRooms) {
        await modifyLayout({
          layout_id: layoutId,
          room_id: movedRoom.room_id,
          new_position: movedRoom.new_position,
        });
      }

      setSuccess(
        `Layout modifications saved successfully (${movedRooms.length} rooms updated)`
      );
      setLoading(false);
    } catch (err) {
      setError(err.message);
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

      // In a real app, we'd send feedback to the backend for RL training

      setSuccess("Feedback submitted for RL training");
      setLoading(false);

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
      setLoading(false);
    }
  };

  // Generate improved layout with RL
  const generateImprovedLayout = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);

      // In a real app, we'd call the backend to generate an improved layout

      setSuccess("Improved layout generation would happen here");
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  return (
    <div className="interactive-layout-page">
      <header>
        <h1>Interactive Hotel Design</h1>
        <p>
          Building: {buildingId} | Program: {programId}
          {layoutId && <span> | Layout: {layoutId}</span>}
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
            {/* Add Standard Floor Parameters Form */}
            <div className="standard-floor-params">
              <h3>Standard Floor Parameters</h3>
              <div className="form-row">
                <div className="half">
                  <label>Start Floor</label>
                  <input
                    type="number"
                    value={standardFloorParams.start_floor}
                    onChange={(e) =>
                      setStandardFloorParams({
                        ...standardFloorParams,
                        start_floor: parseInt(e.target.value),
                      })
                    }
                  />
                </div>
                <div className="half">
                  <label>End Floor</label>
                  <input
                    type="number"
                    value={standardFloorParams.end_floor}
                    onChange={(e) =>
                      setStandardFloorParams({
                        ...standardFloorParams,
                        end_floor: parseInt(e.target.value),
                      })
                    }
                  />
                </div>
              </div>

              <div className="form-row">
                <div className="half">
                  <label>Width (m)</label>
                  <input
                    type="number"
                    value={standardFloorParams.width}
                    onChange={(e) =>
                      setStandardFloorParams({
                        ...standardFloorParams,
                        width: parseFloat(e.target.value),
                      })
                    }
                  />
                </div>
                <div className="half">
                  <label>Length (m)</label>
                  <input
                    type="number"
                    value={standardFloorParams.length}
                    onChange={(e) =>
                      setStandardFloorParams({
                        ...standardFloorParams,
                        length: parseFloat(e.target.value),
                      })
                    }
                  />
                </div>
              </div>

              <div className="form-row">
                <div className="half">
                  <label>Corridor Width (m)</label>
                  <input
                    type="number"
                    value={standardFloorParams.corridor_width}
                    onChange={(e) =>
                      setStandardFloorParams({
                        ...standardFloorParams,
                        corridor_width: parseFloat(e.target.value),
                      })
                    }
                  />
                </div>
                <div className="half">
                  <label>Room Depth (m)</label>
                  <input
                    type="number"
                    value={standardFloorParams.room_depth}
                    onChange={(e) =>
                      setStandardFloorParams({
                        ...standardFloorParams,
                        room_depth: parseFloat(e.target.value),
                      })
                    }
                  />
                </div>
              </div>
            </div>

            <div className="action-buttons">
              <button
                className="btn-secondary"
                onClick={() => navigate("/configure")}
              >
                Back to Configuration
              </button>

              <button
                className="btn-primary"
                // onClick={updateBuildingConfig}
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
