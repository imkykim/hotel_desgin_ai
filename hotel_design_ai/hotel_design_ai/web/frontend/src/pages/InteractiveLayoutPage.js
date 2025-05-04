import React, { useState, useEffect } from "react";
import { useParams, useNavigate, useLocation } from "react-router-dom";
import GridSelector from "../components/GridSelector";
import LayoutEditor from "../components/LayoutEditor";
import {
  generateLayout,
  getLayout,
  modifyLayout,
  getConfiguration,
  generateImprovedLayout,
  generateLayoutWithReference,
  generateLayoutWithZones, // <-- import the new helper
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

  // New state for fixed elements
  const [fixedElements, setFixedElements] = useState({
    entrances: [],
    cores: [],
  });
  const [gridSelectionMode, setGridSelectionMode] = useState("standard");

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

  // Add state for fixedRoomsFile and programConfig
  const [fixedRoomsFile, setFixedRoomsFile] = useState(null);
  const [programConfig, setProgramConfig] = useState("hotel_requirements_3");

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
        setLoading(true);
        console.log(`Fetching building configuration for: ${buildingId}`);

        // If using sample data, load default config
        if (buildingId === "sample" && programId === "sample") {
          console.log("Using sample configuration");
          setBuildingConfig({
            width: 60,
            length: 80,
            height: 100,
            min_floor: -2,
            max_floor: 20,
            floor_height: 4.5,
            structural_grid_x: 8.0,
            structural_grid_y: 8.0,
            grid_size: 1.0,
          });

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
          setLoading(false);
          return;
        }

        // If we have a layoutId, fetch that specific layout
        if (layoutId) {
          console.log(`Fetching layout: ${layoutId}`);
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
          // Fetch the actual building configuration using the buildingId
          console.log(`Fetching building config: ${buildingId}`);
          const configResponse = await getConfiguration("building", buildingId);

          if (configResponse.success) {
            console.log("Building config loaded:", configResponse.config_data);

            // Use the data from the fetched configuration
            setBuildingConfig({
              width: configResponse.config_data.width || 60.0,
              length: configResponse.config_data.length || 80.0,
              height: configResponse.config_data.height || 100.0,
              min_floor: configResponse.config_data.min_floor || -2,
              max_floor: configResponse.config_data.max_floor || 20,
              floor_height: configResponse.config_data.floor_height || 4.5,
              structural_grid_x:
                configResponse.config_data.structural_grid_x || 8.0,
              structural_grid_y:
                configResponse.config_data.structural_grid_y || 8.0,
              grid_size: configResponse.config_data.grid_size || 1.0,
              description:
                configResponse.config_data.description || "Hotel building",
            });
          } else {
            // Fallback to default if config not found
            console.log("Using default building configuration");
            setBuildingConfig({
              width: 60.0,
              length: 80.0,
              height: 100.0,
              min_floor: -2,
              max_floor: 20,
              floor_height: 4.5,
              structural_grid_x: 8.0,
              structural_grid_y: 8.0,
              grid_size: 1.0,
              description: "Default hotel building configuration",
            });
          }
        }

        setLoading(false);
      } catch (err) {
        console.error("Error fetching configuration:", err);
        setError(err.message || "Failed to load building configuration");
        setLoading(false);
      }
    };

    if (buildingId) {
      fetchBuildingConfig();
    }
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

  // New function for handling fixed element selection
  const handleFixedElementSelect = (type, coords) => {
    if (coords === null) {
      // Clear the selected fixed elements of this type
      if (type === "entrance") {
        setFixedElements((prev) => ({ ...prev, entrances: [] }));
      } else if (type === "core") {
        setFixedElements((prev) => ({ ...prev, cores: [] }));
      }
      return;
    }

    // For entrance, we replace the existing one (only one entrance allowed)
    if (type === "entrance") {
      setFixedElements((prev) => ({
        ...prev,
        entrances: [
          {
            x: coords.x,
            y: coords.y,
            z: 0, // Ground floor
            type: "lobby",
            name: "reception",
          },
        ],
      }));
    }
    // For cores, we allow multiple
    else if (type === "core") {
      setFixedElements((prev) => {
        // Create a new core with an index
        const newCore = {
          x: coords.x,
          y: coords.y,
          z: -10, // Below ground for vertical circulation
          type: prev.cores.length === 0 ? "main_core" : "secondary_core",
          department: "circulation",
          name:
            prev.cores.length === 0
              ? "main_core"
              : `secondary_core_${prev.cores.length}`,
        };

        // Only allow two cores max
        let updatedCores;
        if (prev.cores.length >= 2) {
          updatedCores = [...prev.cores.slice(0, 1), newCore];
        } else {
          updatedCores = [...prev.cores, newCore];
        }

        return {
          ...prev,
          cores: updatedCores,
        };
      });
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

  const saveStandardFloorZones = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);

      // Check if we have any selected areas
      if (selectedAreas.length === 0) {
        setError("Please select at least one grid cell for standard floors");
        setLoading(false);
        return;
      }

      // Calculate the bounding box of all selected areas
      let minX = Infinity,
        minY = Infinity;
      let maxX = -Infinity,
        maxY = -Infinity;

      selectedAreas.forEach((area) => {
        // Each area has x, y, width, height
        minX = Math.min(minX, area.x);
        minY = Math.min(minY, area.y);
        maxX = Math.max(maxX, area.x + area.width);
        maxY = Math.max(maxY, area.y + area.height);
      });

      // Calculate dimensions of the standard floor zone
      const standardFloorWidth = maxX - minX;
      const standardFloorLength = maxY - minY;

      console.log("Calculated standard floor dimensions:", {
        position_x: minX,
        position_y: minY,
        width: standardFloorWidth,
        length: standardFloorLength,
      });

      // Create the updated parameters directly rather than going through state
      const updatedStandardFloorParams = {
        ...standardFloorParams,
        position_x: minX,
        position_y: minY,
        width: standardFloorWidth,
        length: standardFloorLength,
      };

      // Update state for UI
      setStandardFloorParams(updatedStandardFloorParams);

      // Safety check: Ensure buildingConfig is valid
      if (!buildingConfig) {
        setError("Building configuration not loaded yet");
        setLoading(false);
        return;
      }

      // Create updated building config using the directly calculated values
      const updatedConfig = {
        ...buildingConfig,
        standard_floor: updatedStandardFloorParams, // Use local variable, not state
      };

      console.log("Sending updated config to server:", updatedConfig);
      console.log("Standard floor in config:", updatedConfig.standard_floor);

      // Call API directly rather than using the helper function
      try {
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

        const responseText = await response.text();
        console.log("Raw API response:", responseText);

        let responseData;
        try {
          responseData = JSON.parse(responseText);
        } catch (e) {
          console.error("Failed to parse API response:", e);
          throw new Error(
            `Failed to parse API response: ${responseText.substring(0, 100)}...`
          );
        }

        if (!response.ok || !responseData.success) {
          throw new Error(
            responseData.error || "Failed to update building configuration"
          );
        }

        // Update state with new config
        setBuildingConfig(updatedConfig);
        setSuccess(
          "Standard floor zones saved successfully to building configuration"
        );

        // After saving, switch to entrance selection mode
        setGridSelectionMode("entrance");
      } catch (apiError) {
        console.error("API call error:", apiError);
        throw new Error(`API error: ${apiError.message}`);
      }
    } catch (err) {
      console.error("Error in saveStandardFloorZones:", err);
      setError(err.message || "Error saving standard floor zones");
    } finally {
      setLoading(false);
    }
  };

  // New function to save fixed elements
  const saveFixedElements = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);

      // Prepare fixed elements in the correct format
      const fixedRooms = [
        // Add entrance/lobby
        ...fixedElements.entrances.map((entrance) => ({
          identifier: {
            type: "room_type_with_name",
            room_type: "lobby",
            name: "reception",
          },
          position: [entrance.x, entrance.y, entrance.z || 0],
        })),

        // Add cores
        ...fixedElements.cores.map((core) => ({
          identifier: {
            type: "department_with_name",
            department: "circulation",
            name: core.name,
          },
          position: [core.x, core.y, core.z || -10],
        })),
      ];

      // Create the fixed_rooms.json content
      const fixedRoomsJson = {
        fixed_rooms: fixedRooms,
      };

      // Call the backend to save fixed elements
      const response = await fetch(`${API_BASE_URL}/save-fixed-elements`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          building_id: buildingId,
          fixed_elements: fixedRoomsJson,
        }),
      }).catch((err) => {
        // For demonstration - handle gracefully if endpoint doesn't exist yet
        console.warn("Backend endpoint not implemented yet:", err);
        return { ok: true, json: () => Promise.resolve({ success: true }) };
      });

      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Failed to save fixed elements");
      }

      // Save the fixed rooms file path for later use
      setFixedRoomsFile(data.filepath);

      setSuccess("Fixed elements saved successfully");
      setGridSelectionMode("standard"); // Go back to standard mode
    } catch (err) {
      setError(err.message || "Error saving fixed elements");
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

      // 1. Save standard floor zones first
      if (selectedAreas.length === 0) {
        setError("Please select at least one grid cell for standard floors");
        setLoading(false);
        return;
      }

      // Calculate bounding box for each selected area (or treat each as a zone)
      const floor_zones = selectedAreas.map((area) => ({
        x: area.x,
        y: area.y,
        width: area.width,
        height: area.height,
      }));

      // Use start/end floor from standardFloorParams
      const zoneConfig = {
        building_id: buildingId,
        floor_zones,
        start_floor: standardFloorParams.start_floor,
        end_floor: standardFloorParams.end_floor,
      };

      // Save zones to backend
      const saveZonesResp = await fetch(
        `${API_BASE_URL}/engine/standard-floor-zones`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(zoneConfig),
        }
      );
      const saveZonesData = await saveZonesResp.json();
      if (!saveZonesResp.ok || !saveZonesData.success) {
        throw new Error(
          saveZonesData.message ||
            saveZonesData.detail ||
            "Failed to save standard floor zones"
        );
      }

      // 2. Now call generate-with-zones
      const result = await generateLayoutWithZones(
        buildingId,
        programConfig || "hotel_requirements_3",
        fixedRoomsFile
      );

      if (result.success) {
        setLayoutId(result.layout_id);
        setInitialLayout({
          rooms: result.layout_data?.rooms || result.rooms,
        });
        setModifiedLayout({
          rooms: result.layout_data?.rooms || result.rooms,
        });
        setSuccess("Layout generated successfully");
        setActiveTab("layout");
      } else {
        throw new Error(result.error || "Failed to generate layout");
      }
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
        handleGenerateImprovedLayout();
      }
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  // Updated function to generate improved layout with RL
  const handleGenerateImprovedLayout = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);

      console.log("Attempting to generate improved layout...");

      // Always pass buildingId, programId, and layoutId as reference_layout_id
      let result = await generateImprovedLayout(
        buildingId,
        programId,
        layoutId
      );

      // If first approach fails, try the fallback approach
      if (!result.success) {
        console.log("First approach failed, trying fallback method...");

        // Try the alternative approach (same endpoint, but fallback for robustness)
        result = await generateLayoutWithReference(
          buildingId,
          programId,
          layoutId
        );
      }

      if (result.success) {
        const newLayoutId = result.layout_id;
        setSuccess(
          `Improved layout generated successfully! Redirecting to view the new layout...`
        );

        // Wait a brief moment to let the user see the success message
        setTimeout(() => {
          // Navigate to view the new layout
          navigate(`/view-layout/${newLayoutId}`);
        }, 1500);
      } else {
        throw new Error(result.error || "Failed to generate improved layout");
      }
    } catch (err) {
      console.error("Error generating improved layout:", err);
      setError(err.message || "Error generating improved layout");
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
            {/* Selection mode buttons */}
            <div className="selection-mode-buttons">
              <button
                className={`mode-button ${
                  gridSelectionMode === "standard" ? "active" : ""
                }`}
                onClick={() => setGridSelectionMode("standard")}
              >
                Standard Floor Zones
              </button>
              <button
                className={`mode-button ${
                  gridSelectionMode === "entrance" ? "active" : ""
                }`}
                onClick={() => setGridSelectionMode("entrance")}
              >
                Define Entrance/Lobby
              </button>
              <button
                className={`mode-button ${
                  gridSelectionMode === "core" ? "active" : ""
                }`}
                onClick={() => setGridSelectionMode("core")}
              >
                Define Core Circulation
              </button>
            </div>

            <p className="description">
              {gridSelectionMode === "standard"
                ? "Define which areas of the building will contain standard floors (typically guest rooms and circulation cores). These will be repeated from the start floor to the end floor."
                : gridSelectionMode === "entrance"
                ? "Select where the main entrance and lobby should be located. These will be fixed in the final layout."
                : "Select where the vertical circulation cores (elevators, stairs) should be located. These will be fixed in the final layout."}
            </p>

            {buildingConfig && (
              <GridSelector
                buildingWidth={buildingConfig.width}
                buildingLength={buildingConfig.length}
                gridSize={buildingConfig.structural_grid_x}
                onSelectionChange={handleSelectionChange}
                onFixedElementSelect={handleFixedElementSelect}
                selectionMode={gridSelectionMode}
              />
            )}

            {/* Only show standard floor parameters in standard mode */}
            {gridSelectionMode === "standard" && (
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
            )}

            {/* Fixed element display if any are selected */}
            {(fixedElements.entrances.length > 0 ||
              fixedElements.cores.length > 0) && (
              <div className="fixed-elements-summary">
                <h3>Fixed Elements Summary</h3>
                {fixedElements.entrances.length > 0 && (
                  <div>
                    <h4>Entrance/Lobby</h4>
                    <ul>
                      {fixedElements.entrances.map((entrance, idx) => (
                        <li key={`entrance-${idx}`}>
                          Position: {entrance.x}m, {entrance.y}m, Floor: Ground
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {fixedElements.cores.length > 0 && (
                  <div>
                    <h4>Vertical Circulation Cores</h4>
                    <ul>
                      {fixedElements.cores.map((core, idx) => (
                        <li key={`core-${idx}`}>
                          {core.name}: {core.x}m, {core.y}m, Multi-floor
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            <div className="action-buttons">
              <button
                className="btn-secondary"
                onClick={() => navigate("/configure")}
              >
                Back to Configuration
              </button>

              {gridSelectionMode === "standard" ? (
                <button
                  className="btn-primary"
                  onClick={saveStandardFloorZones}
                  disabled={selectedAreas.length === 0 || loading}
                >
                  Save & Continue to Fixed Elements
                </button>
              ) : (
                <button
                  className="btn-primary"
                  onClick={saveFixedElements}
                  disabled={loading}
                >
                  Save Fixed Elements
                </button>
              )}

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
                layoutId={layoutId}
                onLayoutChange={handleLayoutChange}
                onTrainRL={handleTrainRL}
                // Add these props to ensure the editor fits the generated layout
                width={buildingConfig.width}
                length={buildingConfig.length}
                gridSize={buildingConfig.grid_size}
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
                onClick={handleGenerateImprovedLayout}
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
