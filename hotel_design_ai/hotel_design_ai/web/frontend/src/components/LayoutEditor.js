import React, { useState, useEffect, useRef } from "react";
import "../styles/LayoutEditor.css";

// Room color configuration
const ROOM_COLORS = {
  entrance: "#2c7fb8",
  lobby: "#7fcdbb",
  vertical_circulation: "#FF0000",
  restaurant: "#f0f9e8",
  meeting_room: "#edf8b1",
  guest_room: "#f7fcb9",
  service_area: "#d9f0a3",
  back_of_house: "#addd8e",
  parking: "#D3D3D3",
  mechanical: "#A0A0A0",
  maintenance: "#909090",
  default: "#efefef",
};

// Room transparency configuration
const ROOM_ALPHAS = {
  entrance: 0.9,
  lobby: 0.7,
  vertical_circulation: 0.8,
  restaurant: 0.7,
  meeting_room: 0.7,
  guest_room: 0.6,
  service_area: 0.5,
  back_of_house: 0.5,
  parking: 0.5,
  mechanical: 0.6,
  maintenance: 0.6,
  default: 0.5,
};

// Room information display component
const RoomInfo = ({ room }) => {
  if (!room)
    return <p>No room selected. Click on a room to view its details.</p>;

  return (
    <>
      <h4>Selected Room</h4>
      <p>ID: {room.id}</p>
      <p>Type: {room.type}</p>
      <p>Name: {room.metadata?.name || "Unnamed"}</p>
      <p>Position: ({room.position.map((v) => v.toFixed(1)).join(", ")})</p>
      <p>Dimensions: {room.dimensions.map((v) => v.toFixed(1)).join(" × ")}m</p>
    </>
  );
};

// Floor selector component
const FloorSelector = ({ buildingConfig, currentFloor, onFloorChange }) => {
  // Generate floors based on building config
  const generateFloors = () => {
    const floors = [];
    const minFloor = buildingConfig?.min_floor || -1;
    const maxFloor = buildingConfig?.max_floor || 3;

    for (let f = minFloor; f <= maxFloor; f++) {
      floors.push(f);
    }

    return floors;
  };

  return (
    <div className="floor-selector">
      <label>Floor: </label>
      {generateFloors().map((floor) => (
        <button
          key={`floor-${floor}`}
          className={`floor-button ${currentFloor === floor ? "active" : ""}`}
          onClick={() => onFloorChange(floor)}
        >
          {floor < 0 ? `B${Math.abs(floor)}` : floor}
        </button>
      ))}
      {/* Add a 3D view button */}
      <button
        key="floor-3d"
        className={`floor-button ${currentFloor === "3d" ? "active" : ""}`}
        onClick={() => onFloorChange("3d")}
      >
        3D View
      </button>
    </div>
  );
};

// Main layout editor component
const LayoutEditor = ({
  initialLayout,
  buildingConfig,
  layoutId,
  onLayoutChange = () => {},
  onTrainRL = () => {},
  width,
  length,
  gridSize,
}) => {
  const [layout, setLayout] = useState(initialLayout || { rooms: {} });
  const [selectedRoom, setSelectedRoom] = useState(null);
  const [currentFloor, setCurrentFloor] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [userFeedback, setUserFeedback] = useState(5); // 0-10 rating
  const [showFloorPlan, setShowFloorPlan] = useState(true);
  const [show3DView, setShow3DView] = useState(false);

  // Define refs and state for the canvas
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const apiBaseUrl = process.env.REACT_APP_API_URL || "http://localhost:8000";
  // Use calculated dimensions based on props, not internal state
  // This ensures we always use the latest values from props
  const buildingWidth = width || (buildingConfig && buildingConfig.width) || 80;
  const buildingLength =
    length || (buildingConfig && buildingConfig.length) || 120;
  const floorHeight = (buildingConfig && buildingConfig.floor_height) || 4.0;

  // Debug dimensions - remove this in production
  useEffect(() => {
    console.log("LayoutEditor dimensions:", {
      buildingWidth,
      buildingLength,
      floorHeight,
      gridSize,
    });
    console.log("Original props:", {
      width,
      length,
      "buildingConfig.width": buildingConfig?.width,
      "buildingConfig.length": buildingConfig?.length,
    });
  }, [
    buildingWidth,
    buildingLength,
    floorHeight,
    gridSize,
    width,
    length,
    buildingConfig,
  ]);

  // Sync layout state with initialLayout prop when it changes
  useEffect(() => {
    if (initialLayout) {
      setLayout(initialLayout);
      console.log("Layout updated from initialLayout prop");
    }
  }, [initialLayout]);

  // Calculate canvas dimensions when container size or building dimensions change
  const [canvasDimensions, setCanvasDimensions] = useState({
    width: 800,
    height: 600,
  });

  useEffect(() => {
    const calculateCanvasDimensions = () => {
      if (!containerRef.current) return { width: 800, height: 600 };

      // Get container width (with some margin)
      const containerWidth = containerRef.current.clientWidth - 40;

      // Calculate aspect ratio
      const aspectRatio = buildingWidth / buildingLength;

      // Calculate height based on width and aspect ratio
      const height = containerWidth / aspectRatio;

      // Limit height to a reasonable value
      const maxHeight = window.innerHeight * 0.6;

      if (height > maxHeight) {
        // If too tall, constrain by height instead
        return {
          width: maxHeight * aspectRatio,
          height: maxHeight,
        };
      }

      return {
        width: containerWidth,
        height: height,
      };
    };

    // Set initial dimensions
    setCanvasDimensions(calculateCanvasDimensions());

    // Add window resize listener
    const handleResize = () => {
      setCanvasDimensions(calculateCanvasDimensions());
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [buildingWidth, buildingLength]);

  // Function to draw the layout on the canvas
  const drawLayout = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate the scale factor to fit the building in the canvas
    // Add a padding of 10px on each side
    const padding = 20;
    const availableWidth = canvas.width - padding * 2;
    const availableHeight = canvas.height - padding * 2;

    const scaleX = availableWidth / buildingWidth;
    const scaleY = availableHeight / buildingLength;

    // Use the smaller scale to ensure the entire building fits
    const scale = Math.min(scaleX, scaleY);

    // Calculate offsets to center the building
    const offsetX = padding + (availableWidth - buildingWidth * scale) / 2;
    const offsetY = padding + (availableHeight - buildingLength * scale) / 2;

    // Draw building outline
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.strokeRect(
      offsetX,
      offsetY,
      buildingWidth * scale,
      buildingLength * scale
    );

    // Draw grid lines if gridSize is provided
    if (gridSize) {
      ctx.strokeStyle = "#eee";
      ctx.lineWidth = 0.5;

      // Draw vertical grid lines
      for (let x = 0; x <= buildingWidth; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(offsetX + x * scale, offsetY);
        ctx.lineTo(offsetX + x * scale, offsetY + buildingLength * scale);
        ctx.stroke();
      }

      // Draw horizontal grid lines
      for (let y = 0; y <= buildingLength; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(offsetX, offsetY + y * scale);
        ctx.lineTo(offsetX + buildingWidth * scale, offsetY + y * scale);
        ctx.stroke();
      }
    }

    // Draw rooms for the current floor
    let renderedRoomCount = 0;
    const renderedTypes = new Set();

    // First pass: Draw regular rooms for the current floor
    Object.entries(layout.rooms || {}).forEach(([roomId, roomData]) => {
      // Skip if we're in 3D view mode
      if (currentFloor === "3d") return;

      const roomFloor = Math.floor(roomData.position[2] / floorHeight);
      const roomType = roomData.type;

      // If this is vertical circulation, we'll handle it in the second pass
      if (roomType === "vertical_circulation") return;

      // Skip rooms not on the current floor
      if (roomFloor !== currentFloor) return;

      // Draw the room with proper scaling and offset
      drawRoom(ctx, parseInt(roomId), roomData, scale, offsetX, offsetY);
      renderedRoomCount++;
      renderedTypes.add(roomData.type);
    });

    // Second pass: Draw vertical circulation cores that should appear on all floors
    Object.entries(layout.rooms || {}).forEach(([roomId, roomData]) => {
      // Skip if we're in 3D view mode
      if (currentFloor === "3d") return;

      // Only process vertical circulation rooms on any floor
      if (roomData.type !== "vertical_circulation") return;

      // Always draw vertical circulation on all floors
      if (
        currentFloor >= buildingConfig?.min_floor &&
        currentFloor <= buildingConfig?.max_floor
      ) {
        drawRoom(ctx, parseInt(roomId), roomData, scale, offsetX, offsetY);
        renderedRoomCount++;
        renderedTypes.add(roomData.type);
      }
    });

    console.log(`Rendered ${renderedRoomCount} rooms on floor ${currentFloor}`);

    // Store the current transformation for mouse events
    canvas.transformData = { scale, offsetX, offsetY };
  };

  // Draw an individual room
  const drawRoom = (ctx, roomId, roomData, scale, offsetX, offsetY) => {
    const [x, y] = roomData.position;
    const [w, l] = roomData.dimensions;
    const roomType = roomData.type;

    // Calculate screen coordinates
    const screenX = offsetX + x * scale;
    const screenY = offsetY + y * scale;
    const screenW = w * scale;
    const screenH = l * scale;

    // Get room styling
    ctx.fillStyle = ROOM_COLORS[roomType] || ROOM_COLORS.default;
    ctx.strokeStyle = selectedRoom === roomId ? "#ff0000" : "#000";
    ctx.lineWidth = selectedRoom === roomId ? 2 : 1;

    // Draw room rectangle
    ctx.fillRect(screenX, screenY, screenW, screenH);
    ctx.strokeRect(screenX, screenY, screenW, screenH);

    // Draw room label
    ctx.fillStyle = "#000";
    ctx.font = "12px Arial";
    ctx.textAlign = "center";

    const name = roomData.metadata?.name || roomType;
    ctx.fillText(
      name.length > 15 ? name.substring(0, 12) + "..." : name,
      screenX + screenW / 2,
      screenY + screenH / 2
    );
  };

  // Draw the layout when component updates
  useEffect(() => {
    if (currentFloor === "3d") {
      setShowFloorPlan(false);
      setShow3DView(true);
    } else {
      setShowFloorPlan(true);
      setShow3DView(false);
      drawLayout();
    }
  }, [
    layout,
    selectedRoom,
    currentFloor,
    canvasDimensions.width,
    canvasDimensions.height,
    // Be explicit about dependencies - include the values directly, not state variables
    buildingWidth,
    buildingLength,
    floorHeight,
    gridSize,
  ]);

  // Handle floor change
  const handleFloorChange = (floor) => {
    setCurrentFloor(floor);
    setSelectedRoom(null);
  };

  // Handle mouse down for room selection and dragging
  const handleMouseDown = (e) => {
    // Don't handle mouse events when in 3D view
    if (currentFloor === "3d") return;

    const canvas = canvasRef.current;
    if (!canvas || !canvas.transformData) return;

    const rect = canvas.getBoundingClientRect();
    const { scale, offsetX, offsetY } = canvas.transformData;

    // Convert screen coordinates to world coordinates
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;

    // Convert to world coordinates
    const worldX = (screenX - offsetX) / scale;
    const worldY = (screenY - offsetY) / scale;

    // Find room under mouse
    let foundRoom = null;

    Object.entries(layout.rooms || {}).forEach(([roomId, data]) => {
      const roomFloor = Math.floor(data.position[2] / floorHeight);
      if (roomFloor !== currentFloor) return;

      const [x, y] = data.position;
      const [w, l] = data.dimensions;

      if (worldX >= x && worldX <= x + w && worldY >= y && worldY <= y + l) {
        foundRoom = parseInt(roomId);
        setDragOffset({
          x: worldX - x,
          y: worldY - y,
        });
      }
    });

    setSelectedRoom(foundRoom);

    if (foundRoom !== null) {
      setIsDragging(true);
    }
  };

  // Handle mouse move for dragging rooms
  const handleMouseMove = (e) => {
    if (!isDragging || selectedRoom === null || currentFloor === "3d") return;

    const canvas = canvasRef.current;
    if (!canvas || !canvas.transformData) return;

    const rect = canvas.getBoundingClientRect();
    const { scale, offsetX, offsetY } = canvas.transformData;

    // Convert screen coordinates to world coordinates
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;

    // Convert to world coordinates
    const worldX = (screenX - offsetX) / scale;
    const worldY = (screenY - offsetY) / scale;

    // Calculate new position
    const roomWidth = layout.rooms[selectedRoom].dimensions[0];
    const roomLength = layout.rooms[selectedRoom].dimensions[1];

    const newX = Math.max(
      0,
      Math.min(buildingWidth - roomWidth, worldX - dragOffset.x)
    );
    const newY = Math.max(
      0,
      Math.min(buildingLength - roomLength, worldY - dragOffset.y)
    );

    // Optionally snap to grid
    const snapToGrid = true; // Set to false to disable
    let finalX = newX;
    let finalY = newY;

    if (snapToGrid && gridSize) {
      finalX = Math.round(newX / gridSize) * gridSize;
      finalY = Math.round(newY / gridSize) * gridSize;
    }

    // Update room position
    const updatedLayout = {
      ...layout,
      rooms: {
        ...layout.rooms,
        [selectedRoom]: {
          ...layout.rooms[selectedRoom],
          position: [finalX, finalY, layout.rooms[selectedRoom].position[2]],
        },
      },
    };

    setLayout(updatedLayout);
  };

  // Handle mouse up to end dragging
  const handleMouseUp = () => {
    if (isDragging) {
      setIsDragging(false);
      // Notify parent of layout change
      onLayoutChange(layout);
    }
  };

  // Submit layout for RL training
  const handleTrainRL = () => {
    onTrainRL({
      layout,
      userRating: userFeedback,
    });
  };

  // Get selected room data for display
  const getSelectedRoomData = () => {
    if (selectedRoom === null || !layout.rooms[selectedRoom]) return null;

    return {
      id: selectedRoom,
      ...layout.rooms[selectedRoom],
    };
  };

  // Get 3D view image URL
  const get3DImageUrl = () => {
    if (!layoutId) return null;
    return `${apiBaseUrl}/layouts/${layoutId}/hotel_layout_3d.png`;
  };

  return (
    <div className="layout-editor" ref={containerRef}>
      <h3>Interactive Layout Editor</h3>
      <p>
        {currentFloor === "3d"
          ? "3D visualization of the layout. Switch to floor view to modify rooms."
          : "Drag rooms to modify the layout. When satisfied, rate the layout and submit for RL training."}
      </p>
      <div className="grid-info">
        <p>
          Building: {buildingWidth}m × {buildingLength}m
        </p>
      </div>

      <FloorSelector
        buildingConfig={buildingConfig}
        currentFloor={currentFloor}
        onFloorChange={handleFloorChange}
      />

      {showFloorPlan && (
        <div className="canvas-container">
          <canvas
            ref={canvasRef}
            width={600}
            height={800}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          />
        </div>
      )}

      {show3DView && (
        <div className="view-3d-container">
          {layoutId ? (
            <img
              src={get3DImageUrl()}
              alt="3D View of Layout"
              className="layout-3d-image"
              onError={(e) => {
                e.target.onerror = null;
                e.target.src =
                  "https://via.placeholder.com/800x600?text=No+3D+View+Available";
              }}
            />
          ) : (
            <div className="placeholder-image">
              3D view not available. Save layout first.
            </div>
          )}
        </div>
      )}

      <div className="room-info">
        <RoomInfo room={getSelectedRoomData()} />
      </div>

      <div className="feedback-section">
        <h4>Provide Feedback for RL Training</h4>
        <label>
          Rate this layout (0-10):
          <input
            type="range"
            min="0"
            max="10"
            step="1"
            value={userFeedback}
            onChange={(e) => setUserFeedback(parseInt(e.target.value))}
          />
          <span>{userFeedback}</span>
        </label>

        <button
          className="btn-submit"
          onClick={handleTrainRL}
          disabled={!layout.rooms || Object.keys(layout.rooms).length === 0}
        >
          Submit for RL Training
        </button>
      </div>
    </div>
  );
};

export default LayoutEditor;
