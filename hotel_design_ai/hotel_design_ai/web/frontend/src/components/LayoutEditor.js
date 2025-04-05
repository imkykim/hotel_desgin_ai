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
    </div>
  );
};

// Main layout editor component
const LayoutEditor = ({
  initialLayout,
  buildingConfig,
  onLayoutChange = () => {},
  onTrainRL = () => {},
}) => {
  const [layout, setLayout] = useState(initialLayout || { rooms: {} });
  const [selectedRoom, setSelectedRoom] = useState(null);
  const [currentFloor, setCurrentFloor] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [userFeedback, setUserFeedback] = useState(5); // 0-10 rating

  const canvasRef = useRef(null);

  // Extract building dimensions from config
  const { width, length, floor_height } = buildingConfig || {
    width: 80,
    length: 120,
    floor_height: 4.0,
  };

  // Function to draw the layout on the canvas
  const drawLayout = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const scale = Math.min(canvas.width / width, canvas.height / length);

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw building outline
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, width * scale, length * scale);

    // Draw rooms for the current floor
    Object.entries(layout.rooms || {}).forEach(([roomId, roomData]) => {
      const roomFloor = Math.floor(roomData.position[2] / floor_height);

      // Skip rooms not on the current floor
      if (roomFloor !== currentFloor) return;

      drawRoom(ctx, parseInt(roomId), roomData, scale);
    });
  };

  // Draw an individual room
  const drawRoom = (ctx, roomId, roomData, scale) => {
    const [x, y] = roomData.position;
    const [w, l] = roomData.dimensions;
    const roomType = roomData.type;

    // Get room styling
    ctx.fillStyle = ROOM_COLORS[roomType] || ROOM_COLORS.default;
    ctx.strokeStyle = selectedRoom === roomId ? "#ff0000" : "#000";
    ctx.lineWidth = selectedRoom === roomId ? 2 : 1;

    // Draw room rectangle
    ctx.fillRect(x * scale, y * scale, w * scale, l * scale);
    ctx.strokeRect(x * scale, y * scale, w * scale, l * scale);

    // Draw room label
    ctx.fillStyle = "#000";
    ctx.font = "12px Arial";
    ctx.textAlign = "center";

    const name = roomData.metadata?.name || roomType;
    ctx.fillText(
      name.length > 15 ? name.substring(0, 12) + "..." : name,
      (x + w / 2) * scale,
      (y + l / 2) * scale
    );
  };

  // Draw the layout when component updates
  useEffect(() => {
    drawLayout();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout, selectedRoom, currentFloor, width, length, floor_height]);

  // Handle mouse down for room selection and dragging
  const handleMouseDown = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scale = Math.min(canvas.width / width, canvas.height / length);

    const mouseX = (e.clientX - rect.left) / scale;
    const mouseY = (e.clientY - rect.top) / scale;

    // Find room under mouse
    let foundRoom = null;

    Object.entries(layout.rooms || {}).forEach(([roomId, data]) => {
      const roomFloor = Math.floor(data.position[2] / floor_height);
      if (roomFloor !== currentFloor) return;

      const [x, y] = data.position;
      const [w, l] = data.dimensions;

      if (mouseX >= x && mouseX <= x + w && mouseY >= y && mouseY <= y + l) {
        foundRoom = parseInt(roomId);
        setDragOffset({
          x: mouseX - x,
          y: mouseY - y,
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
    if (!isDragging || selectedRoom === null) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scale = Math.min(canvas.width / width, canvas.height / length);

    const mouseX = (e.clientX - rect.left) / scale;
    const mouseY = (e.clientY - rect.top) / scale;

    // Calculate new position
    const roomWidth = layout.rooms[selectedRoom].dimensions[0];
    const roomLength = layout.rooms[selectedRoom].dimensions[1];

    const newX = Math.max(
      0,
      Math.min(width - roomWidth, mouseX - dragOffset.x)
    );
    const newY = Math.max(
      0,
      Math.min(length - roomLength, mouseY - dragOffset.y)
    );

    // Update room position
    const updatedLayout = {
      ...layout,
      rooms: {
        ...layout.rooms,
        [selectedRoom]: {
          ...layout.rooms[selectedRoom],
          position: [newX, newY, layout.rooms[selectedRoom].position[2]],
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

  // Handle floor change
  const handleFloorChange = (floor) => {
    setCurrentFloor(floor);
    setSelectedRoom(null);
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

  return (
    <div className="layout-editor">
      <h3>Interactive Layout Editor</h3>
      <p>
        Drag rooms to modify the layout. When satisfied, rate the layout and
        submit for RL training.
      </p>

      <FloorSelector
        buildingConfig={buildingConfig}
        currentFloor={currentFloor}
        onFloorChange={handleFloorChange}
      />

      <div className="canvas-container">
        <canvas
          ref={canvasRef}
          width={800}
          height={600}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        />
      </div>

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
