import React, { useState, useEffect, useRef } from 'react';
import '../styles/LayoutEditor.css';

const LayoutEditor = ({ 
  initialLayout, 
  buildingConfig,
  onLayoutChange = () => {},
  onTrainRL = () => {}
}) => {
  const [layout, setLayout] = useState(initialLayout || { rooms: {} });
  const [selectedRoom, setSelectedRoom] = useState(null);
  const [currentFloor, setCurrentFloor] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [userFeedback, setUserFeedback] = useState(5); // 0-10 rating
  
  const canvasRef = useRef(null);
  
  // Room colors by type
  const roomColors = {
    "entrance": "#2c7fb8",
    "lobby": "#7fcdbb",
    "vertical_circulation": "#FF0000",
    "restaurant": "#f0f9e8",
    "meeting_room": "#edf8b1",
    "guest_room": "#f7fcb9",
    "service_area": "#d9f0a3",
    "back_of_house": "#addd8e",
    "parking": "#D3D3D3",
    "mechanical": "#A0A0A0",
    "default": "#efefef",
  };
  
  // Extract building dimensions from config
  const { width, length, floor_height } = buildingConfig || { 
    width: 80, 
    length: 120, 
    floor_height: 4.0 
  };
  
  // Draw the layout
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const scale = Math.min(
      canvas.width / width,
      canvas.height / length
    );
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw building outline
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, width * scale, length * scale);
    
    // Draw rooms for the current floor
    Object.entries(layout.rooms || {}).forEach(([roomId, roomData]) => {
      const roomFloor = Math.floor(roomData.position[2] / floor_height);
      
      // Skip rooms not on the current floor
      if (roomFloor !== currentFloor) return;
      
      const [x, y, z] = roomData.position;
      const [w, l, h] = roomData.dimensions;
      const roomType = roomData.type;
      
      // Draw room rectangle
      ctx.fillStyle = roomColors[roomType] || roomColors.default;
      ctx.strokeStyle = selectedRoom === parseInt(roomId) ? '#ff0000' : '#000';
      ctx.lineWidth = selectedRoom === parseInt(roomId) ? 2 : 1;
      
      ctx.fillRect(x * scale, y * scale, w * scale, l * scale);
      ctx.strokeRect(x * scale, y * scale, w * scale, l * scale);
      
      // Draw room label
      ctx.fillStyle = '#000';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      
      const name = roomData.metadata?.name || roomType;
      ctx.fillText(
        name.length > 15 ? name.substring(0, 12) + '...' : name, 
        (x + w/2) * scale, 
        (y + l/2) * scale
      );
    });
    
  }, [layout, selectedRoom, currentFloor, width, length, floor_height]);
  
  // Handle mouse down for room selection and dragging
  const handleMouseDown = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scale = Math.min(
      canvas.width / width,
      canvas.height / length
    );
    
    const mouseX = (e.clientX - rect.left) / scale;
    const mouseY = (e.clientY - rect.top) / scale;
    
    // Find room under mouse
    let foundRoom = null;
    Object.entries(layout.rooms || {}).forEach(([roomId, roomData]) => {
      const roomFloor = Math.floor(roomData.position[2] / floor_height);
      if (roomFloor !== currentFloor) return;
      
      const [x, y, z] = roomData.position;
      const [w, l, h] = roomData.dimensions;
      
      if (mouseX >= x && mouseX <= x + w && mouseY >= y && mouseY <= y + l) {
        foundRoom = parseInt(roomId);
        setDragOffset({
          x: mouseX - x,
          y: mouseY - y
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
    const scale = Math.min(
      canvas.width / width,
      canvas.height / length
    );
    
    const mouseX = (e.clientX - rect.left) / scale;
    const mouseY = (e.clientY - rect.top) / scale;
    
    // Calculate new position
    const newX = Math.max(0, Math.min(width - layout.rooms[selectedRoom].dimensions[0], mouseX - dragOffset.x));
    const newY = Math.max(0, Math.min(length - layout.rooms[selectedRoom].dimensions[1], mouseY - dragOffset.y));
    
    // Update room position
    const updatedLayout = {
      ...layout,
      rooms: {
        ...layout.rooms,
        [selectedRoom]: {
          ...layout.rooms[selectedRoom],
          position: [
            newX,
            newY,
            layout.rooms[selectedRoom].position[2]
          ]
        }
      }
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
  
  // Generate floor buttons
  const generateFloorButtons = () => {
    const floors = [];
    if (buildingConfig) {
      for (let f = buildingConfig.min_floor || -1; f <= (buildingConfig.max_floor || 3); f++) {
        floors.push(f);
      }
    } else {
      // Default floors if no config
      floors.push(-1, 0, 1, 2, 3);
    }
    
    return floors.map(floor => (
      <button 
        key={`floor-${floor}`}
        className={`floor-button ${currentFloor === floor ? 'active' : ''}`}
        onClick={() => handleFloorChange(floor)}
      >
        {floor < 0 ? `B${Math.abs(floor)}` : floor}
      </button>
    ));
  };
  
  // Submit layout for RL training
  const handleTrainRL = () => {
    onTrainRL({
      layout,
      userRating: userFeedback
    });
  };
  
  return (
    <div className="layout-editor">
      <h3>Interactive Layout Editor</h3>
      <p>Drag rooms to modify the layout. When satisfied, rate the layout and submit for RL training.</p>
      
      <div className="floor-selector">
        <label>Floor: </label>
        {generateFloorButtons()}
      </div>
      
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
        {selectedRoom !== null && (
          <>
            <h4>Selected Room</h4>
            <p>ID: {selectedRoom}</p>
            <p>Type: {layout.rooms[selectedRoom].type}</p>
            <p>Name: {layout.rooms[selectedRoom].metadata?.name || 'Unnamed'}</p>
            <p>Position: ({layout.rooms[selectedRoom].position.map(v => v.toFixed(1)).join(', ')})</p>
            <p>Dimensions: {layout.rooms[selectedRoom].dimensions.map(v => v.toFixed(1)).join(' × ')}m</p>
          </>
        )}
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
