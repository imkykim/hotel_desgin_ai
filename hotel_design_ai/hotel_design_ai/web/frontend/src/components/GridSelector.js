import React, { useState, useEffect, useRef } from "react";
import "../styles/GridSelector.css";

const GridSelector = ({
  buildingWidth = 80,
  buildingLength = 120,
  gridSize = 8,
  onSelectionChange = () => {},
}) => {
  // Calculate grid dimensions
  const gridCellsX = Math.floor(buildingWidth / gridSize);
  const gridCellsY = Math.floor(buildingLength / gridSize);

  // State for selected cells
  const [selectedCells, setSelectedCells] = useState(new Set());
  const [isSelecting, setIsSelecting] = useState(false);
  const [selectionStart, setSelectionStart] = useState(null);
  const [selectionMode, setSelectionMode] = useState("add"); // 'add' or 'remove'

  const canvasRef = useRef(null);

  // Draw the grid
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const cellSize = Math.min(
      Math.floor(canvas.width / gridCellsX),
      Math.floor(canvas.height / gridCellsY)
    );

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = "#ccc";
    ctx.lineWidth = 0.5;

    // Draw cells
    for (let x = 0; x < gridCellsX; x++) {
      for (let y = 0; y < gridCellsY; y++) {
        const cellKey = `${x},${y}`;
        const isSelected = selectedCells.has(cellKey);

        ctx.fillStyle = isSelected ? "#3B71CA" : "#f8f9fa";
        ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
        ctx.strokeRect(x * cellSize, y * cellSize, cellSize, cellSize);
      }
    }

    // Draw building outline
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, gridCellsX * cellSize, gridCellsY * cellSize);
  }, [selectedCells, gridCellsX, gridCellsY]);

  // Handle mouse interactions for selection
  const handleMouseDown = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const cellSize = Math.min(
      Math.floor(canvas.width / gridCellsX),
      Math.floor(canvas.height / gridCellsY)
    );

    const x = Math.floor((e.clientX - rect.left) / cellSize);
    const y = Math.floor((e.clientY - rect.top) / cellSize);

    setIsSelecting(true);
    setSelectionStart({ x, y });

    // Determine selection mode based on whether cell is already selected
    const cellKey = `${x},${y}`;
    if (selectedCells.has(cellKey)) {
      setSelectionMode("remove");
    } else {
      setSelectionMode("add");
    }
  };

  const handleMouseMove = (e) => {
    if (!isSelecting || !selectionStart) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const cellSize = Math.min(
      Math.floor(canvas.width / gridCellsX),
      Math.floor(canvas.height / gridCellsY)
    );

    const currentX = Math.floor((e.clientX - rect.left) / cellSize);
    const currentY = Math.floor((e.clientY - rect.top) / cellSize);

    // Calculate selection rectangle
    const startX = Math.min(selectionStart.x, currentX);
    const startY = Math.min(selectionStart.y, currentY);
    const endX = Math.max(selectionStart.x, currentX);
    const endY = Math.max(selectionStart.y, currentY);

    // Create new selection set
    const newSelection = new Set(selectedCells);

    // Add or remove cells in the selection rectangle
    for (let x = startX; x <= endX; x++) {
      for (let y = startY; y <= endY; y++) {
        const cellKey = `${x},${y}`;
        if (selectionMode === "add") {
          newSelection.add(cellKey);
        } else {
          newSelection.delete(cellKey);
        }
      }
    }

    setSelectedCells(newSelection);
  };

  const handleMouseUp = () => {
    setIsSelecting(false);

    // Convert selected cells to grid coordinates and notify parent
    const selectedGridAreas = Array.from(selectedCells).map((key) => {
      const [x, y] = key.split(",").map(Number);
      return {
        x: x * gridSize,
        y: y * gridSize,
        width: gridSize,
        height: gridSize,
      };
    });

    onSelectionChange(selectedGridAreas);
  };

  const clearSelection = () => {
    setSelectedCells(new Set());
    onSelectionChange([]);
  };

  // Fill entire area
  const selectAll = () => {
    const allCells = new Set();
    for (let x = 0; x < gridCellsX; x++) {
      for (let y = 0; y < gridCellsY; y++) {
        allCells.add(`${x},${y}`);
      }
    }
    setSelectedCells(allCells);

    const allAreas = Array.from(allCells).map((key) => {
      const [x, y] = key.split(",").map(Number);
      return {
        x: x * gridSize,
        y: y * gridSize,
        width: gridSize,
        height: gridSize,
      };
    });

    onSelectionChange(allAreas);
  };

  return (
    <div className="grid-selector">
      <h3>Standard Floor Zone Selection</h3>
      <p>
        Select the areas that will contain standard floors (guest rooms and
        core)
      </p>

      <div className="canvas-container">
        <canvas
          ref={canvasRef}
          width={gridCellsX * 50} // Scale up for better visibility
          height={gridCellsY * 50}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        />
      </div>

      <div className="controls">
        <button onClick={clearSelection} className="btn btn-secondary">
          Clear Selection
        </button>
        <button onClick={selectAll} className="btn btn-primary">
          Select All
        </button>
      </div>

      <div className="statistics">
        <p>
          Building dimensions: {buildingWidth}m × {buildingLength}m
        </p>
        <p>Grid size: {gridSize}m</p>
        <p>Selected area: {selectedCells.size * gridSize * gridSize}m²</p>
      </div>
    </div>
  );
};

export default GridSelector;
