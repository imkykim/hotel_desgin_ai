import React, { useState, useEffect, useRef } from "react";
import "../styles/GridSelector.css";

const GridSelector = ({
  buildingWidth = 80,
  buildingLength = 120,
  gridSize = 8,
  onSelectionChange = () => {},
  onFixedElementSelect = () => {},
  selectionMode = "standard", // 'standard', 'entrance', 'core'
}) => {
  // Calculate grid dimensions
  const gridCellsX = Math.floor(buildingWidth / gridSize);
  const gridCellsY = Math.floor(buildingLength / gridSize);

  // State for selected cells
  const [selectedCells, setSelectedCells] = useState(new Set());
  const [entranceCells, setEntranceCells] = useState(new Set());
  const [coreCells, setCoreCells] = useState(new Set());
  const [isSelecting, setIsSelecting] = useState(false);
  const [selectionStart, setSelectionStart] = useState(null);
  const [selectionModeInternal, setSelectionModeInternal] = useState("add"); // 'add' or 'remove'

  const canvasRef = useRef(null);

  // Draw the grid
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    const aspectRatio = buildingWidth / buildingLength;

    // Set a reasonable canvas size while maintaining proportions
    const CELL_SIZE = 50; // Increase this value for bigger grid cells
    if (aspectRatio > 1) {
      canvas.width = Math.min(1200, gridCellsX * CELL_SIZE);
      canvas.height = canvas.width / aspectRatio;
    } else {
      canvas.height = Math.min(900, gridCellsY * CELL_SIZE);
      canvas.width = canvas.height * aspectRatio;
    }

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
        const isEntrance = entranceCells.has(cellKey);
        const isCore = coreCells.has(cellKey);

        // Determine cell color based on what it represents
        if (isEntrance) {
          ctx.fillStyle = "#FF9800"; // Orange for entrance
        } else if (isCore) {
          ctx.fillStyle = "#F44336"; // Red for core
        } else if (isSelected) {
          ctx.fillStyle = "#3B71CA"; // Blue for standard floors
        } else {
          ctx.fillStyle = "#f8f9fa"; // Default background
        }

        // Highlight active selection mode
        if (selectionMode === "entrance" && isEntrance) {
          ctx.lineWidth = 2;
          ctx.strokeStyle = "#FF9800";
        } else if (selectionMode === "core" && isCore) {
          ctx.lineWidth = 2;
          ctx.strokeStyle = "#F44336";
        } else {
          ctx.lineWidth = 0.5;
          ctx.strokeStyle = "#000";
        }

        ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
        ctx.strokeRect(x * cellSize, y * cellSize, cellSize, cellSize);
      }
    }

    // Draw building outline
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, gridCellsX * cellSize, gridCellsY * cellSize);
  }, [
    selectedCells,
    entranceCells,
    coreCells,
    gridCellsX,
    gridCellsY,
    buildingWidth,
    buildingLength,
    selectionMode,
  ]);

  // Handle mouse down for room selection and dragging
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

    const cellKey = `${x},${y}`;

    // Determine which selection we're working with based on mode
    if (selectionMode === "entrance") {
      // For entrance and core, we only want single cells (not dragging)
      if (entranceCells.has(cellKey)) {
        setSelectionModeInternal("remove");
        const newEntranceCells = new Set(entranceCells);
        newEntranceCells.delete(cellKey);
        setEntranceCells(newEntranceCells);
      } else {
        setSelectionModeInternal("add");
        // Clear previous entrance cells (we only want one)
        setEntranceCells(new Set([cellKey]));
      }
      // Update parent component
      const gridCoords = { x: x * gridSize, y: y * gridSize };
      onFixedElementSelect("entrance", gridCoords);

      // No dragging needed for fixed elements
      setIsSelecting(false);
    } else if (selectionMode === "core") {
      if (coreCells.has(cellKey)) {
        setSelectionModeInternal("remove");
        const newCoreCells = new Set(coreCells);
        newCoreCells.delete(cellKey);
        setCoreCells(newCoreCells);
      } else {
        setSelectionModeInternal("add");
        // For core, we might want to allow multiple cores
        const newCoreCells = new Set(coreCells);
        newCoreCells.add(cellKey);
        setCoreCells(newCoreCells);
      }
      // Update parent component
      const gridCoords = { x: x * gridSize, y: y * gridSize };
      onFixedElementSelect("core", gridCoords);

      // No dragging needed for fixed elements
      setIsSelecting(false);
    } else {
      // Standard floor zone selection - check if cell is already selected
      if (selectedCells.has(cellKey)) {
        setSelectionModeInternal("remove");
      } else {
        setSelectionModeInternal("add");
      }
    }
  };

  const handleMouseMove = (e) => {
    if (!isSelecting || !selectionStart || selectionMode !== "standard") return;

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
        if (selectionModeInternal === "add") {
          newSelection.add(cellKey);
        } else {
          newSelection.delete(cellKey);
        }
      }
    }

    setSelectedCells(newSelection);
  };

  const handleMouseUp = () => {
    if (!isSelecting) return;

    setIsSelecting(false);

    // Only update parent for standard floor zones
    if (selectionMode === "standard") {
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
    }
  };

  const clearSelection = () => {
    if (selectionMode === "entrance") {
      setEntranceCells(new Set());
      onFixedElementSelect("entrance", null);
    } else if (selectionMode === "core") {
      setCoreCells(new Set());
      onFixedElementSelect("core", null);
    } else {
      setSelectedCells(new Set());
      onSelectionChange([]);
    }
  };

  // Fill entire area (only for standard floor zones)
  const selectAll = () => {
    if (selectionMode !== "standard") return;

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
      <h3>
        {selectionMode === "entrance"
          ? "Select Entrance/Lobby Position"
          : selectionMode === "core"
          ? "Select Core Circulation Position"
          : "Standard Floor Zone Selection"}
      </h3>
      <p>
        {selectionMode === "entrance"
          ? "Click on the grid to place the main entrance/lobby (orange)"
          : selectionMode === "core"
          ? "Click on the grid to place vertical circulation cores (red)"
          : "Select the areas that will contain standard floors (guest rooms and core)"}
      </p>

      <div className="canvas-container">
        <canvas
          ref={canvasRef}
          width={gridCellsX * 100} // Scale up for better visibility
          height={gridCellsY * 100}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        />
      </div>

      <div className="controls">
        <button onClick={clearSelection} className="btn btn-secondary">
          Clear{" "}
          {selectionMode === "standard"
            ? "Selection"
            : selectionMode === "entrance"
            ? "Entrance"
            : "Cores"}
        </button>
        {selectionMode === "standard" && (
          <button onClick={selectAll} className="btn btn-primary">
            Select All
          </button>
        )}
      </div>

      <div className="statistics">
        <p>
          Building dimensions: {buildingWidth}m × {buildingLength}m
        </p>
        <p>Grid size: {gridSize}m</p>
        {selectionMode === "standard" && (
          <p>Selected area: {selectedCells.size * gridSize * gridSize}m²</p>
        )}
        {selectionMode === "entrance" && (
          <p>Entrance positions: {entranceCells.size}</p>
        )}
        {selectionMode === "core" && <p>Core positions: {coreCells.size}</p>}
        {selectionMode !== "standard" && (
          <p>
            <em>Fixed elements will be saved as exact positions, not zones.</em>
          </p>
        )}
      </div>
    </div>
  );
};

export default GridSelector;
