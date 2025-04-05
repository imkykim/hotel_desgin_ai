// API service for connecting to the backend
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// Helper function for handling errors
const handleResponse = async (response) => {
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "An error occurred");
  }
  return response.json();
};

// Generate building configuration and program requirements
export const generateConfigs = async (formData) => {
  const response = await fetch(`${API_URL}/generate-configs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(formData),
  });

  return handleResponse(response);
};

// Generate a hotel layout
export const generateLayout = async (params) => {
  const response = await fetch(`${API_URL}/generate-layout`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(params),
  });

  return handleResponse(response);
};

// Modify a room in a layout
export const modifyLayout = async (params) => {
  const response = await fetch(`${API_URL}/modify-layout`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(params),
  });

  return handleResponse(response);
};

// Get a specific layout
export const getLayout = async (layoutId) => {
  const response = await fetch(`${API_URL}/layouts/${layoutId}`);
  return handleResponse(response);
};

// List all layouts
export const listLayouts = async () => {
  const response = await fetch(`${API_URL}/layouts`);
  return handleResponse(response);
};

// Helper function to convert layout data from backend to frontend format
export const processLayoutData = (layoutData) => {
  const rooms = layoutData.rooms || {};

  // Convert to array for easier rendering
  const roomsArray = Object.entries(rooms).map(([id, room]) => ({
    id: parseInt(id),
    ...room,
  }));

  return {
    rooms: roomsArray,
    metrics: layoutData.metrics || {},
    buildingDimensions: layoutData.building_dimensions || {
      width: 60,
      length: 80,
      height: 30,
    },
  };
};
