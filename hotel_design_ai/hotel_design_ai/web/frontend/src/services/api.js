// src/services/api.js
const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// Constants for local storage keys
const LOCAL_STORAGE_CONFIGS_KEY = "hotel_design_ai_configs";
const LOCAL_STORAGE_LAYOUTS_KEY = "hotel_design_ai_layouts";

// Function to handle API errors
const handleApiError = (error) => {
  console.error("API Error:", error);
  if (error.response) {
    // The request was made and the server responded with a status code
    // that falls out of the range of 2xx
    return {
      success: false,
      error: error.response.data.detail || "API error occurred",
    };
  } else if (error.request) {
    // The request was made but no response was received
    return { success: false, error: "No response from server" };
  } else {
    // Something happened in setting up the request that triggered an Error
    return { success: false, error: error.message };
  }
};

// Helper function to initialize storage
const initializeStorage = (key, initialValue) => {
  // Check if we already have configurations stored
  const storedData = localStorage.getItem(key);
  if (!storedData) {
    // If not, initialize with empty array or provided initial value
    localStorage.setItem(key, JSON.stringify(initialValue || []));
    return initialValue || [];
  }

  try {
    // Parse existing data
    return JSON.parse(storedData);
  } catch (error) {
    console.error(`Error parsing ${key} from localStorage:`, error);
    localStorage.setItem(key, JSON.stringify(initialValue || []));
    return initialValue || [];
  }
};
