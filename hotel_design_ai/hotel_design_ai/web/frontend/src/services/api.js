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

// Generate configurations
export const generateConfigs = async (userData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/generate-configs`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(userData),
    });

    if (!response.ok) {
      const errorData = await response.json();
      return {
        success: false,
        error: errorData.detail || "Failed to generate configurations",
      };
    }

    return await response.json();
  } catch (error) {
    return handleApiError(error);
  }
};

const apiBaseUrl = process.env.REACT_APP_API_URL || "http://localhost:8000";

export async function listLayouts() {
  const response = await fetch(`${apiBaseUrl}/files/layouts`);
  return await response.json();
}

// Update getLayout function in api.js to use /files/layouts path
export const getLayout = async (layoutId) => {
  try {
    // Use the files router path - this matches the backend router structure
    const response = await fetch(`${API_BASE_URL}/files/layouts/${layoutId}`);
    console.log(
      `Fetching layout from: ${API_BASE_URL}/files/layouts/${layoutId}`
    );

    if (!response.ok) {
      const errorData = await response
        .json()
        .catch(() => ({ detail: "Error parsing response" }));
      return {
        success: false,
        error:
          errorData.detail || `Failed to fetch layout (${response.status})`,
      };
    }

    return await response.json();
  } catch (error) {
    console.error("Error in getLayout:", error);
    return handleApiError(error);
  }
};

// List all configurations
export const listConfigurations = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/files/configurations`);

    if (!response.ok) {
      const errorData = await response.json();
      return {
        success: false,
        error: errorData.detail || "Failed to fetch configurations",
      };
    }

    return await response.json();
  } catch (error) {
    return handleApiError(error);
  }
};

// Get specific configuration
export const getConfiguration = async (configType, configId) => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/configuration/${configType}/${configId}`
    );

    if (!response.ok) {
      const errorData = await response.json();
      return {
        success: false,
        error: errorData.detail || "Failed to fetch configuration",
      };
    }

    return await response.json();
  } catch (error) {
    return handleApiError(error);
  }
};
// Generate visualizations for a layout
export const generateLayoutVisualizations = async (layoutId) => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/visualize-layout/${layoutId}`,
      {
        method: "POST",
      }
    );

    if (!response.ok) {
      const errorData = await response.json();
      return {
        success: false,
        error: errorData.detail || "Failed to generate layout visualizations",
      };
    }

    return await response.json();
  } catch (error) {
    return handleApiError(error);
  }
};

// Generate visualizations for a configuration
export const generateVisualizations = async (configType, configId) => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/visualize/${configType}/${configId}`,
      {
        method: "POST",
      }
    );

    if (!response.ok) {
      const errorData = await response.json();
      return {
        success: false,
        error: errorData.detail || "Failed to generate visualizations",
      };
    }

    return await response.json();
  } catch (error) {
    return handleApiError(error);
  }
};

// Generate layout using configurations
export const generateLayout = async (data) => {
  try {
    const response = await fetch(`${API_BASE_URL}/generate-layout`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const errorData = await response.json();
      return {
        success: false,
        error: errorData.detail || "Failed to generate layout",
      };
    }

    return await response.json();
  } catch (error) {
    return handleApiError(error);
  }
};

// Modify layout
export const modifyLayout = async (data) => {
  try {
    const response = await fetch(`${API_BASE_URL}/modify-layout`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const errorData = await response.json();
      return {
        success: false,
        error: errorData.detail || "Failed to modify layout",
      };
    }

    return await response.json();
  } catch (error) {
    return handleApiError(error);
  }
};
