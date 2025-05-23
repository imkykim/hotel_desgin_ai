// src/services/api.js
const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// Constants for local storage keys
const LOCAL_STORAGE_CONFIGS_KEY = "hotel_design_ai_configs";
const LOCAL_STORAGE_LAYOUTS_KEY = "hotel_design_ai_layouts";

// Function to handle API errors
const handleApiError = (error) => {
  console.error("API Error:", error);

  // If the error is an Axios response error
  if (error.response) {
    // Check if the detail is an object or a string
    const detail = error.response.data.detail;
    const errorMessage =
      typeof detail === "object"
        ? JSON.stringify(detail)
        : detail || "API error occurred";

    return {
      success: false,
      error: errorMessage,
    };
  } else if (error.request) {
    // The request was made but no response was received
    return { success: false, error: "No response from server" };
  } else {
    // Something happened in setting up the request that triggered an Error
    return { success: false, error: error.message || "Unknown error occurred" };
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
// Add these functions to your existing api.js file

// Initialize a chat2plan session
export const startChat2PlanSession = async (context) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat2plan/start`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(context),
    });

    if (!response.ok) {
      const errorData = await response
        .json()
        .catch(() => ({ detail: `HTTP error ${response.status}` }));
      console.error("Start session error:", errorData);
      return {
        success: false,
        error: errorData.detail || "Failed to start chat session",
      };
    }

    return await response.json();
  } catch (error) {
    console.error("Start session exception:", error);
    return handleApiError(error);
  }
};

// Send a message to the chat2plan system
export const sendChat2PlanMessage = async (sessionId, message) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat2plan/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ session_id: sessionId, message }),
    });

    if (!response.ok) {
      const errorData = await response
        .json()
        .catch(() => ({ detail: `HTTP error ${response.status}` }));
      console.error("Send message error:", errorData);
      return {
        success: false,
        error: errorData.detail || "Failed to process chat message",
      };
    }

    return await response.json();
  } catch (error) {
    console.error("Send message exception:", error);
    return handleApiError(error);
  }
};

// Get the current state of the chat2plan system
export const getChat2PlanState = async (sessionId) => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/chat2plan/state?session_id=${sessionId}`
    );

    if (!response.ok) {
      const errorData = await response
        .json()
        .catch(() => ({ detail: `HTTP error ${response.status}` }));
      console.error("Get state error:", errorData);
      return {
        success: false,
        error: errorData.detail || "Failed to get chat state",
      };
    }

    return await response.json();
  } catch (error) {
    console.error("Get state exception:", error);
    return handleApiError(error);
  }
};

export const skipStage = async (sessionId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat2plan/skip_stage`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ session_id: sessionId }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      return {
        success: false,
        error: errorData.detail || "Failed to skip stage",
      };
    }

    return await response.json();
  } catch (error) {
    console.error("Error skipping stage:", error);
    return handleApiError(error);
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

export const exportRequirements = async (sessionId) => {
  try {
    // Make sure we have a valid session ID
    if (!sessionId) {
      return { success: false, error: "Invalid session ID" };
    }

    // The backend will check for hotel_requirements_{session_id}.json
    const response = await fetch(
      `${API_BASE_URL}/api/chat2plan/export_requirements?session_id=${sessionId}`
    );

    if (!response.ok) {
      const errorData = await response.json();
      console.error("Export requirements error:", errorData);
      return {
        success: false,
        error: errorData.detail || "Failed to export requirements",
      };
    }

    const result = await response.json();
    console.log("Successfully exported requirements:", result);

    // Return the successful result including program_id
    return result;
  } catch (error) {
    console.error("Error exporting requirements:", error);
    return handleApiError(error);
  }
};

// Fetch backend logs for a session
export const getChat2PlanLogs = async (sessionId, since = 0) => {
  try {
    if (!sessionId) return { logs: [], total: 0 };
    const response = await fetch(
      `${API_BASE_URL}/api/chat2plan/logs?session_id=${sessionId}&since=${since}`
    );
    if (!response.ok) {
      return { logs: [], total: since };
    }
    return await response.json();
  } catch (error) {
    return { logs: [], total: since };
  }
};

export const updateBuildingConfig = async (buildingId, configData) => {
  try {
    console.log("Updating building config:", buildingId);
    console.log("Config data:", configData);

    // Make sure we don't send undefined values
    if (!buildingId) {
      console.error("Building ID is undefined");
      return {
        success: false,
        error: "Building ID is undefined",
      };
    }

    // Ensure we have a clean object to send
    const cleanedConfigData = JSON.parse(JSON.stringify(configData));

    const response = await fetch(`${API_BASE_URL}/update-building-config`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        building_id: buildingId,
        building_config: cleanedConfigData,
      }),
    });

    // Get response text first to debug any parsing issues
    const responseText = await response.text();
    console.log("Raw API response:", responseText);

    // Try to parse as JSON
    let responseData;
    try {
      responseData = JSON.parse(responseText);
    } catch (parseError) {
      console.error("Failed to parse API response as JSON:", parseError);
      return {
        success: false,
        error: `Failed to parse API response: ${responseText.substring(
          0,
          100
        )}...`,
      };
    }

    if (!response.ok) {
      console.error("Error updating building config:", responseData);
      return {
        success: false,
        error:
          responseData?.detail ||
          `Failed to update building configuration (${response.status})`,
      };
    }

    console.log("Building config update successful:", responseData);

    return {
      success: true,
      message:
        responseData?.message || "Building configuration updated successfully",
      filepath: responseData?.filepath,
    };
  } catch (error) {
    console.error("Exception updating building config:", error);
    return {
      success: false,
      error:
        error?.message ||
        "Unknown error occurred while updating building configuration",
    };
  }
};

// Generate improved layout with RL (POST, JSON body, snake_case keys)
export const generateImprovedLayout = async (
  buildingId,
  programId,
  referenceLayoutId,
  fixedRoomsFile = null // Add this parameter
) => {
  try {
    const response = await fetch(`${API_BASE_URL}/engine/generate-improved`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        building_id: buildingId,
        program_id: programId,
        reference_layout_id: referenceLayoutId || null,
        fixed_rooms_file: fixedRoomsFile || null, // Add this to the request
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({
        detail: `HTTP error ${response.status}: ${response.statusText}`,
      }));

      return {
        success: false,
        error:
          errorData.detail ||
          `Failed to generate improved layout (${response.status})`,
      };
    }

    return await response.json();
  } catch (error) {
    return handleApiError(error);
  }
};

// Generate layout with reference (fallback: also use /engine/generate-improved)
export const generateLayoutWithReference = async (
  buildingId,
  programId,
  referenceLayoutId,
  fixedRoomsFile = null // Add this parameter
) => {
  try {
    const response = await fetch(`${API_BASE_URL}/engine/generate-improved`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        building_id: buildingId,
        program_id: programId,
        reference_layout_id: referenceLayoutId || null,
        fixed_rooms_file: fixedRoomsFile || null, // Add this to the request
      }),
    });

    if (!response.ok) {
      const errorData = await response
        .json()
        .catch(() => ({ detail: `HTTP error ${response.status}` }));

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

// Optionally, if you want to support generateWithZones (for standard floor zones)
export const generateLayoutWithZones = async (
  buildingId,
  programConfig,
  fixedRoomsFile
) => {
  try {
    const response = await fetch(`${API_BASE_URL}/engine/generate-with-zones`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        building_id: buildingId,
        program_id: programConfig,
        fixed_rooms_file: fixedRoomsFile,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({
        detail: `HTTP error ${response.status}`,
      }));
      return {
        success: false,
        error: errorData.detail || "Failed to generate layout with zones",
      };
    }

    return await response.json();
  } catch (error) {
    return handleApiError(error);
  }
};

// Export layout to Rhino script
export const exportLayoutToRhinoScript = async (layoutId) => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/visualize-layout/export-rhino-script/${layoutId}`
    );

    if (!response.ok) {
      // Handle error response
      const errorData = await response.json().catch(() => ({}));
      return {
        success: false,
        error:
          errorData.detail ||
          `Failed to export Rhino script (${response.status})`,
      };
    }

    // Get the file blob
    const blob = await response.blob();

    // Create a download link
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.style.display = "none";
    a.href = url;
    a.download = `hotel_layout_${layoutId}_rhino.py`;

    // Append to document and trigger download
    document.body.appendChild(a);
    a.click();

    // Clean up
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);

    return { success: true };
  } catch (error) {
    console.error("Error exporting to Rhino:", error);
    return { success: false, error: error.message || "Unknown error occurred" };
  }
};
