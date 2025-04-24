import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { generateConfigs, exportRequirements } from "../services/api";
import Chat2PlanInterface from "../components/Chat2PlanInterface";
import "../styles/ConfigGenerator.css";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const ConfigGenerator = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    hotel_name: "",
    hotel_type: "luxury",
    num_rooms: 100,
    building_width: "",
    building_length: "",
    building_height: "",
    min_floor: "",
    max_floor: "",
    floor_height: 4.5,
    structural_grid_x: 8.0,
    structural_grid_y: 8.0,
    grid_size: 1.0,
    podium_min_floor: -2,
    podium_max_floor: 1,
    special_requirements: "",
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [generatedConfigs, setGeneratedConfigs] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [showConfirmation, setShowConfirmation] = useState(false);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === "checkbox" ? checked : value,
    });
  };

  const handleNumericChange = (e) => {
    const { name, value } = e.target;
    // Allow empty string or numeric values
    if (value === "" || !isNaN(value)) {
      setFormData({
        ...formData,
        [name]: value === "" ? "" : Number(value),
      });
    }
  };

  const handleSessionStart = (newSessionId) => {
    console.log("Chat2Plan session started:", newSessionId);
    setSessionId(newSessionId);
  };

  const handleRequirementsUpdate = (requirements) => {
    console.log("Requirements updated:", requirements);
    // Update form data with the latest requirements
    setFormData((prev) => ({
      ...prev,
      special_requirements: requirements,
    }));
  };

  const handleSubmitClick = async (e) => {
    e.preventDefault();

    // Check for requirements file existence via backend
    if (!sessionId) {
      setError("Session not initialized. Please refresh and try again.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await exportRequirements(sessionId);
      if (result && result.success) {
        // Requirements file exists, proceed
        generateConfiguration();
      } else {
        // Not found, show modal
        setShowConfirmation(true);
      }
    } catch (err) {
      setError("Error checking requirements file: " + (err?.message || err));
    } finally {
      setLoading(false);
    }
  };

  const generateConfiguration = async () => {
    // Clear previous confirmations or errors
    setShowConfirmation(false);
    setError(null);
    setLoading(true);

    try {
      // Generate the building envelope config
      const buildingEnvelope = {
        width: formData.building_width || 60.0,
        length: formData.building_length || 80.0,
        height: formData.building_height || 100.0,
        min_floor: formData.min_floor !== "" ? formData.min_floor : -2,
        max_floor: formData.max_floor !== "" ? formData.max_floor : 20,
        floor_height: formData.floor_height,
        structural_grid_x: formData.structural_grid_x,
        structural_grid_y: formData.structural_grid_y,
        grid_size: formData.grid_size,
        podium: {
          min_floor:
            formData.podium_min_floor !== "" ? formData.podium_min_floor : -2,
          max_floor:
            formData.podium_max_floor !== "" ? formData.podium_max_floor : 1,
          description: "Podium section (裙房) of the building",
        },
        standard_floor: {
          start_floor: 2,
          end_floor: 20,
          width: 56.0,
          length: 20.0,
          position_x: 0.0,
          position_y: 32.0,
          corridor_width: 4.0,
          room_depth: 8.0,
        },
      };

      // Generate a unique name for the building config
      const safe_name =
        formData.hotel_name.toLowerCase().replace(/\s+/g, "_") || "hotel";
      const timestamp = new Date()
        .toISOString()
        .replace(/[-:.TZ]/g, "")
        .substring(0, 14);
      const building_filename = `${safe_name}_${timestamp}_building.json`;

      // Save the building config
      const response = await fetch(`${API_BASE_URL}/generate-building-config`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          building_envelope: buildingEnvelope,
          filename: building_filename,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || "Failed to generate building configuration"
        );
      }

      const result = await response.json();

      // Set generated configs
      setGeneratedConfigs({
        building_envelope: {
          filename: building_filename,
          data: buildingEnvelope,
        },
        hotel_requirements: {
          filename: `hotel_requirements_${sessionId}.json`,
          data: {
            note: "Generated by Chat2Plan",
          },
        },
      });
    } catch (err) {
      const errorMessage = err?.message || "Unknown error occurred";
      setError("Error generating configurations: " + errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const proceedToInteractive = () => {
    // Use the building ID from the generated config and program ID from Chat2Plan
    const buildingId =
      generatedConfigs?.building_envelope?.filename.replace(".json", "") ||
      "default";

    // If no program ID was generated, use default
    const finalProgramId = sessionId || "default";

    navigate(`/interactive/${buildingId}/${finalProgramId}`);
  };

  // Safely convert error to string for display
  const getErrorMessage = () => {
    if (!error) return null;

    if (typeof error === "string") {
      return error;
    } else if (error && typeof error === "object") {
      // If error is an object with a message property
      if (error.message) {
        return error.message;
      }
      // Try to convert object to string representation
      try {
        return JSON.stringify(error);
      } catch (e) {
        return "An unknown error occurred";
      }
    }
    return "An unknown error occurred";
  };

  // Confirmation Modal Component
  const ConfirmationModal = ({ onClose, onConfirm }) => (
    <div className="modal-overlay">
      <div className="modal-content">
        <h3>Hotel Requirements Not Generated</h3>
        <p>
          You haven't completed the Chat2Plan process to generate custom hotel
          requirements. Without this, your layout will use default requirements
          which may not meet your specific needs.
        </p>
        <p>
          Would you like to continue with default requirements or return to
          complete the Chat2Plan process?
        </p>
        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onClose}>
            Return to Chat2Plan
          </button>
          <button className="btn btn-primary" onClick={onConfirm}>
            Continue Anyway
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="container">
      <div className="config-form">
        <h2>Hotel Configuration Generator</h2>
        <p className="helper-text">
          Enter your hotel design requirements to generate building envelope and
          program configurations.
        </p>

        {error && (
          <div className="error-message">
            <h3>Error</h3>
            <p>{getErrorMessage()}</p>
          </div>
        )}

        {/* Confirmation Modal */}
        {showConfirmation && (
          <ConfirmationModal
            onClose={() => setShowConfirmation(false)}
            onConfirm={generateConfiguration}
          />
        )}

        {!generatedConfigs ? (
          <form onSubmit={handleSubmitClick}>
            <div className="form-section">
              <h2>Basic Information</h2>
              <div className="form-group">
                <label htmlFor="hotel_name">Hotel Name</label>
                <input
                  type="text"
                  id="hotel_name"
                  name="hotel_name"
                  className="form-control"
                  value={formData.hotel_name}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-row">
                <div className="half">
                  <div className="form-group">
                    <label htmlFor="hotel_type">Hotel Type</label>
                    <select
                      id="hotel_type"
                      name="hotel_type"
                      className="form-control"
                      value={formData.hotel_type}
                      onChange={handleChange}
                    >
                      <option value="luxury">Luxury</option>
                      <option value="business">Business</option>
                      <option value="resort">Resort</option>
                      <option value="boutique">Boutique</option>
                      <option value="economy">Economy</option>
                    </select>
                  </div>
                </div>
                <div className="half">
                  <div className="form-group">
                    <label htmlFor="num_rooms">Number of Guest Rooms</label>
                    <input
                      type="number"
                      id="num_rooms"
                      name="num_rooms"
                      className="form-control"
                      value={formData.num_rooms}
                      onChange={handleNumericChange}
                      min="1"
                      required
                    />
                  </div>
                </div>
              </div>
            </div>
            <div className="form-section">
              <h2>Building Envelope</h2>
              <p className="helper-text">
                Define the physical parameters of your hotel building.
              </p>

              {/* Building Dimensions */}
              <div className="form-row">
                <div className="third">
                  <div className="form-group">
                    <label htmlFor="building_width">Width (m)</label>
                    <input
                      type="number"
                      id="building_width"
                      name="building_width"
                      className="form-control"
                      value={formData.building_width}
                      onChange={handleNumericChange}
                      min="0"
                      step="0.1"
                      placeholder="Auto"
                    />
                  </div>
                </div>
                <div className="third">
                  <div className="form-group">
                    <label htmlFor="building_length">Length (m)</label>
                    <input
                      type="number"
                      id="building_length"
                      name="building_length"
                      className="form-control"
                      value={formData.building_length}
                      onChange={handleNumericChange}
                      min="0"
                      step="0.1"
                      placeholder="Auto"
                    />
                  </div>
                </div>
                <div className="third">
                  <div className="form-group">
                    <label htmlFor="building_height">Height (m)</label>
                    <input
                      type="number"
                      id="building_height"
                      name="building_height"
                      className="form-control"
                      value={formData.building_height}
                      onChange={handleNumericChange}
                      min="0"
                      step="0.1"
                      placeholder="Auto"
                    />
                  </div>
                </div>
              </div>

              {/* Floor Configuration */}
              <div className="form-row">
                <div className="third">
                  <div className="form-group">
                    <label htmlFor="min_floor">Lowest Floor</label>
                    <input
                      type="number"
                      id="min_floor"
                      name="min_floor"
                      className="form-control"
                      value={formData.min_floor}
                      onChange={handleNumericChange}
                      placeholder="-2"
                    />
                  </div>
                </div>
                <div className="third">
                  <div className="form-group">
                    <label htmlFor="max_floor">Highest Floor</label>
                    <input
                      type="number"
                      id="max_floor"
                      name="max_floor"
                      className="form-control"
                      value={formData.max_floor}
                      onChange={handleNumericChange}
                      placeholder="20"
                    />
                  </div>
                </div>
                <div className="third">
                  <div className="form-group">
                    <label htmlFor="floor_height">Floor Height (m)</label>
                    <input
                      type="number"
                      id="floor_height"
                      name="floor_height"
                      className="form-control"
                      value={formData.floor_height}
                      onChange={handleNumericChange}
                      min="2.5"
                      step="0.1"
                      required
                    />
                  </div>
                </div>
              </div>

              {/* Structural Grid */}
              <div className="form-row">
                <div className="third">
                  <div className="form-group">
                    <label htmlFor="structural_grid_x">
                      Structural Grid X (m)
                    </label>
                    <input
                      type="number"
                      id="structural_grid_x"
                      name="structural_grid_x"
                      className="form-control"
                      value={formData.structural_grid_x}
                      onChange={handleNumericChange}
                      placeholder="8"
                      min="0"
                      step="0.1"
                    />
                  </div>
                </div>
                <div className="third">
                  <div className="form-group">
                    <label htmlFor="structural_grid_y">
                      Structural Grid Y (m)
                    </label>
                    <input
                      type="number"
                      id="structural_grid_y"
                      name="structural_grid_y"
                      className="form-control"
                      value={formData.structural_grid_y}
                      onChange={handleNumericChange}
                      placeholder="8"
                      min="0"
                      step="0.1"
                    />
                  </div>
                </div>
                <div className="third">
                  <div className="form-group">
                    <label htmlFor="grid_size">
                      Grid Size (m){" "}
                      <span className="tooltip-text">
                        (Spatial granularity for room placement)
                      </span>
                    </label>
                    <input
                      type="number"
                      id="grid_size"
                      name="grid_size"
                      className="form-control"
                      value={formData.grid_size}
                      onChange={handleNumericChange}
                      placeholder="1"
                      min="0.1"
                      step="0.1"
                    />
                  </div>
                </div>
              </div>

              {/* Podium Configuration */}
              <h3 className="subsection-title">Podium Configuration</h3>
              <div className="form-row">
                <div className="half">
                  <div className="form-group">
                    <label htmlFor="podium_min_floor">Podium Min Floor</label>
                    <input
                      type="number"
                      id="podium_min_floor"
                      name="podium_min_floor"
                      className="form-control"
                      value={formData.podium_min_floor}
                      onChange={handleNumericChange}
                      placeholder="-2"
                    />
                  </div>
                </div>
                <div className="half">
                  <div className="form-group">
                    <label htmlFor="podium_max_floor">Podium Max Floor</label>
                    <input
                      type="number"
                      id="podium_max_floor"
                      name="podium_max_floor"
                      className="form-control"
                      value={formData.podium_max_floor}
                      onChange={handleNumericChange}
                      placeholder="1"
                    />
                  </div>
                </div>
              </div>
            </div>
            <div className="form-section">
              <h2>Special Requirements</h2>
              <div className="form-group">
                <label htmlFor="special_requirements">
                  Additional Requirements or Constraints
                </label>

                {/* Chat2Plan Interface Container */}
                <div className="chat-interface-container">
                  <p className="helper-text">
                    Use the chat interface below to describe your specific
                    requirements. The AI will help identify key design
                    constraints for your hotel project.
                  </p>

                  <Chat2PlanInterface
                    onRequirementsUpdate={handleRequirementsUpdate}
                    onSessionStart={handleSessionStart}
                    initialContext={formData} // Pass the current form data as context
                  />
                </div>
              </div>
            </div>
            <div className="form-actions">
              <button type="submit" className="btn-primary" disabled={loading}>
                {loading ? "Generating..." : "Generate Configuration"}
              </button>
            </div>
          </form>
        ) : (
          <div className="result-section">
            <h2>Configuration Generated Successfully!</h2>

            <div className="result-container">
              <h3>Building Envelope</h3>
              <p>Filename: {generatedConfigs.building_envelope.filename}</p>
              <div className="json-preview">
                <pre>
                  {JSON.stringify(
                    generatedConfigs.building_envelope.data,
                    null,
                    2
                  )}
                </pre>
              </div>
            </div>

            <div className="result-container">
              <h3>Hotel Requirements</h3>
              <p>Filename: {generatedConfigs.hotel_requirements.filename}</p>
              <div className="json-preview">
                <pre>
                  {JSON.stringify(
                    generatedConfigs.hotel_requirements.data,
                    null,
                    2
                  )}
                </pre>
              </div>
            </div>

            <div className="form-actions">
              <button className="btn-primary" onClick={proceedToInteractive}>
                Proceed to Layout Design
              </button>
            </div>
          </div>
        )}
      </div>

      {/* CSS for modal */}
      <style>{`
        .modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background-color: rgba(0, 0, 0, 0.5);
          display: flex;
          justify-content: center;
          align-items: center;
          z-index: 1000;
        }

        .modal-content {
          background-color: white;
          padding: 2rem;
          border-radius: 8px;
          max-width: 500px;
          width: 90%;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        }

        .modal-content h3 {
          margin-top: 0;
          color: #d9534f;
        }

        .modal-actions {
          display: flex;
          justify-content: space-between;
          margin-top: 1.5rem;
        }

        .btn {
          padding: 0.75rem 1.25rem;
          border-radius: 4px;
          border: none;
          cursor: pointer;
          font-weight: 500;
        }

        .btn-secondary {
          background-color: #6c757d;
          color: white;
        }

        .btn-primary {
          background-color: #3B71CA;
          color: white;
        }
      `}</style>
    </div>
  );
};

export default ConfigGenerator;
