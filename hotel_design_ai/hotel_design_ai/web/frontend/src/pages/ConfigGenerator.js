import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { generateConfigs } from "../services/api";
import Chat2PlanInterface from "../components/Chat2PlanInterface";
import "../styles/ConfigGenerator.css";

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

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await generateConfigs(formData);
      if (response.success) {
        setGeneratedConfigs(response);
      } else {
        setError(response.error || "Failed to generate configurations");
      }
    } catch (err) {
      setError("Error generating configurations: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const proceedToInteractive = () => {
    // In a real app, we would use the actual building and program IDs
    const buildingId =
      generatedConfigs?.building_envelope?.filename.replace(".json", "") ||
      "default";
    const programId =
      generatedConfigs?.hotel_requirements?.filename.replace(".json", "") ||
      "default";

    navigate(`/interactive/${buildingId}/${programId}`);
  };

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
            <p>{error}</p>
          </div>
        )}

        {!generatedConfigs ? (
          <form onSubmit={handleSubmit}>
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
                    onRequirementsUpdate={(requirements) => {
                      setFormData({
                        ...formData,
                        special_requirements: requirements,
                      });
                    }}
                    initialContext={formData} // Pass the current form data as context
                  />
                </div>

                {/* Requirements Text Area */}
                <textarea
                  id="special_requirements"
                  name="special_requirements"
                  className="form-control mt-3"
                  value={formData.special_requirements}
                  onChange={handleChange}
                  rows="4"
                  placeholder="Requirements generated from the chat above will appear here. You can also edit directly."
                ></textarea>
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
    </div>
  );
};

export default ConfigGenerator;
