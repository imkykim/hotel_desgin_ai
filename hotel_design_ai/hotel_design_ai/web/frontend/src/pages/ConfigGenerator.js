import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { generateConfigs } from "../services/api";

const ConfigGenerator = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    hotel_name: "",
    hotel_type: "luxury",
    num_rooms: 100,
    building_width: "",
    building_length: "",
    building_height: "",
    num_floors: "",
    num_basement_floors: 1,
    floor_height: 4.5,
    has_restaurant: true,
    has_meeting_rooms: true,
    has_ballroom: false,
    has_pool: false,
    has_gym: true,
    has_spa: false,
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
              <h2>Building Envelope (Optional)</h2>
              <p className="helper-text">
                Leave fields blank to use recommended values based on hotel type
                and size.
              </p>

              <div className="form-row">
                <div className="half">
                  <div className="form-group">
                    <label htmlFor="building_width">Building Width (m)</label>
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
                <div className="half">
                  <div className="form-group">
                    <label htmlFor="building_length">Building Length (m)</label>
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
              </div>

              <div className="form-row">
                <div className="half">
                  <div className="form-group">
                    <label htmlFor="building_height">Building Height (m)</label>
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
                <div className="half">
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

              <div className="form-row">
                <div className="half">
                  <div className="form-group">
                    <label htmlFor="num_floors">Number of Floors</label>
                    <input
                      type="number"
                      id="num_floors"
                      name="num_floors"
                      className="form-control"
                      value={formData.num_floors}
                      onChange={handleNumericChange}
                      min="1"
                      placeholder="Auto"
                    />
                  </div>
                </div>
                <div className="half">
                  <div className="form-group">
                    <label htmlFor="num_basement_floors">
                      Number of Basement Floors
                    </label>
                    <input
                      type="number"
                      id="num_basement_floors"
                      name="num_basement_floors"
                      className="form-control"
                      value={formData.num_basement_floors}
                      onChange={handleNumericChange}
                      min="0"
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="form-section">
              <h2>Facilities</h2>
              <div className="checkbox-group">
                <div className="checkbox-item">
                  <input
                    type="checkbox"
                    id="has_restaurant"
                    name="has_restaurant"
                    checked={formData.has_restaurant}
                    onChange={handleChange}
                  />
                  <label htmlFor="has_restaurant">Restaurant</label>
                </div>
                <div className="checkbox-item">
                  <input
                    type="checkbox"
                    id="has_meeting_rooms"
                    name="has_meeting_rooms"
                    checked={formData.has_meeting_rooms}
                    onChange={handleChange}
                  />
                  <label htmlFor="has_meeting_rooms">Meeting Rooms</label>
                </div>
                <div className="checkbox-item">
                  <input
                    type="checkbox"
                    id="has_ballroom"
                    name="has_ballroom"
                    checked={formData.has_ballroom}
                    onChange={handleChange}
                  />
                  <label htmlFor="has_ballroom">Ballroom</label>
                </div>
                <div className="checkbox-item">
                  <input
                    type="checkbox"
                    id="has_pool"
                    name="has_pool"
                    checked={formData.has_pool}
                    onChange={handleChange}
                  />
                  <label htmlFor="has_pool">Swimming Pool</label>
                </div>
                <div className="checkbox-item">
                  <input
                    type="checkbox"
                    id="has_gym"
                    name="has_gym"
                    checked={formData.has_gym}
                    onChange={handleChange}
                  />
                  <label htmlFor="has_gym">Fitness Center</label>
                </div>
                <div className="checkbox-item">
                  <input
                    type="checkbox"
                    id="has_spa"
                    name="has_spa"
                    checked={formData.has_spa}
                    onChange={handleChange}
                  />
                  <label htmlFor="has_spa">Spa</label>
                </div>
              </div>
            </div>

            <div className="form-section">
              <h2>Special Requirements</h2>
              <div className="form-group">
                <label htmlFor="special_requirements">
                  Additional Requirements or Constraints
                </label>
                <textarea
                  id="special_requirements"
                  name="special_requirements"
                  className="form-control"
                  value={formData.special_requirements}
                  onChange={handleChange}
                  rows="4"
                  placeholder="Enter any special requirements or constraints for your hotel design..."
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
