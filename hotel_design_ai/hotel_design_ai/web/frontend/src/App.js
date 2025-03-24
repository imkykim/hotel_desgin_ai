import React, { useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useNavigate,
} from "react-router-dom";
import "./App.css";

// Import the InteractiveLayoutPage component
import InteractiveLayoutPage from "./pages/InteractiveLayoutPage";

// Configuration Generator Component
function ConfigGenerator() {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const [formData, setFormData] = useState({
    hotel_name: "",
    hotel_type: "business",
    num_rooms: 150,
    building_width: "",
    building_length: "",
    building_height: "",
    num_floors: "",
    num_basement_floors: "",
    floor_height: "",
    has_restaurant: true,
    has_meeting_rooms: true,
    has_ballroom: false,
    has_pool: false,
    has_gym: true,
    has_spa: false,
    special_requirements: "",
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === "checkbox" ? checked : value,
    });
  };

  const handleNumberChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value === "" ? "" : Number(value),
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("http://localhost:8000/generate-configs", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Error generating configurations");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Function to navigate to the interactive design page
  const handleContinueToInteractive = () => {
    if (result) {
      // Extract building and program IDs from the filenames
      const buildingId = result.building_envelope.filename.replace(".json", "");
      const programId = result.hotel_requirements.filename.replace(".json", "");

      // Navigate to the interactive layout page
      navigate(`/interactive/${buildingId}/${programId}`);
    }
  };

  const goToInteractiveDirect = () => {
    navigate("/interactive/sample/sample");
  };

  const hotelTypes = [
    "budget",
    "economy",
    "midscale",
    "upscale",
    "luxury",
    "boutique",
    "resort",
    "business",
    "extended_stay",
    "all_inclusive",
  ];

  return (
    <div className="App">
      <header className="App-header">
        <h1>Hotel Design AI Configuration Generator</h1>
        <p>
          Generate building envelope and hotel requirements JSON configurations
          using AI
        </p>
        <button
          onClick={goToInteractiveDirect}
          style={{
            marginTop: "10px",
            padding: "5px 15px",
            backgroundColor: "#28a745",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            fontSize: "14px",
          }}
        >
          Skip to Interactive Design Demo
        </button>
      </header>

      <main className="container">
        <form onSubmit={handleSubmit} className="config-form">
          <div className="form-section">
            <h2>Basic Hotel Information</h2>

            <div className="form-group">
              <label htmlFor="hotel_name">Hotel Name:</label>
              <input
                type="text"
                id="hotel_name"
                name="hotel_name"
                value={formData.hotel_name}
                onChange={handleChange}
                required
                className="form-control"
              />
            </div>

            <div className="form-group">
              <label htmlFor="hotel_type">Hotel Type:</label>
              <select
                id="hotel_type"
                name="hotel_type"
                value={formData.hotel_type}
                onChange={handleChange}
                required
                className="form-control"
              >
                {hotelTypes.map((type) => (
                  <option key={type} value={type}>
                    {type
                      .split("_")
                      .map(
                        (word) => word.charAt(0).toUpperCase() + word.slice(1)
                      )
                      .join(" ")}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="num_rooms">Number of Guest Rooms:</label>
              <input
                type="number"
                id="num_rooms"
                name="num_rooms"
                value={formData.num_rooms}
                onChange={handleNumberChange}
                required
                min="1"
                className="form-control"
              />
            </div>
          </div>

          <div className="form-section">
            <h2>Building Envelope Parameters (Optional)</h2>
            <p className="helper-text">
              Leave blank to let AI determine appropriate values
            </p>

            <div className="form-row">
              <div className="form-group half">
                <label htmlFor="building_width">Building Width (m):</label>
                <input
                  type="number"
                  id="building_width"
                  name="building_width"
                  value={formData.building_width}
                  onChange={handleNumberChange}
                  min="10"
                  className="form-control"
                />
              </div>

              <div className="form-group half">
                <label htmlFor="building_length">Building Length (m):</label>
                <input
                  type="number"
                  id="building_length"
                  name="building_length"
                  value={formData.building_length}
                  onChange={handleNumberChange}
                  min="10"
                  className="form-control"
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group half">
                <label htmlFor="building_height">Building Height (m):</label>
                <input
                  type="number"
                  id="building_height"
                  name="building_height"
                  value={formData.building_height}
                  onChange={handleNumberChange}
                  min="3"
                  className="form-control"
                />
              </div>

              <div className="form-group half">
                <label htmlFor="floor_height">Floor Height (m):</label>
                <input
                  type="number"
                  id="floor_height"
                  name="floor_height"
                  value={formData.floor_height}
                  onChange={handleNumberChange}
                  min="2.5"
                  step="0.1"
                  className="form-control"
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group half">
                <label htmlFor="num_floors">Number of Floors:</label>
                <input
                  type="number"
                  id="num_floors"
                  name="num_floors"
                  value={formData.num_floors}
                  onChange={handleNumberChange}
                  min="1"
                  className="form-control"
                />
              </div>

              <div className="form-group half">
                <label htmlFor="num_basement_floors">
                  Number of Basement Floors:
                </label>
                <input
                  type="number"
                  id="num_basement_floors"
                  name="num_basement_floors"
                  value={formData.num_basement_floors}
                  onChange={handleNumberChange}
                  min="0"
                  className="form-control"
                />
              </div>
            </div>
          </div>

          <div className="form-section">
            <h2>Hotel Facilities</h2>

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
                Any special requirements or constraints:
              </label>
              <textarea
                id="special_requirements"
                name="special_requirements"
                value={formData.special_requirements}
                onChange={handleChange}
                rows="4"
                className="form-control"
              />
            </div>
          </div>

          <div className="form-actions">
            <button type="submit" className="btn-primary" disabled={isLoading}>
              {isLoading ? "Generating..." : "Generate Configurations"}
            </button>
          </div>
        </form>

        {error && (
          <div className="error-message">
            <h3>Error</h3>
            <p>{error}</p>
          </div>
        )}

        {result && (
          <div className="result-section">
            <h2>Generated Configurations</h2>

            <div className="result-container">
              <h3>Building Envelope Configuration</h3>
              <p>Saved to: {result.building_envelope.path}</p>
              <pre>
                {JSON.stringify(result.building_envelope.data, null, 2)}
              </pre>

              <h3>Hotel Requirements Configuration</h3>
              <p>Saved to: {result.hotel_requirements.path}</p>
              <div className="json-preview">
                <pre>
                  {JSON.stringify(result.hotel_requirements.data, null, 2)}
                </pre>
              </div>
            </div>

            <div className="next-actions">
              <button
                className="btn-interactive"
                onClick={handleContinueToInteractive}
                disabled={!result}
                style={{
                  marginTop: "20px",
                  padding: "12px 24px",
                  backgroundColor: "#28a745",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                  fontSize: "16px",
                  fontWeight: "bold",
                }}
              >
                Continue to Interactive Design
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

// Main App component with routing
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<ConfigGenerator />} />
        <Route path="/configure" element={<ConfigGenerator />} />
        <Route
          path="/interactive/:buildingId/:programId"
          element={<InteractiveLayoutPage />}
        />
      </Routes>
    </Router>
  );
}

export default App;
