import React, { useState, useEffect, useRef } from "react";
import {
  startChat2PlanSession,
  sendChat2PlanMessage,
  getChat2PlanState,
  skipStage,
} from "../services/api";
import "../styles/Chat2PlanInterface.css";

const Chat2PlanInterface = ({ initialContext, onRequirementsUpdate }) => {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [currentStage, setCurrentStage] = useState(null);
  const [keyQuestions, setKeyQuestions] = useState([]);
  const messagesEndRef = useRef(null);
  const [visualizations, setVisualizations] = useState({
    roomGraph: null,
    constraintsTable: null,
    layout: null,
  });

  // Initialize the session when the component mounts
  useEffect(() => {
    initializeSession();
  }, []);

  // Scroll to bottom of messages when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Poll for state changes periodically
  useEffect(() => {
    if (sessionId) {
      const interval = setInterval(() => {
        refreshState();
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [sessionId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const initializeSession = async () => {
    setIsLoading(true);
    try {
      const response = await startChat2PlanSession({
        context: initialContext || {},
      });

      if (response.session_id) {
        setSessionId(response.session_id);
        setMessages([
          {
            role: "system",
            content:
              "Welcome! Please describe any special design requirements or constraints for your hotel project.",
          },
        ]);

        // Get initial state
        refreshState();
      } else if (response.error) {
        console.error("Failed to initialize chat session:", response.error);
        setMessages([
          {
            role: "system",
            content: `Failed to initialize chat: ${response.error}`,
          },
        ]);
      }
    } catch (error) {
      console.error("Failed to initialize chat session:", error);
      setMessages([
        {
          role: "system",
          content: "Failed to initialize chat. Please try again.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const refreshState = async () => {
    if (!sessionId) return;

    try {
      const state = await getChat2PlanState(sessionId);

      if (state.error) {
        console.error("Error getting state:", state.error);
        return;
      }

      // Update current stage
      if (state.current_stage !== currentStage) {
        setCurrentStage(state.current_stage);
      }

      // Update key questions
      if (state.key_questions) {
        setKeyQuestions(state.key_questions);
      }

      // Update the user requirements in the parent component
      if (state.user_requirement_guess && onRequirementsUpdate) {
        onRequirementsUpdate(state.user_requirement_guess);
      }

      // Check for visualizations if we're in later stages
      if (
        state.current_stage &&
        state.current_stage !== "STAGE_REQUIREMENT_GATHERING"
      ) {
        // This is where we would check for visualizations in the original app
        // For simplicity, we're not implementing this part in this example
      }
    } catch (error) {
      console.error("Error refreshing state:", error);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || !sessionId) return;

    const userMessage = input.trim();
    setInput("");

    // Add user message to chat
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);

    setIsLoading(true);
    try {
      const response = await sendChat2PlanMessage(sessionId, userMessage);

      if (response.error) {
        console.error("Failed to send message:", response.error);
        setMessages((prev) => [
          ...prev,
          {
            role: "system",
            content: `Error: ${response.error}`,
          },
        ]);
      } else {
        // Add system response to chat
        setMessages((prev) => [
          ...prev,
          { role: "system", content: response.response },
        ]);

        // Update the requirements in the parent component
        if (response.requirements && onRequirementsUpdate) {
          onRequirementsUpdate(response.requirements);
        }

        // Check if stage changed
        if (response.stage_change) {
          setCurrentStage(response.current_stage);
          refreshState(); // Refresh state after stage change
        }
      }
    } catch (error) {
      console.error("Failed to send message:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          content: "Error: Failed to process your message. Please try again.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Find the skipStage function in your component
  const handleSkipStage = async () => {
    if (!sessionId) return;

    setIsLoading(true);
    try {
      const response = await skipStage(sessionId);

      if (response.error) {
        console.error("Failed to skip stage:", response.error);
        return;
      }

      // Update current stage
      setCurrentStage(response.current_stage);

      // Add message to chat
      setMessages((prev) => [
        ...prev,
        { role: "system", content: "Skipping to next stage..." },
      ]);

      // Refresh state
      await refreshState();
    } catch (error) {
      console.error("Error skipping stage:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // Add this function inside your Chat2PlanInterface component
  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault(); // Prevent default to avoid new line
      sendMessage();
    }
  };

  return (
    <div className="chat2plan-interface">
      {/* Stage indicator */}
      {currentStage && (
        <div className="stage-indicator">
          <span className="stage-label">{currentStage}</span>
          {currentStage !== "STAGE_REQUIREMENT_GATHERING" && (
            <button
              className="skip-button"
              onClick={handleSkipStage}
              disabled={isLoading}
            >
              Skip Stage
            </button>
          )}
        </div>
      )}

      {/* Chat messages container */}
      <div className="chat-messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            {message.content}
          </div>
        ))}
        {isLoading && (
          <div className="message system">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Chat input area */}
      <div className="chat-input">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your design requirements here..."
          disabled={isLoading}
          rows={2}
        />
        <button onClick={sendMessage} disabled={isLoading || !input.trim()}>
          Send
        </button>
      </div>

      {/* Key questions panel (collapsed by default) */}
      {keyQuestions.length > 0 && (
        <div className="key-questions-panel">
          <div className="panel-header">
            <h4>Key Questions</h4>
            <span className="toggle-icon">▼</span>
          </div>
          <div className="panel-content">
            <table>
              <thead>
                <tr>
                  <th>Category</th>
                  <th>Status</th>
                  <th>Details</th>
                </tr>
              </thead>
              <tbody>
                {keyQuestions.map((question, index) => (
                  <tr key={index}>
                    <td>{question.category}</td>
                    <td
                      className={
                        question.status === "已知"
                          ? "status-known"
                          : "status-unknown"
                      }
                    >
                      {question.status}
                    </td>
                    <td>{question.details || ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Visualizations panel (would show room graph, constraints, etc.) */}
      {(visualizations.roomGraph ||
        visualizations.constraintsTable ||
        visualizations.layout) && (
        <div className="visualizations-panel">
          <div className="panel-header">
            <h4>Visualizations</h4>
            <span className="toggle-icon">▼</span>
          </div>
          <div className="panel-content">
            {/* Visualizations would go here */}
            {visualizations.roomGraph && (
              <div className="visualization">
                <h5>Room Graph</h5>
                <img src={visualizations.roomGraph} alt="Room Graph" />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Chat2PlanInterface;
