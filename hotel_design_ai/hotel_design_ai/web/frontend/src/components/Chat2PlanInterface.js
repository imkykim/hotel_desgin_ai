import React, { useState, useEffect, useRef } from "react";
import {
  startChat2PlanSession,
  sendChat2PlanMessage,
  getChat2PlanState,
  skipStage,
  exportRequirements,
  getChat2PlanLogs,
} from "../services/api";
import "../styles/Chat2PlanInterface.css";

const Chat2PlanInterface = ({
  initialContext,
  onRequirementsUpdate,
  onSessionStart,
  onRequirementsReady,
}) => {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [currentStage, setCurrentStage] = useState(null);
  const [keyQuestions, setKeyQuestions] = useState([]);
  const messagesEndRef = useRef(null);
  const [requirementsGenerated, setRequirementsGenerated] = useState(false);
  const [isGeneratingRequirements, setIsGeneratingRequirements] =
    useState(false);
  const [checkingRequirementsInterval, setCheckingRequirementsInterval] =
    useState(null);
  const [visualizations, setVisualizations] = useState({
    roomGraph: null,
    constraintsTable: null,
    layout: null,
  });
  const [logs, setLogs] = useState([]);
  const [showLogs, setShowLogs] = useState(false);
  const [backendLogs, setBackendLogs] = useState([]); // new: backend logs
  const [backendLogTotal, setBackendLogTotal] = useState(0); // new: backend log index

  // Helper to log to both UI and browser console
  const logMessage = (msg, type = "INFO") => {
    const timestamp = new Date().toLocaleString();
    const formatted = `[${timestamp}] ${type}: ${msg}`;
    if (type === "ERROR") {
      // eslint-disable-next-line no-console
      console.error(formatted);
    } else {
      // eslint-disable-next-line no-console
      console.log(formatted);
    }
    setLogs((prev) => [...prev, formatted]);
  };

  // Initialize the session when the component mounts
  useEffect(() => {
    initializeSession();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Check for requirements file periodically after constraint generation stage
  useEffect(() => {
    if (
      sessionId &&
      (currentStage === "STAGE_CONSTRAINT_VISUALIZATION" ||
        currentStage === "STAGE_CONSTRAINT_REFINEMENT") &&
      !requirementsGenerated &&
      !isGeneratingRequirements
    ) {
      const interval = setInterval(checkRequirementsFile, 5000);
      setCheckingRequirementsInterval(interval);
      return () => clearInterval(interval);
    } else if (requirementsGenerated && checkingRequirementsInterval) {
      clearInterval(checkingRequirementsInterval);
      setCheckingRequirementsInterval(null);
    }
  }, [
    sessionId,
    currentStage,
    requirementsGenerated,
    isGeneratingRequirements,
    checkingRequirementsInterval,
  ]);

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

  // In your Chat2PlanInterface.js
  // Find the useEffect hook that polls logs and update it:

  useEffect(() => {
    if (!sessionId) return;

    let logPoller = null;
    let isUnmounted = false;

    const pollLogs = async () => {
      try {
        const result = await getChat2PlanLogs(sessionId, backendLogTotal);
        if (!isUnmounted && result.logs && result.logs.length > 0) {
          setBackendLogs((prev) => [...prev, ...result.logs]);
          setBackendLogTotal((prev) => prev + result.logs.length);

          // Scroll to bottom of log container when new logs appear
          if (logViewerRef.current) {
            logViewerRef.current.scrollTop = logViewerRef.current.scrollHeight;
          }
        }
      } catch (e) {
        console.error("Error fetching logs:", e);
      }
    };

    // Poll more frequently for better real-time experience
    logPoller = setInterval(pollLogs, 1000); // Poll every second

    // Initial fetch
    pollLogs();

    return () => {
      isUnmounted = true;
      if (logPoller) clearInterval(logPoller);
    };
  }, [sessionId, backendLogTotal]);

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
        const newSessionId = response.session_id;
        setSessionId(newSessionId);

        // Call the onSessionStart callback if provided
        if (onSessionStart) onSessionStart(newSessionId);

        setMessages([
          {
            role: "system",
            content:
              "Welcome! Please describe any special design requirements or constraints for your hotel project.",
          },
        ]);
        logMessage(`Chat2Plan session started: ${newSessionId}`);

        // Get initial state
        refreshState();
      } else if (response.error) {
        logMessage(
          `Failed to initialize chat session: ${response.error}`,
          "ERROR"
        );
        setMessages([
          {
            role: "system",
            content: `Failed to initialize chat: ${response.error}`,
          },
        ]);
      }
    } catch (error) {
      logMessage(`Failed to initialize chat session: ${error}`, "ERROR");
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

  const checkRequirementsFile = async () => {
    if (!sessionId || requirementsGenerated || isGeneratingRequirements) return;

    setIsGeneratingRequirements(true);
    try {
      const result = await exportRequirements(sessionId);

      if (result.success) {
        setRequirementsGenerated(true);

        // Notify parent that requirements are ready
        if (onRequirementsReady) onRequirementsReady(result);

        // Add a message about requirements being ready
        setMessages((prev) => [
          ...prev,
          {
            role: "system",
            content:
              "🎉 Hotel requirements have been successfully generated and are ready to use for layout generation! You can now proceed to generate the configuration.",
          },
        ]);
        logMessage("Hotel requirements file generated and ready.");

        // Clear the interval since we found the file
        if (checkingRequirementsInterval) {
          clearInterval(checkingRequirementsInterval);
          setCheckingRequirementsInterval(null);
        }
        return true;
      }
    } catch (error) {
      logMessage(`Error checking requirements file: ${error}`, "ERROR");
    } finally {
      setIsGeneratingRequirements(false);
    }
    return false;
  };

  // Only update special_requirements in parent, not the whole form
  const safeOnRequirementsUpdate = (requirements) => {
    if (onRequirementsUpdate) {
      onRequirementsUpdate(requirements);
    }
  };

  const refreshState = async () => {
    if (!sessionId) return;

    try {
      const state = await getChat2PlanState(sessionId);

      if (state.error) {
        logMessage(`Error getting state: ${state.error}`, "ERROR");
        return;
      }

      // Update current stage
      if (state.current_stage !== currentStage) {
        setCurrentStage(state.current_stage);

        logMessage(`Stage changed: ${state.current_stage}`);

        // If we just moved to constraint visualization or refinement stage,
        // add a system message informing the user
        if (
          state.current_stage === "STAGE_CONSTRAINT_VISUALIZATION" ||
          state.current_stage === "STAGE_CONSTRAINT_REFINEMENT"
        ) {
          if (currentStage === "STAGE_CONSTRAINT_GENERATION") {
            setMessages((prev) => [
              ...prev,
              {
                role: "system",
                content:
                  "✅ Constraint generation complete! Processing requirements file...",
              },
            ]);
            logMessage(
              "Constraint generation complete, processing requirements file."
            );

            // Trigger immediate check for requirements file
            checkRequirementsFile();
          }
        }
      }

      // Update key questions
      if (state.key_questions) {
        setKeyQuestions(state.key_questions);
      }

      // Update the user requirements in the parent component
      if (state.user_requirement_guess && onRequirementsUpdate) {
        safeOnRequirementsUpdate(state.user_requirement_guess);
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
      logMessage(`Error refreshing state: ${error}`, "ERROR");
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || !sessionId) return;

    const userMessage = input.trim();
    setInput("");

    // Add user message to chat
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    logMessage(`User: ${userMessage}`);

    setIsLoading(true);
    try {
      const response = await sendChat2PlanMessage(sessionId, userMessage);

      if (response.error) {
        logMessage(`Failed to send message: ${response.error}`, "ERROR");
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
        logMessage(`System: ${response.response}`);

        // Only update special_requirements in parent, not the whole form
        if (response.requirements && onRequirementsUpdate) {
          safeOnRequirementsUpdate(response.requirements);
        }

        // Check if stage changed
        if (response.stage_change) {
          setCurrentStage(response.current_stage);
          logMessage(`Stage changed: ${response.current_stage}`);
          refreshState(); // Refresh state after stage change
        }
      }
    } catch (error) {
      logMessage(`Failed to send message: ${error}`, "ERROR");
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

  const handleSkipStage = async () => {
    if (!sessionId) return;

    setIsLoading(true);
    try {
      const response = await skipStage(sessionId);

      if (response.error) {
        logMessage(`Failed to skip stage: ${response.error}`, "ERROR");
        return;
      }

      // Update current stage
      setCurrentStage(response.current_stage);
      logMessage(`Stage skipped. Now at: ${response.current_stage}`);

      // Add message to chat
      setMessages((prev) => [
        ...prev,
        { role: "system", content: "Skipping to next stage..." },
      ]);

      // If we're moving to constraint visualization or refinement, check for requirements
      if (
        response.current_stage === "STAGE_CONSTRAINT_VISUALIZATION" ||
        response.current_stage === "STAGE_CONSTRAINT_REFINEMENT"
      ) {
        setMessages((prev) => [
          ...prev,
          {
            role: "system",
            content:
              "✅ Constraint generation complete! Processing requirements file...",
          },
        ]);
        logMessage(
          "Constraint generation complete, processing requirements file."
        );

        // Trigger immediate check for requirements file
        checkRequirementsFile();
      }

      // Refresh state
      await refreshState();
    } catch (error) {
      logMessage(`Error skipping stage: ${error}`, "ERROR");
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault(); // Prevent default to avoid new line
      sendMessage();
    }
  };

  const logViewerRef = useRef(null);

  // Move the log viewer outside the main chat2plan frame
  // So, do NOT render the log viewer inside the main <div className="chat2plan-interface">
  // Instead, export logs and showLogs/setShowLogs via props if needed, or render the log viewer after Chat2PlanInterface in the parent

  // For simplicity, export the log viewer as a named export
  const LogViewer = (
    <div style={{ marginTop: "1rem" }}>
      <button
        className="btn"
        style={{
          background: "#f5f5f5",
          color: "#333",
          border: "1px solid #ccc",
          marginBottom: "0.5rem",
          fontFamily: "inherit",
          fontSize: "0.95rem",
        }}
        onClick={() => setShowLogs((prev) => !prev)}
      >
        {showLogs ? "Hide Logs" : "Show Logs"}
      </button>
      {showLogs && (
        <div
          ref={logViewerRef}
          style={{
            background: "#222",
            color: "#e0e0e0",
            fontFamily: "monospace",
            fontSize: "0.95rem",
            borderRadius: "6px",
            padding: "1rem",
            maxHeight: "200px",
            overflowY: "auto",
            boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
          }}
          data-testid="log-viewer"
        >
          {(() => {
            // Merge and deduplicate logs
            const allLogs = [...backendLogs, ...logs];
            const seen = new Set();
            const deduped = [];

            for (const log of allLogs) {
              if (!seen.has(log)) {
                seen.add(log);
                deduped.push(log);
              }
            }

            return deduped.length === 0 ? (
              <div style={{ color: "#888" }}>No logs yet.</div>
            ) : (
              deduped.map((log, idx) => {
                // Check if log contains JSON
                let formattedLog = log;
                if (log.includes("```json")) {
                  const jsonStart = log.indexOf("```json") + 7;
                  const jsonEnd = log.indexOf("```", jsonStart);
                  if (jsonEnd > jsonStart) {
                    const jsonStr = log.substring(jsonStart, jsonEnd).trim();
                    try {
                      const jsonObj = JSON.parse(jsonStr);
                      formattedLog =
                        log.substring(0, jsonStart - 7) +
                        `<span style="color: #8be9fd">JSON: ${JSON.stringify(
                          jsonObj,
                          null,
                          2
                        )}</span>` +
                        log.substring(jsonEnd + 3);
                    } catch (e) {
                      // If JSON parsing fails, just show the original log
                    }
                  }
                }

                return (
                  <div
                    key={idx}
                    style={{ marginBottom: "0.25rem" }}
                    dangerouslySetInnerHTML={{ __html: formattedLog }}
                  />
                );
              })
            );
          })()}
        </div>
      )}
    </div>
  );

  // Attach the log viewer to the component so parent can use it
  // (If you want to use it outside, you can do: <Chat2PlanInterface ... ref={ref} /> and then <ref.current.LogViewer />)
  // But for most React usage, it's easier to just return it as a second element in an array

  // Instead of returning a single <div>, return an array: [chat2plan, logviewer]
  return [
    <div className="chat2plan-interface" key="chat2plan">
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

      {/* Add an indicator for requirements status */}
      {requirementsGenerated && (
        <div className="requirements-status success">
          <span className="status-icon">✓</span>
          Requirements generated successfully!
        </div>
      )}

      {isGeneratingRequirements && (
        <div className="requirements-status loading">
          <span className="loading-indicator"></span>
          Generating requirements file...
        </div>
      )}

      {/* Visualizations panel (would show room graph, constraints, etc.) */}
      {(visualizations.roomGraph ||
        visualizations.constraintsTable ||
        visualizations.layout) && (
        <div className="visualizations-panel">
          <div className="panel-header">
            <h4>Visualizations</h4>
            <span className="toggle-icon">▼</span>Z
          </div>
          <div className="panel-content">
            {visualizations.roomGraph && (
              <div className="visualization">
                <h5>Room Graph</h5>
                <img src={visualizations.roomGraph} alt="Room Graph" />
              </div>
            )}
          </div>
        </div>
      )}
    </div>,
    <React.Fragment key="logviewer">
      <div style={{ marginTop: "1rem" }}>
        <button
          className="btn"
          style={{
            background: "#f5f5f5",
            color: "#333",
            border: "1px solid #ccc",
            marginBottom: "0.5rem",
            fontFamily: "inherit",
            fontSize: "0.95rem",
          }}
          onClick={() => setShowLogs((prev) => !prev)}
        >
          {showLogs ? "Hide Logs" : "Show Logs"}
        </button>
        {showLogs && (
          <div
            style={{
              background: "#222",
              color: "#e0e0e0",
              fontFamily: "monospace",
              fontSize: "0.95rem",
              borderRadius: "6px",
              padding: "1rem",
              maxHeight: "200px",
              overflowY: "auto",
              boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
            }}
            data-testid="log-viewer"
          >
            {/* Merge backendLogs and UI logs, deduplicate, show newest last */}
            {(() => {
              // Merge and deduplicate logs (backend first, then UI logs)
              const allLogs = [...backendLogs, ...logs];
              const seen = new Set();
              const deduped = [];
              for (const log of allLogs) {
                if (!seen.has(log)) {
                  seen.add(log);
                  deduped.push(log);
                }
              }
              return deduped.length === 0 ? (
                <div style={{ color: "#888" }}>No logs yet.</div>
              ) : (
                deduped.map((log, idx) => (
                  <div key={idx} style={{ marginBottom: "0.25rem" }}>
                    {log}
                  </div>
                ))
              );
            })()}
          </div>
        )}
      </div>
    </React.Fragment>,
  ];
};

export default Chat2PlanInterface;
