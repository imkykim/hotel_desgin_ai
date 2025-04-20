import React, { useState } from "react";
import {
  startChat2PlanSession,
  sendChat2PlanMessage,
  getChat2PlanState,
} from "../services/api";

const Chat2PlanInterface = ({ initialContext, onRequirementsUpdate }) => {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const initializeSession = async () => {
    setIsLoading(true);
    try {
      const response = await startChat2PlanSession(initialContext);

      if (response.session_id) {
        setSessionId(response.session_id);

        // Add welcome message
        setMessages([
          {
            role: "system",
            content:
              "Welcome! Please describe any special design requirements or constraints for your hotel project.",
          },
        ]);
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
        if (response.requirements) {
          onRequirementsUpdate(response.requirements);
        }

        // Check if we should refresh state (stage changes, etc.)
        if (response.stage_change) {
          refreshState();
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

  const refreshState = async () => {
    if (!sessionId) return;

    try {
      const response = await getChat2PlanState(sessionId);

      if (response.error) {
        console.error("Failed to refresh state:", response.error);
      } else if (response.user_requirement_guess) {
        onRequirementsUpdate(response.user_requirement_guess);
      }
    } catch (error) {
      console.error("Failed to refresh state:", error);
    }
  };

  return (
    <div>
      <div>
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            {message.content}
          </div>
        ))}
      </div>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        disabled={isLoading}
      />
      <button onClick={sendMessage} disabled={isLoading}>
        Send
      </button>
    </div>
  );
};

export default Chat2PlanInterface;
